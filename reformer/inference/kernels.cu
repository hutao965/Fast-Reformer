#include "kernels.cuh"

namespace FastReformer {

//TODO multi small fixed block size
// TODO add stream
// TODO fp16
// TODO template for common hidden size (1024 768 and so on)


// TODO half2 version
template<typename T, typename F>
__forceinline__ __device__ T reduce_warp(T value, F reduction) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value = reduction(value, __shfl_xor_sync(0xffffffff, value, mask, 32));
    return value;
}

// instance for argmax
template<typename T, typename F>
__forceinline__ __device__ thrust::pair<T, int> reduce_warp(thrust::pair<T, int> value, F reduction) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value = reduction(value, mask);
    return value;
}

template<typename T, typename F>
__forceinline__ __device__ T reduce_block(T value, F reduction, T null_value) {
    value = reduce_warp(value, reduction);
    __shared__ T s[32];
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    if (laneId == 0) s[warpId] = value;
    __syncthreads();
    value = (laneId < ((blockDim.x + 31) >> 5)) ? s[laneId] : null_value;
    value = reduce_warp(value, reduction);
    return value;
}

template<typename T>
__forceinline__ __device__ T reduce_block_sum(T value) {
    return reduce_block(
        value,
        [](T x, T y){ return x + y; },
        static_cast<T>(0.0f)
    );
}

__forceinline__ __device__ float reduce_block_max(float value) {
    return reduce_block(
        value,
        [](float x, float y){ return max(x, y); },
        -FLT_MAX
    );
}

__forceinline__ __device__ thrust::pair<float, int> reduce_block_argmax(thrust::pair<float, int> value) {
    return reduce_block(
        value,
        [](thrust::pair<float, int> x, int mask){
            float other_first = __shfl_xor_sync(0xffffffff, x.first, mask, 32);
            int other_second = __shfl_xor_sync(0xffffffff, x.second, mask, 32);
            return x.first >= other_first ?
                   x : thrust::make_pair(other_first, other_second);
        },
        thrust::make_pair(-FLT_MAX, 0)
    );
}

// 1, 1, ..., 1 -> 1, 2, ..., 32
template<typename T>
__forceinline__ __device__ T reduce_warp_prefix_sum(T value, int laneId) {
    #pragma unroll
    for (int STRIDE = 1; STRIDE <= 16; STRIDE <<= 1) {
        T temp = __shfl_up_sync(0xffffffff, value, STRIDE, 32);
        value = laneId >= STRIDE ? temp + value : value;
    }
    return value;
}

// prefix sum for radix sort
// 1, 1, ..., 1 -> 0, 1, ..., 127
template<typename T>
__forceinline__ __device__ T reduce_block_prefix_sum(T value) {
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    value = reduce_warp_prefix_sum(value, laneId);
    // sum for each warp
    __shared__ T s[32];
    if (laneId == 31) s[warpId] = value;
    __syncthreads();
    if (warpId == 0) {
        T prefix = s[laneId];
        prefix = reduce_warp_prefix_sum(prefix, laneId);
        s[laneId] = prefix;
    }
    __syncthreads();
    value = __shfl_up_sync(0xffffffff, value, 1, 32);
    if (laneId == 0) value = static_cast<T>(0.0f);
    if (warpId != 0) value += s[warpId - 1];
    return value;
}


template<typename T>
__global__ void encoder_embedding(
    const int *input_ids, const T *tok_embd_weights,
    const T *pos_embd_weight_0, const T *pos_embd_weight_1,
    int pos_embds_dim_0, int pos_embds_dim_1, int pos_shape_0, int pos_shape_1,
    int hidden_size, int pad_id, int start_idx_pos_encodings,
    T *output, int *pad_mask) 
{
    int batch_seq_len = gridDim.x;
    int batch_seq_id = blockIdx.x;
    // int batch_size = gridDim.y;
    int batch_id = blockIdx.y;
    int pos_id = batch_seq_id + start_idx_pos_encodings;

    int input_id = input_ids[batch_id * batch_seq_len + batch_seq_id];
    pad_mask[batch_id * batch_seq_len + batch_seq_id] = static_cast<int>(input_id != pad_id);
    // tok embd
    T embd = tok_embd_weights[input_id * hidden_size + threadIdx.x];
    // axial pos embd
    embd += threadIdx.x < pos_embds_dim_0 ?
        pos_embd_weight_0[(pos_id / pos_shape_1) * pos_embds_dim_0 + threadIdx.x] :
        pos_embd_weight_1[(pos_id % pos_shape_1) * pos_embds_dim_1 + threadIdx.x - pos_embds_dim_0];
    // store
    output[
        batch_id * batch_seq_len * hidden_size +
        batch_seq_id * hidden_size +
        threadIdx.x
    ] = embd;
}

/**
 * @param input_ids [bs, seq_len]
 * @param tok_embd_weights [vocab_size, hidden_size]
 * @param pos_embd_weight_0 [pos_shape_0, 1, pos_embds_dim_0]
 * @param pos_embd_weight_1 [1, pos_shape_1, pos_embds_dim_1]
 * @param output [bs, seq_len, hidden_size]
 * @param pad_mask [bs, seq_len]
 *
 * gridDimX = batch_seq_len (dim x can be larger)
 * gridDimY = batch_size
 * blokcsize = hidden_size
 * 
 * only support 2 dim axial_pos_embds
 * only support hidden_size <= 1024
 */
template<typename T>
void encoder_embedding_launcher(
    const int *input_ids, const T *tok_embd_weights,
    const T *pos_embd_weight_0, const T *pos_embd_weight_1,
    int pos_embds_dim_0, int pos_embds_dim_1, int pos_shape_0, int pos_shape_1,
    int batch_size, int batch_seq_len, int hidden_size,
    int pad_id, int start_idx_pos_encodings,
    T *output, int *pad_mask) 
{
    assert(pos_embds_dim_0 % 32 == 0);
    assert(pos_embds_dim_1 % 32 == 0);
    assert(hidden_size <= 1024);
    encoder_embedding<<<dim3(batch_seq_len, batch_size), hidden_size>>>(
        input_ids, tok_embd_weights, pos_embd_weight_0, pos_embd_weight_1,
        pos_embds_dim_0, pos_embds_dim_1, pos_shape_0, pos_shape_1,
        hidden_size, pad_id, start_idx_pos_encodings,
        output, pad_mask);
}
template void encoder_embedding_launcher<float>(
    const int *input_ids, const float *tok_embd_weights,
    const float *pos_embd_weight_0, const float *pos_embd_weight_1,
    int pos_embds_dim_0, int pos_embds_dim_1, int pos_shape_0, int pos_shape_1,
    int batch_size, int batch_seq_len, int hidden_size,
    int pad_id, int start_idx_pos_encodings,
    float *output, int *pad_mask);


template<typename T>
__global__ void layer_norm(
    T *input, const T *weight, const T* bias,
    T eps)
{
    int norm_size = blockDim.x;
    T value = input[blockIdx.x * norm_size + threadIdx.x];
    T gamma = __ldg(&weight[threadIdx.x]);
    T beta = __ldg(&bias[threadIdx.x]);
    T mean = reduce_block_sum(value) / norm_size;
    T diff = value - mean;
    T var = diff * diff;
    var = reduce_block_sum(var) / norm_size;
    value = diff * rsqrtf(var + eps) * gamma + beta;
    input[blockIdx.x * norm_size + threadIdx.x] = value;
}
/**
 * @param input [size/norm_size, norm_size]
 * @param weight [norm_size]
 * @param bias [norm_size]
 */
// TODO only support norm_size <= 1024
template<typename T>
void layer_norm_launcher(
    T *input, const T *weight, const T *bias,
    T eps, int norm_size, int size)
{
    assert(norm_size <= 1024);
    layer_norm<<<size / norm_size, norm_size>>>(
        input, weight, bias, eps);
}
template void layer_norm_launcher<float>(
    float *input, const float *weight, const float *bias,
    float eps, int norm_size, int size);


template<typename T>
__global__ void bias_relu(T *input, const T *bias) {
    T value = input[blockIdx.x * blockDim.x + threadIdx.x] + __ldg(&bias[threadIdx.x]);
    value = max(value, static_cast<T>(0.0f));
    input[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

/**
 * @param input [size/hidden_size, hidden_size]
 * @param bias [hidden_size]
 * @param output [size/hidden_size, hidden_size]
 */
template<typename T>
void bias_relu_launcher(
    T *input, const T *bias, int hidden_size, int size)
{
    bias_relu<<<size / hidden_size, hidden_size>>>(
        input, bias);
}
template void bias_relu_launcher<float>(
    float *input, const float *bias,
    int hidden_size, int size);


template<typename T>
__global__ void add_bias(T *input, const T *bias) {
    input[blockIdx.x * blockDim.x + threadIdx.x] += __ldg(&bias[threadIdx.x]);
}

template<typename T>
void add_bias_launcher(
    T *input, const T *bias, int hidden_size, int size)
{
    add_bias<<<size / hidden_size, hidden_size>>>(
        input, bias);
}
template void add_bias_launcher<float>(
    float *input, const float *bias,
    int hidden_size, int size);



template<typename T>
__global__ void softmax(T *input) {
    T value = input[blockIdx.x * blockDim.x + threadIdx.x];
    value -= reduce_block_max(value);
    value = expf(value);
    value /= reduce_block_sum(value);
    input[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

template<typename T>
void softmax_launcher(
    T *input, int reduce_size, int size)
{
    softmax<<<size / reduce_size, reduce_size>>>(input);
}
template void softmax_launcher<float>(
    float *input, int reduce_size, int size);



template<typename T>
__global__ void atten_split_transpose(
    const T *input, int batch_size, int n_chunks,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    T value = input[threadIdx.x + blockIdx.x * blockDim.x];
    // 0, 1, 2, 3, 4 -> 0, 3, 1, 2, 4
    int idx_4 = threadIdx.x & (head_size - 1);
    int idx_3 = threadIdx.x / head_size;
    int idx_2 = blockIdx.x & (chunk_len - 1);
    int idx_1 = (blockIdx.x / chunk_len) & (n_chunks - 1);
    int idx_0 = blockIdx.x / chunk_len / n_chunks;
    output[
        idx_4 +
        head_size * (idx_2 +
        chunk_len * (idx_1 +
        n_chunks * (idx_3 +
        num_heads * idx_0)))
    ] = value;

}

// [bs, n_chunks, chunk_len, num_heads, head_size] ->
// [bs, num_heads, n_chunks, chunk_len, head_size]
template<typename T>
void atten_split_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    atten_split_transpose<<<batch_size * seq_len, num_heads * head_size>>>(
        input, batch_size, seq_len/chunk_len, chunk_len, num_heads, head_size, output);
}
template void atten_split_transpose_launcher<float>(
    const float *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    float *output);



template<typename T>
__global__ void atten_merge_transpose(
    const T *input, int batch_size, int n_chunks,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    T value = input[threadIdx.x + blockIdx.x * blockDim.x];
    // 0, 1, 2, 3, 4 -> 0, 2, 3, 1, 4
    int idx_4 = threadIdx.x;
    int idx_3 = blockIdx.x & (chunk_len - 1);
    int idx_2 = (blockIdx.x / chunk_len) & (n_chunks - 1);
    int idx_1 = (blockIdx.x / chunk_len / n_chunks) & (num_heads - 1);
    int idx_0 = blockIdx.x / chunk_len / n_chunks / num_heads;
    output[
        idx_4 +
        head_size * (idx_1 +
        num_heads * (idx_3 +
        chunk_len * (idx_2 +
        n_chunks * idx_0)))
    ] = value;
}
// [bs, num_heads, n_chunks, chunk_len, head_size] ->
// [bs, n_chunks, chunk_len, num_heads, head_size]
template<typename T>
void atten_merge_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    atten_merge_transpose<<<batch_size * num_heads * seq_len, head_size>>>(
        input, batch_size, seq_len/chunk_len, chunk_len, num_heads, head_size, output);
}
template void atten_merge_transpose_launcher<float>(
    const float *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    float *output);



template<typename T>
__global__ void look_adjacent(
    const T *input, int last_dim_size,
    int before, int after, int N,
    T *output)
{
    for (int k = 0; k < last_dim_size; k += blockDim.x) {
        T value = input[
            threadIdx.x + k +
            (blockIdx.x + blockIdx.y * gridDim.x) * last_dim_size
        ];
        // i on the N dim
        for (int i = 0; i < N; i ++) {
            int tgt_block_x = blockIdx.x - i + before;
            tgt_block_x = tgt_block_x < 0 ? tgt_block_x + gridDim.x : tgt_block_x;
            tgt_block_x = tgt_block_x >= gridDim.x ? tgt_block_x - gridDim.x : tgt_block_x;
            output[
                threadIdx.x + k +
                (i + (tgt_block_x + blockIdx.y * gridDim.x) * N) * last_dim_size
            ] = value;
        }
    }
}
// [bs * num_heads, n_chunks, chunk_len * head_size] ->
// [bs * num_heads, n_chunks, (Num_bef + Num_aft + 1), chunk_len * head_size]
// chunk_len * head_size = K * block_size
// gridDim.x = n_chunks
// gridDim.y = bs * num_heads
template<typename T>
void look_adjacent_launcher(
    const T *input, int batch_size, int num_heads, int n_chunks,
    int chunk_len, int head_size, int before, int after,
    T *output)
{
    // head_size can be 1, so should not use head_size as block_size
    int last_dim_size = chunk_len * head_size;
    int block_size = min(1024, last_dim_size);
    look_adjacent<<<dim3(n_chunks, batch_size * num_heads), block_size>>>(
        input, chunk_len * head_size, before, after, before + after + 1, output);
}
template void look_adjacent_launcher<int>(
    const int *input, int batch_size, int num_heads, int n_chunks,
    int chunk_len, int head_size, int before, int after,
    int *output);
template void look_adjacent_launcher<float>(
    const float *input, int batch_size, int num_heads, int n_chunks,
    int chunk_len, int head_size, int before, int after,
    float *output);



template<typename T>
__global__ void local_atten_enc_mask(
    T *qk_dots, const int *mask, T mask_value, int num_heads,
    int n_chunks, int chunk_len)
{
    int m = __ldg(&mask[
        threadIdx.x +
        (blockIdx.x / chunk_len + (blockIdx.y / num_heads) * n_chunks) * blockDim.x
    ]);

    int index = 
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
    qk_dots[index] = m ? qk_dots[index] : mask_value;

}

// [bs, num_heads, n_chunks, chunk_len,  N * chunk_len] * [bs, n_chunks, N * chunk_len]
// gridDim.x = n_chunks * chunk_len
// gridDim.y = batch_size * num_heads
// blockDim.x = N * chunk_len
template<typename T>
void local_atten_enc_mask_launcher(
    T *qk_dots, const int *mask, T mask_value, int batch_size, int num_heads,
    int n_chunks, int chunk_len, int N)
{
    local_atten_enc_mask<<<dim3(n_chunks * chunk_len, batch_size * num_heads), N * chunk_len>>>(
        qk_dots, mask, mask_value, num_heads, n_chunks, chunk_len);
}
template void local_atten_enc_mask_launcher<float>(
    float *qk_dots, const int *mask, float mask_value, int batch_size, int num_heads,
    int n_chunks, int chunk_len, int N);


template<typename T>
__global__ void repeat(const T *in, T *out) {
    out[
        threadIdx.x +
        (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x) * blockDim.x
    ] =
    __ldg(&in[
        threadIdx.x +
        (blockIdx.x + blockIdx.z * gridDim.x) * blockDim.x
    ]);
}

/**
 * @param in [dim0, dim1]
 * @param out [dim0, repeat_num, dim1]
 */
template<typename T>
void repeat_launcher(
    const T *in, int dim0, int dim1, int repeat_num,
    T *out)
{
    // TODO if dim1 very small
    // [dim0, dim1/blocksize, blocksize] ->
    // [dim0, repeat_num, dim1/blocksize, blocksize]
    int blocksize = min(1024, dim1);
    dim3 grid(dim1/blocksize, repeat_num, dim0);
    repeat<<<grid, blocksize>>>(in, out);
}
template void repeat_launcher<float>(
    const float *in, int dim0, int dim1, int repeat_num,
    float *out);



/**
 * @param in [bs * num_heads * num_hashes * seq_len, num_bucket/2]
 * @param out [bs * num_heads * num_hashes * seq_len]
 */
template<typename T>
__global__ void lsh_bucket_argmax(
    const T *in, int *out)
{
    // argmax
    auto p = thrust::make_pair(
        in[threadIdx.x + blockIdx.x * blockDim.x],
        threadIdx.x
    );
    p = p.first > 0 ? p : thrust::make_pair(-p.first, p.second + blockDim.x);
    p = reduce_block_argmax(p);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = p.second;
    }
}

/**
 * @param in [bs, num_heads, num_hashes, seq_len]
 * @param atten_mask [bs, seq_len]
 */
__global__ void lsh_bucket_mask_offset(
    int *in, const int *atten_mask, int num_bucket,
    int num_heads, int num_hashes)
{
    int value = in[threadIdx.x + blockIdx.x * blockDim.x];
    // mask
    int mask = __ldg(&atten_mask[threadIdx.x + (blockIdx.x / num_heads / num_hashes) * blockDim.x]);
    value = mask ? value : num_bucket - 1;
    //offset
    int offset = (blockIdx.x % num_hashes) * num_bucket;
    value += offset;
    in[threadIdx.x + blockIdx.x * blockDim.x] = value;
}


template<typename T>
void lsh_bucket_argmax_mask_offset_launcher(
    const T *in, const int *atten_mask,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int num_bucket,
    int *out)
{
    lsh_bucket_argmax<<<batch_size * num_heads * num_hashes * seq_len, num_bucket/2>>>(
        in, out);
    lsh_bucket_mask_offset<<<batch_size * num_heads * num_hashes, seq_len>>>(
        out, atten_mask, num_bucket + 1, num_heads, num_hashes);
}
template void lsh_bucket_argmax_mask_offset_launcher<float>(
    const float *in, const int *atten_mask,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int num_bucket,
    int *out);


// __global__ void arrange_last(int *out) {
//     out[
//         threadIdx.x +
//         (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
//     ] = threadIdx.x + blockIdx.x * blockDim.x;
// }

// /**
//  * @param out [size/last_size, last_size]
//  */
// void arrange_last_launcher(
//     int *out, int last_size, int size)
// {
//     // [size/last_size, last_size/blocksize, blocksize]
//     int blocksize = min(1024, last_size);
//     dim3 grid(last_size/blocksize, size/last_size);
//     arrange_last<<<grid, blocksize>>>(out);
// }



template<typename T, int ITEMS_PER_THREAD,
         int BLOCK_SIZE=128, int BEGIN_BIT=0, int END_BIT=sizeof(T)*8, int RADIX_BITS=4>
__global__ void block_unsigned_radix_sort(T *in, int *idx) {
    // linear load, line->ITEM_PER_THREAD, row->BLOCK_SIZE
    // 0, 1, ..., items_per_thread-1,
    // items_per_thread, ..., 2*items_per_thread-1,
    // ...
    T keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];
    __shared__ T smem_keys[ITEMS_PER_THREAD * BLOCK_SIZE];
    __shared__ int smem_values[ITEMS_PER_THREAD * BLOCK_SIZE];

    // load
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        keys[i] = in[threadIdx.x * ITEMS_PER_THREAD + i + blockIdx.x * ITEMS_PER_THREAD * BLOCK_SIZE];
        values[i] = threadIdx.x * ITEMS_PER_THREAD + i;
    }

    #pragma unroll
    for (int start = BEGIN_BIT; start < END_BIT; start += RADIX_BITS) {
        int pass_bits = RADIX_BITS < END_BIT - start ? RADIX_BITS : END_BIT - start;
        // shift extractor
        auto digit_extractor = [start, pass_bits](T k) -> uint32_t {
            return uint32_t(k >> T(start)) & uint32_t((1 << pass_bits) - 1);
        };
        
        // rank keys
        // ranks = prev_bucket_ranks (ranks of prev buckets) +
        //         prev_lines_ranks (ranks of this bucket and prev lines) + 
        //         line_ranks (ranks of this bucket and this line)
        int bucket_counter[1<<RADIX_BITS] {};
        int ranks[ITEMS_PER_THREAD] {};
        uint32_t digits[ITEMS_PER_THREAD] {};

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
            digits[ITEM] = digit_extractor(keys[ITEM]);
            // add line_ranks,
            ranks[ITEM] = bucket_counter[digits[ITEM]];
            bucket_counter[digits[ITEM]] += 1;
        }

        int bucket_prefix_sum_accum = 0;
        #pragma unroll
        for (int BIT = 0; BIT < (1<<RADIX_BITS); BIT++) {
            int prev_lines_rank = reduce_block_prefix_sum(bucket_counter[BIT]);
            int prev_bucket_rank = bucket_prefix_sum_accum;
            bucket_prefix_sum_accum += reduce_block_sum(bucket_counter[BIT]);
            // prev_lines_rank and prev_bucket_rank
            bucket_counter[BIT] = prev_lines_rank + prev_bucket_rank;
        }

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
            ranks[ITEM] += bucket_counter[digits[ITEM]];
        }

        // scatter to smem
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
            smem_keys[ranks[ITEM]] = keys[ITEM];
            smem_values[ranks[ITEM]] = values[ITEM];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            keys[i] = smem_keys[threadIdx.x * ITEMS_PER_THREAD + i];
            values[i] = smem_values[threadIdx.x * ITEMS_PER_THREAD + i];
        }
    }

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        in[threadIdx.x * ITEMS_PER_THREAD + i + blockIdx.x * ITEMS_PER_THREAD * BLOCK_SIZE] = keys[i];
        idx[threadIdx.x * ITEMS_PER_THREAD + i + blockIdx.x * ITEMS_PER_THREAD * BLOCK_SIZE] = values[i];
    }
}

template<typename T>
void block_unsigned_radix_sort_launcher(T *in, int *idx, int grid, int N) {
    constexpr int BLOCK_SIZE = 128;
    switch(N) {
        case 128:
            block_unsigned_radix_sort<T, 128/BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(in, idx);
            break;
        case 256:
            block_unsigned_radix_sort<T, 256/BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(in, idx);
            break;
        case 512:
            block_unsigned_radix_sort<T, 512/BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(in, idx);
            break;
        case 1024:
            block_unsigned_radix_sort<T, 1024/BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(in, idx);
            break;
        case 2048:
            block_unsigned_radix_sort<T, 2048/BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(in, idx);
            break;
        default:
            throw "num_hashes * seq_len must be 128, 256, 512, 1024 or 2048";
    }
}
template void block_unsigned_radix_sort_launcher<int>(int *in, int *idx, int grid, int N);



__global__ void lsh_scatter_undo_idx(
    int *sorted_idx, int *undo_sorted_idx, int seq_len)
{
    int idx = sorted_idx[
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
    ];
    // arange num_hashes*seq_len
    int value = threadIdx.x + blockIdx.x * blockDim.x;

    undo_sorted_idx[
        idx +
        blockIdx.y * gridDim.x * blockDim.x] = value;

    sorted_idx[
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
    ] = idx % seq_len;
}

/**
 * produce undo_sorted_idx, and also make sorted_idx %= seq_len
 * @param sorted_idx [bs*num_heads, num_hashes*seq_len]
 * @param undo_sorted_idx [bs*num_heads, num_hashes*seq_len]
 */
void lsh_scatter_undo_idx_launcher(
    int *sorted_idx, int *undo_sorted_idx,
    int batch_size, int num_heads, int num_hashes, int seq_len)
{
    // [bs*num_heads, num_hashes*seq_len/blocksize, blocksize]
    int blocksize = min(1024, num_hashes * seq_len);
    dim3 grid(num_hashes * seq_len / blocksize, batch_size * num_heads);
    lsh_scatter_undo_idx<<<grid, blocksize>>>(
        sorted_idx, undo_sorted_idx, seq_len);
}


template<typename T>
__global__ void lsh_gather_by_expansion(
    const T *in, const int *idx, int seq_len, T *out)
{
    int gather_idx = idx[blockIdx.x + blockIdx.y * gridDim.x];
    out[
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
    ] = 
    in[
        threadIdx.x +
        (gather_idx % seq_len + blockIdx.y * seq_len) * blockDim.x
    ];
}
/**
 * gather the num_hashes*seq_len dim
 * @param in [bs, num_heads, seq_len, head_size] (expand to [bs, num_heads, num_hashes*seq_len, head_size])
 * @param idx [bs, num_heads, num_hashes*seq_len]
 * @param out [bs, num_heads, num_hashes*seq_len, head_size]
 */
template<typename T>
void lsh_gather_by_expansion_launcher(
    const T *in, const int *idx, int batch_size,
    int num_heads, int num_hashes, int seq_len, int head_size,
    T *out)
{
    dim3 grid(num_hashes * seq_len, batch_size * num_heads);
    lsh_gather_by_expansion<<<grid, head_size>>>(
        in, idx, seq_len, out);
}
template void lsh_gather_by_expansion_launcher<float>(
    const float *in, const int *idx, int batch_size,
    int num_heads, int num_hashes, int seq_len, int head_size,
    float *out);


template<typename T>
__global__ void lsh_len_norm(
    const T *in, T norm_scalar, T *out)
{
    T value = in[threadIdx.x + blockIdx.x * blockDim.x];
    T rstd = rsqrtf(reduce_block_sum(value * value) / blockDim.x + static_cast<T>(1e-6f));
    out[threadIdx.x + blockIdx.x * blockDim.x] = value * rstd * norm_scalar;
}

template<typename T>
void lsh_len_norm_launcher(
    const T *in, int norm_size, int size, T norm_scalar, T *out)
{
    lsh_len_norm<<<size/norm_size, norm_size>>>(in, norm_scalar, out);
}
template void lsh_len_norm_launcher<float>(
    const float *in, int norm_size, int size, float norm_scalar, float *out);


/**
 * atten mask and self mask
 * gridDim.x = bs * num_heads * num_hashes*seq_len/chunk_len * chunk_len
 * blockDim.x = N * chunk_len
 * 
 * @param qk_dots [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len,  N * chunk_len]
 * @param q_idx [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len]
 * @param k_idx [bs, num_heads, num_hashes*seq_len/chunk_len, N * chunk_len]
 * @param atten_mask [bs, seq_len]
 */
template<typename T>
__global__ void lsh_enc_mask(
    T *qk_dots, const int *q_idx, const int *k_idx, const int *atten_mask,
    T mask_value, T self_mask_value, int num_heads, int num_hashes,
    int seq_len, int chunk_len)
{
    T value = qk_dots[threadIdx.x + blockIdx.x * blockDim.x];
    // mask
    int k_gather_idx = __ldg(&k_idx[
        threadIdx.x +
        (blockIdx.x / chunk_len) * blockDim.x
    ]);
    int mask = __ldg(&atten_mask[
        k_gather_idx +
        (blockIdx.x / (num_heads * num_hashes * seq_len)) * seq_len
    ]);
    // self mask
    int q = __ldg(&q_idx[
        blockIdx.x
    ]);
    int k = __ldg(&k_idx[
        threadIdx.x +
        (blockIdx.x / chunk_len) * blockDim.x
    ]);
    bool self_mask = q != k;

    value = mask ? value : mask_value;
    value = self_mask ? value : self_mask_value;
    qk_dots[threadIdx.x + blockIdx.x * blockDim.x] = value;
}

template<typename T>
void lsh_enc_mask_launcher(
    T *qk_dots, const int *q_idx, const int *k_idx, const int *atten_mask,
    T mask_value, T self_mask_value, int batch_size, int num_heads, int num_hashes,
    int seq_len, int chunk_len, int N)
{
    dim3 grid(batch_size * num_heads * num_hashes * seq_len);
    lsh_enc_mask<<<grid, N * chunk_len>>>(
        qk_dots, q_idx, k_idx, atten_mask, mask_value, self_mask_value,
        num_heads, num_hashes, seq_len, chunk_len);
}

template void lsh_enc_mask_launcher<float>(
    float *qk_dots, const int *q_idx, const int *k_idx, const int *atten_mask,
    float mask_value, float self_mask_value, int batch_size, int num_heads, int num_hashes,
    int seq_len, int chunk_len, int N);


/**
 * softmax version that also return logits
 */
template<typename T>
__global__ void softmax_with_logits(T *input, T *logits) {
    T value = input[blockIdx.x * blockDim.x + threadIdx.x];
    T max_value = reduce_block_max(value);
    value -= max_value;
    value = expf(value);
    T sum_value = reduce_block_sum(value);
    input[blockIdx.x * blockDim.x + threadIdx.x] = value / sum_value;
    if (threadIdx.x == 0) {
        logits[blockIdx.x] = max_value + logf(sum_value);
    }
}

template<typename T>
void softmax_with_logits_launcher(
    T *input, T *logits, int reduce_size, int size)
{
    softmax_with_logits<<<size / reduce_size, reduce_size>>>(input, logits);
}
template void softmax_with_logits_launcher<float>(
    float *input, float *logits, int reduce_size, int size);


template<typename T>
__global__ void lsh_undo_sort(
    const int *undo_sort_idx, const T *vec, const T *logits,
    T *rev_vec, T *rev_logits)
{
    int idx = undo_sort_idx[blockIdx.x + blockIdx.y * gridDim.x];
    rev_vec[
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
    ] =
    vec[
        threadIdx.x +
        (idx + blockIdx.y * gridDim.x) * blockDim.x
    ];

    if (threadIdx.x == 0) {
        rev_logits[blockIdx.x + blockIdx.y * gridDim.x] = logits[idx + blockIdx.y * gridDim.x];
    }
}

/**
 * gather
 * @param undo_sort_idx [bs*num_heads, num_hashes*seq_len]
 * @param vec [bs*num_heads, num_hashes*seq_len, head_size]
 * @param logits [bs*num_heads, num_hashes*seq_len]
 * @param rev_vec [bs*num_heads, num_hashes*seq_len, head_size]
 * @param rev_logits [bs*num_heads, num_hashes*seq_len]
 */
template<typename T>
void lsh_undo_sort_launcher(
    const int *undo_sort_idx, const T *vec, const T *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    T *rev_vec, T *rev_logits)
{
    dim3 grid(num_hashes * seq_len, batch_size * num_heads);
    lsh_undo_sort<<<grid, head_size>>>(
        undo_sort_idx, vec, logits, rev_vec, rev_logits);
}
template void lsh_undo_sort_launcher<float>(
    const int *undo_sort_idx, const float *vec, const float *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    float *rev_vec, float *rev_logits);


/**
 * gridDim.x = seq_len
 * gridDim.y = bs * num_heads
 * blockDim.x = head_size
 * 
 * @param in [bs, num_heads, num_hashes, seq_len, head_size]
 * @param logits [bs, num_heads, num_hashes, seq_len]
 * @param out [bs, num_heads, seq_len, head_size]
 */
template<typename T, int num_hashes>
__global__ void sum_up_hashes(
    const T *in, const T *logits, T *out)
{
    T vs[num_hashes];
    T ls[num_hashes];
    # pragma unroll
    for (int i = 0; i < num_hashes; i ++) {
        int logits_idx = 
            blockIdx.x +
            (i + blockIdx.y * num_hashes) * gridDim.x;
        vs[i] = in[threadIdx.x + logits_idx * blockDim.x];
        ls[i] = __ldg(&logits[logits_idx]);
    }
    T logsumexp = static_cast<T>(0.0f);
    T *max_l = thrust::max_element(thrust::device, ls, ls + num_hashes);
    # pragma unroll
    for (int i = 0; i < num_hashes; i ++) {
        logsumexp += expf(ls[i] - *max_l);
    }
    logsumexp = logf(logsumexp) + *max_l;
    T res = static_cast<T>(0.0f);
    # pragma unroll
    for (int i = 0; i < num_hashes; i ++) {
        res += vs[i] * expf(ls[i] - logsumexp);
    }
    out[
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x
    ] = res;
}

template<typename T>
void sum_up_hashes_launcher(
    const T *in, const T *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    T *out)
{
    dim3 grid(seq_len, batch_size * num_heads);
    dim3 block(head_size);
    switch(num_hashes) {
        case 1:
            thrust::copy(
                thrust::device,
                in, in + batch_size * num_heads * seq_len * head_size,
                out);
            break;
        case 2:
            sum_up_hashes<T, 2><<<grid, block>>>(in, logits, out);
            break;
        case 4:
            sum_up_hashes<T, 4><<<grid, block>>>(in, logits, out);
            break;
        case 8:
            sum_up_hashes<T, 8><<<grid, block>>>(in, logits, out);
            break;
        default:
            throw "num_hashes must be 1, 2, 4 or 8";
    }
}
template void sum_up_hashes_launcher<float>(
    const float *in, const float *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    float *out);


template<typename T>
__global__ void add(T *first, T *second) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = first[idx] + second[idx];
    first[idx] = sum;
    second[idx] = sum;
}
/**
 * first and second both store the result
 */
template<typename T>
void add_launcher(T *first, T *second, int size) {
    int blocksize = min(1024, size);
    add<<<size/blocksize, blocksize>>>(first, second);
}
template void add_launcher<float>(float *first, float *second, int size);


} // namespace FastReformer