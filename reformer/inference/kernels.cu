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
__forceinline__ __device__ thrust::pair<T, int> reduce_warp<thrust::pair<T, int>, F>(thrust::pair<T, int> value, F reduction) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value = reduction(value, mask);
    return value;
}

template<typename T, typename F>
__forceinline__ __device__ T reduce_block(T value, F reduction, T init_value) {
    value = reduce_warp(value, reduction);
    __shared__ T s[32];
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    if (laneId == 0) s[warpId] = value;
    __syncthreads();
    value = (laneId < ((blockDim.x + 31) >> 5)) ? s[laneId] : init_value;
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

template<typename T>
__forceinline__ __device__ T reduce_block_max(T value) {
    return reduce_block(
        value,
        [](T x, T y){ return max(x, y); },
        static_cast<T>(-FLT_MAX)
    );
}



template<typename T>
__forceinline__ __device__ thrust::pair<T, int> reduce_block_argmax(thrust::pair<T, int> value) {
    return reduce_block(
        value,
        [](thrust::pair<T, int> x, int mask){
            T other_first = __shfl_xor_sync(0xffffffff, x.first, mask, 32);
            int other_second = __shfl_xor_sync(0xffffffff, x.second, mask, 32);
            return x.first >= other_first ?
                   x : thrust::make_pair(other_first, other_second);
        },
        thrust::make_pair(static_cast<T>(-FLT_MAX), 0)
    );
}



// TODO float4
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
    assert(pos_embds_dims.size() == 2);
    assert(pos_embds_dims[0] % 32 == 0);
    assert(pos_embds_dims[1] % 32 == 0);
    assert(hidden_size <= 1024);
    encoder_embedding<T><<<dim3(batch_seq_len, batch_size), hidden_size>>>(
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



// TODO float4
template<typename T>
__global__ void layer_norm(
    T *input, const T *weight, const T* bias,
    T eps)
{
    int norm_size = blockDim.x;
    T value = input[blockIdx.x * norm_size + threadIdx.x];
    T gamma = __ldg(&weight[threadIdx.x]);
    T beta = __ldg(&bias[threadIdx.x]);
    T mean = reduce_block_sum<T>(value) / norm_size;
    T diff = value - mean;
    T var = diff * diff;
    var = reduce_block_sum<T>(var) / norm_size;
    value = diff * rsqrtf(var + eps) * gamma + beta;
    input[blockIdx.x * blockDim.x + threadIdx.x] = value;
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
    layer_norm<T><<<size / norm_size, norm_size>>>(
        input, weight, bias, eps);
}
template void layer_norm_launcher<float>(
    float *input, const float *weight, const float *bias,
    float eps, int norm_size, int size);




// TODO hidden size > 1024
// TODO float4
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
    bias_relu<T><<<size / hidden_size, hidden_size>>>(
        input, bias);
}
template void bias_relu_launcher<float>(
    float *input, const float *bias,
    int hidden_size, int size);


template<typename T>
__global__ void add_bias(T *input, const T *bias) {
    input[blockIdx.x * blockDim.x + threadIdx.x] = 
        input[blockIdx.x * blockDim.x + threadIdx.x] + __ldg(&bias[threadIdx.x]);
}

template<typename T>
void add_bias_launcher(
    T *input, const T *bias, int hidden_size, int size)
{
    add_bias<T><<<size / hidden_size, hidden_size>>>(
        input, bias);
}
template void add_bias_launcher<float>(
    float *input, const float *bias,
    int hidden_size, int size);



template<typename T>
__global__ void softmax(T *input) {
    T value = input[blockIdx.x * blockDim.x + threadIdx.x];
    value -= reduce_block_max<T>(value);
    value = expf(value);
    value /= reduce_block_sum<T>(value);
    input[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

template<typename T>
void softmax_launcher(
    T *input, int reduce_size, int size)
{
    softmax<T><<<size / reduce_size, reduce_size>>>(input);
}
template void softmax_launcher<float>(
    float *input, int reduce_size, int size);



// [bs, n_chunks, chunk_len, num_heads, head_size] ->
// [bs, num_heads, n_chunks, chunk_len, head_size]
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

template<typename T>
void atten_split_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    atten_split_transpose<T><<<batch_size * seq_len, num_heads * head_size>>>(
        input, batch_size, seq_len/chunk_len, chunk_len, num_heads, head_size, output);
}
template void atten_split_transpose_launcher<float>(
    const float *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    float *output);


// [bs, num_heads, n_chunks, chunk_len, head_size] ->
// [bs, n_chunks, chunk_len, num_heads, head_size]
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

template<typename T>
void atten_merge_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output)
{
    atten_merge_transpose<T><<<batch_size * num_heads * seq_len, head_size>>>(
        input, batch_size, seq_len/chunk_len, chunk_len, num_heads, head_size, output);
}
template void atten_merge_transpose_launcher<float>(
    const float *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    float *output);


// [bs * num_heads, n_chunks, chunk_len * head_size] ->
// [bs * num_heads, n_chunks, (Num_bef + Num_aft + 1), chunk_len * head_size]
// chunk_len * head_size = K * block_size
// gridDim.x = n_chunks
// gridDim.y = bs * num_heads
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

template<typename T>
void look_adjacent_launcher(
    const T *input, int batch_size, int num_heads, int n_chunks,
    int chunk_len, int head_size, int before, int after,
    T *output)
{
    // head_size can be 1, so should not use head_size as block_size
    int last_dim_size = chunk_len * head_size;
    int block_size = min(1024, last_dim_size);
    look_adjacent<T><<<dim3(n_chunks, batch_size * num_heads), block_size>>>(
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


// [bs, num_heads, n_chunks, chunk_len,  N * chunk_len] * [bs, n_chunks, N * chunk_len]
// gridDim.x = n_chunks * chunk_len
// gridDim.y = batch_size * num_heads
// blockDim.x = N * chunk_len
template<typename T>
__global__ void local_atten_enc_mask(
    T *qk_dots, const int *mask, int num_heads,
    int n_chunks, int chunk_len)
{
    int mask_value = __ldg(&mask[
        threadIdx.x +
        (blockIdx.x / chunk_len + (blockIdx.y / num_heads) * n_chunks) * blockDim.x
    ]);

    int index = 
        threadIdx.x +
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
    qk_dots[index] = mask_value ? qk_dots[index] : static_cast<T>(-1e9f);

}


template<typename T>
void local_atten_enc_mask_launcher(
    T *qk_dots, const int *mask, int batch_size, int num_heads,
    int n_chunks, int chunk_len, int N)
{
    local_atten_enc_mask<T><<<dim3(n_chunks * chunk_len, batch_size * num_heads), N * chunk_len>>>(
        qk_dots, mask, num_heads, n_chunks, chunk_len);
}
template void local_atten_enc_mask_launcher<float>(
    float *qk_dots, const int *mask, int batch_size, int num_heads,
    int n_chunks, int chunk_len, int N);



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
    p = reduce_block_argmax<T>(p);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = p.second;
    }
}

/**
 * @param in [bs, num_heads, num_hashes, seq_len]
 * @param atten_mask [bs, seq_len]
 */
template<typename T>
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
    lsh_bucket_argmax<T><<<batch_size * num_heads * num_hashes * seq_len, num_bucket/2>>>(
        in, out);
    lsh_bucket_mask_offset<T><<<batch_size * num_heads * num_hashes, seq_len>>>(
        out, atten_mask, num_bucket + 1, num_heads, num_hashes);
}
template void lsh_bucket_argmax_mask_offset_launcher<float>(
    const float *in, const int *atten_mask,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int num_bucket,
    int *out);


/**
 * gather the num_hashes*seq_len dim
 * @param in [bs, num_heads, seq_len, head_size] (expand to [bs, num_heads, num_hashes*seq_len, head_size])
 * @param idx [bs, num_heads, num_hashes*seq_len]
 * @param out [bs, num_heads, num_hashes*seq_len, head_size]
 */
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

template<typename T>
void lsh_gather_by_expansion_launcher(
    const T *in, const int *idx, int batch_size,
    int num_heads, int num_hashes, int seq_len, int head_size,
    T *out)
{
    dim3 grid(num_hashes * seq_len, batch_size * num_heads);
    lsh_gather_by_expansion<T><<<grid, head_size>>>(
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
    lsh_len_norm<T><<<size/norm_size, norm_size>>>(in, norm_scalar, out);
}
template void lsh_len_norm_launcher<float>(
    const float *in, int norm_size, int size, float norm_scalar, float *out);

} // namespace FastReformer