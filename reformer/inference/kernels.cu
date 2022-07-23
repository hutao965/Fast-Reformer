#include "kernels.cuh"

namespace FastReformer {

// TODO add stream
// TODO fp16
// TODO template for common hidden size (1024 768 and so on)


// TODO half2 version
template<typename T>
__forceinline__ __device__ T reduce_warp_sum(T value) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, mask, 32);
    return value;
}

template<typename T>
__forceinline__ __device__ T reduce_block_sum(T value) {
    value = reduce_warp_sum<T>(value);
    __shared__ T s[32];
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    if (laneId == 0) s[warpId] = value;
    __syncthreads();
    value = (laneId < ((blockDim.x + 31) >> 5)) ? s[laneId] : static_cast<T>(0.0f);
    value = reduce_warp_sum<T>(value);
    return value;
}


// TODO float4, l2 cache
template<typename T>
__global__ void encoder_embedding(
    const int *input_ids, const T *tok_embd_weights,
    const T *pos_embd_weight_0, const T *pos_embd_weight_1,
    int pos_embds_dim_0, int pos_embds_dim_1, int pos_shape_0, int pos_shape_1,
    int hidden_size, int pad_id, int start_idx_pos_encodings,
    T *output, T *pad_mask) 
{
    int batch_seq_len = gridDim.x;
    int batch_seq_id = blockIdx.x;
    // int batch_size = gridDim.y;
    int batch_id = blockIdx.y;
    int pos_id = batch_seq_id + start_idx_pos_encodings;

    int input_id = input_ids[batch_id * batch_seq_len + batch_seq_id];
    pad_mask[batch_id * batch_seq_len + batch_seq_id] = static_cast<T>(input_id != pad_id);
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
    T *output, T *pad_mask) 
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
    float *output, float *pad_mask);



// TODO float4, l2cache
template<typename T>
__global__ void layer_norm(
    const T *input, const T *weight, const T* bias,
    T eps, T *output)
{
    int norm_size = blockDim.x;
    T value = input[blockIdx.x * norm_size + threadIdx.x];
    T gamma = weight[threadIdx.x];
    T beta = bias[threadIdx.x];
    T mean = reduce_block_sum<T>(value) / norm_size;
    T diff = value - mean;
    T var = diff * diff;
    var = reduce_block_sum<T>(var) / norm_size;
    value = diff * rsqrtf(var + eps) * gamma + beta;
    output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}
/**
 * @param input [size/norm_size, norm_size]
 * @param weight [norm_size]
 * @param bias [norm_size]
 * @param output [size/norm_size, norm_size]
 */
// TODO only support norm_size <= 1024
template<typename T>
void layer_norm_launcher(
    const T *input, const T *weight, const T *bias,
    T eps, int norm_size, int size, T *output)
{
    assert(norm_size <= 1024);
    layer_norm<T><<<size / norm_size, norm_size>>>(
        input, weight, bias, eps, output);
}
template void layer_norm_launcher<float>(
    const float *input, const float *weight, const float *bias,
    float eps, int norm_size, int size, float *output);




// TODO hidden size > 1024
// TODO float4
// TODO ldg
template<typename T>
__global__ void bias_relu(
    const T *input, const T *bias, T *output)
{
    T value = input[blockIdx.x * blockDim.x + threadIdx.x] + bias[threadIdx.x];
    value = max(value, static_cast<T>(0.0f));
    output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

/**
 * @param input [size/hidden_size, hidden_size]
 * @param bias [hidden_size]
 * @param output [size/hidden_size, hidden_size]
 */
template<typename T>
void bias_relu_launcher(
    const T *input, const T *bias,
    int hidden_size, int size, T *output)
{
    bias_relu<T><<<size / hidden_size, hidden_size>>>(
        input, bias, output);
}
template void bias_relu_launcher<float>(
    const float *input, const float *bias,
    int hidden_size, int size, float *output);




template<typename T>
__global__ void softmax(const T *input, T *output) {
    T value = input[blockIdx.x * blockDim.x + threadIdx.x];
    value -= log(reduce_block_sum<T>(exp(value)));
    value = exp(value);
    output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}


template<typename T>
void softmax_launcher(
    const T *input, int reduce_size, int size,
    T *output)
{
    softmax<T><<<size / reduce_size, reduce_size>>>(
        input, output);
}
template void softmax_launcher<float>(
    const float *input, int reduce_size, int size,
    float *output);



// lsh atten

// 1. random b/2 vecs -> mul -> sort -> b buckets (num_hashes round)
// 




// [bs, seq_len, hiddens] -> [bs, num_heads, seq_len, head_size]
template<typename T>
__global__ void split_hiddens() {

}

template<typename T>
void split_hiddens_launcher() {
    std::cout << "split_launcher" << std::endl;
}


// [bs, num_heads, seq_len, head_size] -> [bs, seq_len, hiddens]
template<typename T>
__global__ void merge_hiddens() {

}

template<typename T>
void merge_hiddens_launcher() {
    std::cout << "merge_launcher" << std::endl;
}


//[B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size]
template<typename T>
__global__ void split_chunks() {

}

template<typename T>
void split_chunks_launcher() {
    std::cout << "split_launcher" << std::endl;
}

// merge chunks

// look adjacent kernel

// mask for local


// softmax


// compute mask for lsh



// sort 和 reverse sort
// num hashes 多轮



} // namespace FastReformer