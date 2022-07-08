#include "kernels.cuh"

namespace FastReformer {

// embedding
// template<typename T>
// __forceinline__ __device__ T tok_embedding()
// {
//     // TODO inline or other methods
// }

template<typename T>
__global__ void encoder_embedding_with_aixal_pos(
    const int *input_ids, const T *tok_emb_weights, const T *pos_emb_weights,
    int pad_id, int batch_size, int batch_seq_len, int hidden_size,
    T *output, int *pad_mask)
{
    // TODO axial params
    //produce position ids

    // produce pad mask

    // x += word_embedding();
}

template<typename T>
__global__ void encoder_embedding(
    const int *input_ids, const T *tok_emb_weights, const T *pos_emb_weights,
    int pad_id, int batch_size, int batch_seq_len, int hidden_size,
    int start_pos_idx, T *output, int *pad_mask) {
    // x += word_embedding();
}

// set gridDim
template<typename T>
void encoder_embedding_launcher() {
    std::cout << "encoder_embedding_launcher" << std::endl;
}
template void encoder_embedding_launcher<float>();


// encoder out 2 hiddens concat

// encoder ln has 2 * hiddensize for revnet
template<typename T>
__global__ void layer_norm() {

}

template<typename T>
void layer_norm_launcher() {
    std::cout << "layer_norm_launcher" << std::endl;
}


// local atten
// ln + revnet
template<typename T>
__global__ void rev_layer_norm() {

}

template<typename T>
void rev_layer_norm_launcher() {
    std::cout << "rev_layer_norm_launcher" << std::endl;
}

// project
// reshape qkv
template<typename T>
__global__ void split() {

}

template<typename T>
void split_launcher() {
    std::cout << "split_launcher" << std::endl;
}
// q*k + softmax
// TODO q * k 因为multi head 使用stridedbatchedEx
template<typename T>
__global__ void softmax() {

}

template<typename T>
void softmax_launcher() {
    std::cout << "softmax_launcher" << std::endl;
}
// soft * v
// reshape back + revnet
template<typename T>
__global__ void merge() {

}

template<typename T>
void merge_launcher() {
    std::cout << "merge_launcher" << std::endl;
}


// lsh atten

// 1. random b/2 vecs -> mul -> sort -> b buckets (num_hashes round)
// 





// ffn
template<typename T>
__global__ void bias_relu() {

}

template<typename T>
void bias_relu_launcher() {
    std::cout << "bias_relu_launcher" << std::endl;
}

//cublasGemmEx TODO chunk ffn dense



} // namespace FastReformer