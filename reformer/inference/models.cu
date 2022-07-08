#include "models.cuh"

namespace FastReformer {

// template<FloatType fp>
// ReformerLayer<fp>::ReformerLayer(
//     py::dict &weights, py::dict &config, cudaStream_t &stream, cublasHandle_t &cublasHandle)
// {
//     std::cout << "ReformerLayer construct" << std::endl;
// }
// template ReformerLayer<FloatType::FP32>::ReformerLayer(
//     py::dict &weights, py::dict &config, cudaStream_t &stream, cublasHandle_t &cublasHandle);


template<FloatType fp>
void ReformerLayer<fp>::infer(
    const T *pre_output, const T *hiddens, const int *padding_mask, T *output,
    int batch_size, int batch_seq_len) 
{
    std::cout << "ReformerLayer infer" << std::endl;
}
template void ReformerLayer<FloatType::FP32>::infer(
    const T *pre_output, const T *hiddens, const int *padding_mask, T *output,
    int batch_size, int batch_seq_len);


template<FloatType fp>
ReformerModel<fp>::ReformerModel(py::dict &weights, py::dict &config) {
    _word_embed_weight = nullptr;
    _pos_embed_weight_0 = nullptr;
    _pos_embed_weight_1 = nullptr;
    std::cout << "ReformerModel construct" << std::endl;
}
template ReformerModel<FloatType::FP32>::ReformerModel(py::dict &weights, py::dict &config);

template<FloatType fp>
ReformerModel<fp>::~ReformerModel() {
}
template ReformerModel<FloatType::FP32>::~ReformerModel();


template<FloatType fp>
void ReformerModel<fp>::infer_one_batch(
    const int *input_ids, T *output, int pad_id, int batch_size, int batch_seq_len)
{
    std::cout << "ReformerModel infer" << std::endl;
}
template void ReformerModel<FloatType::FP32>::infer_one_batch(
    const int *input_ids, T *output, int pad_id, int batch_size, int batch_seq_len);



}
