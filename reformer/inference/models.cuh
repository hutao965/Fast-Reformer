#pragma once

#include "kernels.cuh"

namespace FastReformer {

// calculate a device memory size
template<FloatType fp>
class ReformerLayer {
private:
    using traits = TypeTraits<fp>;
    using T = typename traits::DataType;
    const cudaDataType_t cublasComputeType = traits::cublasComputeType;
    const cudaDataType_t cublasAType = traits::cublasAType;
    const cudaDataType_t cublasBType = traits::cublasBType;
    const cudaDataType_t cublasCType = traits::cublasCType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;
    const T *_d_atten_ln_weight;
    const T *_d_atten_ln_bias;
    const T atten_mask_value; // or __constant__
    const T *_d_atten_q_weight;
    const T *_d_atten_k_weight;
    const T *_d_atten_v_weight;
    // const T *_d_atten_qkv_weight;
    // const T *_d_atten_qk_v_weight;
    const T *_d_atten_output_weight;
    const T *_d_ffn_ln_weight;
    const T *_d_ffn_ln_bias;
    const T *_d_ffn_dense_weight;
    const T *_d_ffn_dense_bias;
    const T *_d_ffn_output_weight;
    const T *_d_ffn_output_bias;
    const bool _share_QK;
    const int _hidden_size;
    const int _head_dim;
    const int _head_num;
    void _set_buffer(void *buffer);
    // num_hashes -> multi rounds
    void _lsh_atten();
    void _local_atten();
    void _chunk_reformer_ffn(const T *in, T *out);
public:
    // hold a device memory, dont need release
    ReformerLayer(py::dict &weights, py::dict &config,
                  cudaStream_t &stream, cublasHandle_t &cublasHandle);
    ~ReformerLayer();
    // atten + revnet(内部实现) + chuncked LSH
    void infer(const T *pre_output, const T *hiddens, const int *padding_mask, T *output,
               int batch_size, int batch_seq_len);
};


template<FloatType fp>
class ReformerEncoder {
private:
    using T = typename TypeTraits<fp>::DataType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;
    const T *_d_ln_weight;
    const T *_d_ln_bias;
    void _set_buffer(void *buffer);
public:
    std::vector<std::shared_ptr<ReformerLayer<fp>>> layers;
    ReformerEncoder(py::dict &weights, py::dict &config,
                    cudaStream_t &stream, cublasHandle_t &cublasHandle);
    ~ReformerEncoder();
    void infer(const T *hiddens, int *padding_mask, T *output,
               int batch_size, int batch_seq_len);
};


template<FloatType fp>
class ReformerModel {
private:
    using T = typename TypeTraits<fp>::DataType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;
    int *_d_padding_mask;
    const T *_word_embed_weight;
    const T *_pos_embed_weight_0;
    const T *_pos_embed_weight_1;

    void *_buffer;
    void _embedding();
public:
    std::shared_ptr<ReformerEncoder<fp>> encoder;
    // malloc buffer & copy from host to device 
    ReformerModel(py::dict &weights, py::dict &config);
    ~ReformerModel();
    void infer_one_batch(const int *input_ids, T *output, int pad_id,
                         int batch_size, int batch_seq_len);
};


} // namespace FastReformer
