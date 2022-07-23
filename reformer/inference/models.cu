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

    // const int _num_heads;
    // const int _head_size;
    // const int _all_head_size;
    const int _layer_id;
    const int _batch_size;
    const int _batch_seq_len;
    const int _hidden_size;
    const int _ffn_size;
    const int _ffn_chunk_size; // on the seq_len dimension
    const T _eps;
    // const int _head_dim;
    // const int _head_num;


    // const T *_d_atten_ln_weight;
    // const T *_d_atten_ln_bias;
    // const T atten_mask_value; // or __constant__
    // const T *_d_atten_q_weight;
    // const T *_d_atten_k_weight;
    // const T *_d_atten_v_weight;
    // // const T *_d_atten_qkv_weight;
    // // const T *_d_atten_qk_v_weight;
    // const T *_d_atten_output_weight;
    
    thrust::device_vector<T> _d_ffn_ln_weight; // [hidden_size]
    thrust::device_vector<T> _d_ffn_ln_bias; // [hidden_size]
    thrust::device_vector<T> _d_ffn_dense_weight; // [ffn_size, hidden_size]
    thrust::device_vector<T> _d_ffn_dense_bias; // [ffn_size]
    thrust::device_vector<T> _d_ffn_output_weight; // [ffn_size, hidden_size]
    thrust::device_vector<T> _d_ffn_output_bias; // [hidden_size]
    void _set_weights(py::dict &weights, py::dict &config, int layer_id) {
        T *h_ffn_ln_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.layer_norm.weight")]).request().ptr);
        T *h_ffn_ln_bias = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.layer_norm.bias")]).request().ptr);
        T *h_ffn_dense_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.dense.dense.weight")]).request().ptr);
        T *h_ffn_dense_bias = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.dense.dense.bias")]).request().ptr);
        T *h_ffn_output_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.output.dense.weight")]).request().ptr);
        T *h_ffn_output_bias = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".feed_forward.output.dense.bias")]).request().ptr);

        _d_ffn_ln_weight.assign(h_ffn_ln_weight, h_ffn_ln_weight + _hidden_size);
        _d_ffn_ln_bias.assign(h_ffn_ln_bias, h_ffn_ln_bias + _hidden_size);
        _d_ffn_dense_weight.assign(h_ffn_dense_weight, h_ffn_dense_weight + _ffn_size * _hidden_size);
        _d_ffn_dense_bias.assign(h_ffn_dense_bias, h_ffn_dense_bias + _ffn_size);
        _d_ffn_output_weight.assign(h_ffn_output_weight, h_ffn_output_weight + _ffn_size * _hidden_size);
        _d_ffn_output_bias.assign(h_ffn_output_bias, h_ffn_output_bias + _hidden_size);
    }


    T * _d_buf_ffn_0; // [bs, ffn_chunk_size, hidden_size]
    T * _d_buf_ffn_1; // [bs, ffn_chunk_size, hidden_size]
    void _set_buffer(void *buffer) {
        // TODO atten buf

        _d_buf_ffn_0 = reinterpret_cast<T*>(buffer);
        _d_buf_ffn_1 = _d_buf_ffn_0 + _batch_size * _ffn_chunk_size * _hidden_size;
    }
    
public:
    ReformerLayer(
        py::dict &weights, py::dict &config, void *buffer, int layer_id,
        cudaStream_t &stream, cublasHandle_t &cublasHandle
    ) :
        _stream(stream),
        _cublasHandle(cublasHandle),
        _layer_id(layer_id),
        _batch_size(py2int(config["batch_size"])),
        _batch_seq_len(py2int(config["batch_seq_len"])),
        _hidden_size(py2int(config["hidden_size"])),
        _ffn_size(py2int(config["feed_forward_size"])),
        _ffn_chunk_size(py2int(config["chunk_size_feed_forward"])),
        _eps(static_cast<T>(py2float(config["layer_norm_eps"])))
    {
        _set_weights(weights, config, layer_id);
        _set_buffer(buffer);
    }
    void lsh_atten() {}
    void local_atten(
        const T *prev_hiddens, const T *hiddens, T *out,
        int batch_size, int batch_seq_len, int chunk_len)
    {
        //chunk size, chunk seq len 矩阵乘法的维度/chunk len?

        //ln => infer

        // qkv proj
        // split to multi head
        // /(d)1/2

        // indices -> atten mask


        // out dense => infer
    }

    /**
     * @param atten_out [bs * seq_len, hidden_size]
     * @param out [bs * seq_len, hidden_size]
     */
    void chunk_ffn(const T *atten_out, T *out)
    {
        assert(_batch_seq_len % _ffn_chunk_size == 0);
        int num_chunks = _batch_seq_len / _ffn_chunk_size;
        int chunk_size = _batch_size * _ffn_chunk_size * _hidden_size;
        for (int i = 0; i < num_chunks; i ++) {
            const T *chunk_atten_out = atten_out + i * chunk_size;
            T *chunk_out = out + i * chunk_size;

            layer_norm_launcher<T>(
                chunk_atten_out,
                thrust::raw_pointer_cast(_d_ffn_ln_weight.data()),
                thrust::raw_pointer_cast(_d_ffn_ln_bias.data()),
                _eps, _hidden_size, chunk_size,
                _d_buf_ffn_0);
            
            // dense1 cublas buf0 -> buf1
            
            bias_relu_launcher<T>(
                _d_buf_ffn_1,
                thrust::raw_pointer_cast(.data()),
                
            );
            // dense2 cublas
            bias_relu_launcher<T>(
                _d_buf_ffn_1
            );

        }
    }
    // atten + revnet(内部实现) + chuncked LSH
    void infer(
        const T *pre_output, const T *hiddens, const T *padding_mask, T *output,
        int batch_size, int batch_seq_len)
    {

    }
};
template class ReformerLayer<FloatType::FP32>;


template<FloatType fp>
class ReformerModel {
private:
    using T = typename TypeTraits<fp>::DataType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;
    thrust::device_vector<T> _d_word_embed_weight;
    thrust::device_vector<T> _d_pos_embed_weight_0;
    thrust::device_vector<T> _d_pos_embed_weight_1;
    thrust::device_vector<T> _d_enc_ln_weight;
    thrust::device_vector<T> _d_enc_ln_bias;
    const int _batch_size;
    const int _batch_seq_len;
    const int _hidden_size;
    const int _ffn_chunk_size;

    void *_buffer;
    int *_d_padding_mask; // [bs, seq_len]
    T *_d_buf_0; // [bs, seq_len, hidden_size]
    T *_d_buf_1;

public:
    std::vector<ReformerLayer<fp>> enc_layers;
    ReformerModel(py::dict &weights, py::dict &config) :
        _batch_size(py2int(config["batch_size"])),
        _batch_seq_len(py2int(config["batch_seq_len"])),
        _hidden_size(py2int(config["hidden_size"])),
        _ffn_chunk_size(py2int(config["chunk_size_feed_forward"]))
    {
        cudaStreamCreate(&_stream);
        cublasCreate(&_cublasHandle);
        cublasSetStream(_cublasHandle, _stream);
        // TODO malloc buffer

        int buf_size = 
            sizeof(int) * _batch_size * _batch_seq_len + // padding_mask
            sizeof(T) * 2 * _batch_size * _batch_seq_len + _hidden_size + // encoder
            sizeof(T) * max(
                // 0, // atten
                2 * _batch_size * _ffn_chunk_size * _hidden_size // ffn
            );
        cudaMalloc(&_buffer, buf_size);
        _d_padding_mask = reinterpret_cast<int*>(_buffer);
        _d_buf_0 = reinterpret_cast<T*>(_d_padding_mask + _batch_size * _batch_seq_len);
        _d_buf_1 = _d_buf_0 + _batch_size * _batch_seq_len + _hidden_size;
        T *d_buf_layer = _d_buf_1 + _batch_size * _batch_seq_len + _hidden_size;
        // encoder
        int layer_num = py2int(config["num_hidden_layers"]);
        for (int layer_idx = 0; layer_idx < layer_num; layer_idx ++) {
            enc_layers.emplace_back(weights, config, d_buf_layer, layer_idx, stream, cublasHandle);
        }

    }
    ~ReformerModel() {
        // free buffer
        cudaStreamDestroy(_stream);
        cublasDestroy(_cublasHandle);
        cudaFree(_buffer);
    }
    void infer_one_batch(
        const int *input_ids, T *output, int pad_id,
        int batch_size, int batch_seq_len)
    {
        // embedding -> buf0, padding_mask
        // enc_layers, buf0->buf1, buf1->buf0


        std::cout << "ReformerModel infer" << std::endl;
    }
};
template class ReformerModel<FloatType::FP32>;


} // namespace FastReformer
