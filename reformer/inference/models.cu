#include "kernels.cuh"

namespace FastReformer {

// calculate a device memory size
template<FloatType fp>
class ReformerLayer {
private:
    using traits = TypeTraits<fp>;
    using T = typename traits::DataType;
    const cudaDataType_t _cublasComputeType = traits::cublasComputeType;
    const cudaDataType_t _cublasAType = traits::cublasAType;
    const cudaDataType_t _cublasBType = traits::cublasBType;
    const cudaDataType_t _cublasCType = traits::cublasCType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;

    const int _layer_id;
    const std::string _layer_type; // "lsh", "local"
    const int _batch_size;
    const int _batch_seq_len;
    const int _hidden_size;
    const int _num_heads;
    const int _head_size;
    const int _all_head_size;
    const int _num_hashes;
    const int _num_bucket;
    const int _atten_lsh_chunk_len;
    const int _atten_lsh_num_chunks_before;
    const int _atten_lsh_num_chunks_after;
    const int _atten_local_chunk_len;
    const int _atten_local_num_chunks_before;
    const int _atten_local_num_chunks_after;
    const int _ffn_size;
    const int _ffn_chunk_size; // on the seq_len dimension
    const T _eps;
    const T _mask_value;
    const T _one;
    const T _zero;
    const T _norm_scalar;
    void *_buffer;

    thrust::device_vector<T> _d_atten_ln_weight; // [hidden_size]
    thrust::device_vector<T> _d_atten_ln_bias; // [hidden_size]
    thrust::device_vector<T> _d_atten_lsh_proj_qk_weight; // [hidden_size, all_head_size]^T
    thrust::device_vector<T> _d_atten_lsh_proj_v_weight; // [hidden_size, all_head_size]^T
    thrust::device_vector<T> _d_atten_local_proj_q_weight; // [hidden_size, all_head_size]^T
    thrust::device_vector<T> _d_atten_local_proj_k_weight; // [hidden_size, all_head_size]^T
    thrust::device_vector<T> _d_atten_local_proj_v_weight; // [hidden_size, all_head_size]^T
    thrust::device_vector<T> _d_atten_output_proj_weight; // [all_head_size, hidden_size]^T
    
    thrust::device_vector<T> _d_ffn_ln_weight; // [hidden_size]
    thrust::device_vector<T> _d_ffn_ln_bias; // [hidden_size]
    thrust::device_vector<T> _d_ffn_dense_weight; // [hidden_size, ffn_size]^T
    thrust::device_vector<T> _d_ffn_dense_bias; // [ffn_size]
    thrust::device_vector<T> _d_ffn_output_weight; // [ffn_size, hidden_size]^T
    thrust::device_vector<T> _d_ffn_output_bias; // [hidden_size]
public:
    ReformerLayer(
        py::dict &weights, py::dict &config, void *buffer, int layer_id,
        cudaStream_t &stream, cublasHandle_t &cublasHandle
    ) :
        _stream(stream),
        _cublasHandle(cublasHandle),
        _layer_id(layer_id),
        _layer_type(py2str(py::list(config["attn_layers"])[layer_id])),
        _batch_size(py2int(config["batch_size"])),
        _batch_seq_len(py2int(config["batch_seq_len"])),
        _hidden_size(py2int(config["hidden_size"])),
        _num_heads(py2int(config["num_attention_heads"])),
        _head_size(py2int(config["attention_head_size"])),
        _all_head_size(_num_heads * _head_size),
        _num_hashes(py2int(config["num_hashes"])),
        _num_bucket(py2int(config["num_buckets"])),
        _atten_lsh_chunk_len(py2int(config["lsh_attn_chunk_length"])),
        _atten_lsh_num_chunks_before(py2int(config["lsh_num_chunks_before"])),
        _atten_lsh_num_chunks_after(py2int(config["lsh_num_chunks_after"])),
        _atten_local_chunk_len(py2int(config["local_attn_chunk_length"])),
        _atten_local_num_chunks_before(py2int(config["local_num_chunks_before"])),
        _atten_local_num_chunks_after(py2int(config["local_num_chunks_after"])),
        _ffn_size(py2int(config["feed_forward_size"])),
        _ffn_chunk_size(py2int(config["chunk_size_feed_forward"])),
        _eps(static_cast<T>(py2float(config["layer_norm_eps"]))),
        _mask_value(typeid(T) == typeid(__half) ? static_cast<T>(1e-4f) : static_cast<T>(1e-9f)),
        _one(static_cast<T>(1.0f)),
        _zero(static_cast<T>(0.0f)),
        _norm_scalar(static_cast<T>(1/sqrt(_head_size))),
        _buffer(buffer)
    {
        // assign weights
        T *h_atten_ln_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.layer_norm.weight")]).request().ptr);
        T *h_atten_ln_bias = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.layer_norm.bias")]).request().ptr);
        _d_atten_ln_weight.assign(h_atten_ln_weight, h_atten_ln_weight + _hidden_size);
        _d_atten_ln_bias.assign(h_atten_ln_bias, h_atten_ln_bias + _hidden_size);
        if (_layer_type == "lsh") {
            T *h_atten_lsh_proj_qk_weight = static_cast<T*>(py::array_t<T>(weights[
                py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.self_attention.query_key.weight")]).request().ptr);
            T *h_atten_lsh_proj_v_weight = static_cast<T*>(py::array_t<T>(weights[
                py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.self_attention.value.weight")]).request().ptr);
            _d_atten_lsh_proj_qk_weight.assign(h_atten_lsh_proj_qk_weight, h_atten_lsh_proj_qk_weight + _hidden_size * _all_head_size);
            _d_atten_lsh_proj_v_weight.assign(h_atten_lsh_proj_v_weight, h_atten_lsh_proj_v_weight + _hidden_size * _all_head_size);
        }
        else { // "local"
            T *h_atten_local_proj_q_weight = static_cast<T*>(py::array_t<T>(weights[
                py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.self_attention.query.weight")]).request().ptr);
            T *h_atten_local_proj_k_weight = static_cast<T*>(py::array_t<T>(weights[
                py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.self_attention.key.weight")]).request().ptr);
            T *h_atten_local_proj_v_weight = static_cast<T*>(py::array_t<T>(weights[
                py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.self_attention.value.weight")]).request().ptr);
            _d_atten_local_proj_q_weight.assign(h_atten_local_proj_q_weight, h_atten_local_proj_q_weight + _hidden_size * _all_head_size);
            _d_atten_local_proj_k_weight.assign(h_atten_local_proj_k_weight, h_atten_local_proj_k_weight + _hidden_size * _all_head_size);
            _d_atten_local_proj_v_weight.assign(h_atten_local_proj_v_weight, h_atten_local_proj_v_weight + _hidden_size * _all_head_size);
        }
        T *h_atten_output_proj_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layers." + std::to_string(_layer_id) + ".attention.output.dense.weight")]).request().ptr);
        _d_atten_output_proj_weight.assign(h_atten_output_proj_weight, h_atten_output_proj_weight + _hidden_size * _all_head_size);

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
    
    // [bs, seq_len, hidden size] -> [bs, seq_len, all_head_size]
    void _atten_in_proj(thrust::device_vector<T> &weight, const T *in, T *out) {
        cublasGemmEx(
            _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            _all_head_size, _batch_size * _batch_seq_len, _hidden_size,
            &_one,
            thrust::raw_pointer_cast(weight.data()), _cublasAType, _hidden_size,
            in, _cublasBType, _hidden_size,
            &_zero,
            out, _cublasCType, _all_head_size,
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // [bs, seq_len, all_head_size] -> [bs, seq_len, hidden size]
    void _atten_out_proj(thrust::device_vector<T> &weight, const T *in, T *out) {
        cublasGemmEx(
            _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            _hidden_size, _batch_size * _batch_seq_len, _all_head_size,
            &_one,
            thrust::raw_pointer_cast(weight.data()), _cublasAType, _all_head_size,
            in , _cublasBType, _all_head_size,
            &_zero,
            out, _cublasCType, _hidden_size,
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // [bs, seq_len, all_head_size] -> [bs, num_heads, seq_len/chunk_len, chunk_len, head_size]
    void _atten_split_transpose(const T *in, T *out, int chunk_len) {
        atten_split_transpose_launcher<T>(
            in, _batch_size, _batch_seq_len,
            chunk_len, _num_heads, _head_size,
            out);
    }

    // [bs, num_heads, seq_len/chunk_len, chunk_len, head_size] -> [bs, seq_len, all_head_size]
    void _atten_merge_transpose(const T *in, T *out, int chunk_len) {
        atten_merge_transpose_launcher<T>(
            in, _batch_size, _batch_seq_len,
            chunk_len, _num_heads, _head_size,
            out);
    }

    void lsh_atten(T *x, const int *atten_mask, const T *random_rotations) {
        // num hashes
        T *_d_buf_temp = reinterpret_cast<T*>(_buffer);
        T *_d_buf_qk = _d_buf_temp + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_v = _d_buf_qk + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_gather_qk = _d_buf_v + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_atten_q = _d_buf_gather_qk;
        T *_d_buf_atten_k = _d_buf_atten_q + _batch_size * _batch_seq_len * _all_head_size * _num_hashes;
        T *_d_buf_gather_v = _d_buf_atten_k + _batch_size * _batch_seq_len * _all_head_size * _num_hashes;
        T *_d_buf_atten_v = _d_buf_gather_v;
        int N = _atten_lsh_num_chunks_before + _atten_lsh_num_chunks_after + 1;
        T *_d_buf_k_adjacent = _d_buf_atten_v + _batch_size * _batch_seq_len * _all_head_size * _num_hashes;
        T *_d_buf_v_adjacent = _d_buf_k_adjacent + _batch_size * _batch_seq_len * _all_head_size * _num_hashes * N;
        T *_d_buf_qk_dots = _d_buf_v_adjacent + _batch_size * _batch_seq_len * _all_head_size * _num_hashes * N;
        T *last = _d_buf_qk_dots + _batch_size * _num_heads * _num_hashes * _batch_seq_len * N * _atten_lsh_chunk_len;

        T *_d_buf_qk_expand = last;
        T *_d_buf_rrotations_expand = _d_buf_qk_expand + _batch_size * _batch_seq_len * _all_head_size * _num_hashes;
        T *_d_buf_rotated_vec = _d_buf_rrotations_expand + _batch_size * _num_bucket / 2 * _all_head_size * _num_hashes;
        int *_d_buf_bucket = reinterpret_cast<int*>(_d_buf_rotated_vec + _batch_size * _num_bucket / 2 * _num_heads * _num_hashes * _batch_seq_len);
        // _d_buf_bucket + _batch_size * _num_heads * _num_hashes * _batch_seq_len);

        int *_d_buf_indices = reinterpret_cast<int*>(last);
        int *_d_buf_sorted_bucket_idx = _d_buf_indices + _num_hashes * _batch_seq_len;
        int *_d_buf_undo_sorted_bucket_idx = _d_buf_sorted_bucket_idx + _batch_size * _num_heads * _num_hashes * _batch_seq_len;
        int *_d_buf_sorted_bucket_idx_per_hash = _d_buf_undo_sorted_bucket_idx + _batch_size * _num_heads * _num_hashes * _batch_seq_len;


        // ln
        layer_norm_launcher<T>(
            x,
            thrust::raw_pointer_cast(_d_atten_ln_weight.data()),
            thrust::raw_pointer_cast(_d_atten_ln_bias.data()),
            _eps,
            _hidden_size,
            _batch_size * _batch_seq_len * _hidden_size);

        // proj, split, transpose qk v
        // [bs, seq_len, hidden_size] -> 
        // [bs, seq_len, all_head_size] ->
        // [bs, num_heads, seq_len, head_size]
        _atten_in_proj(_d_atten_lsh_proj_qk_weight, x, _d_buf_temp);
        _atten_split_transpose(_d_buf_temp, _d_buf_qk, 1);
        _atten_in_proj(_d_atten_lsh_proj_v_weight, x, _d_buf_temp);
        _atten_split_transpose(_d_buf_temp, _d_buf_v, 1);

        // ===== hash qk vector =====
        // vector [bs, num_heads, seq_len, head_size] ->
        //        [bs, num_heads, num_hashes, seq_len, head_size]
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size * _num_heads * _num_hashes),
            [_d_buf_qk, _d_buf_qk_expand,
            _num_hashes=_num_hashes,
            size=_batch_seq_len*_head_size] __device__ (int i) {
                thrust::copy(
                    thrust::device,
                    _d_buf_qk + (i / _num_hashes) * size,
                    _d_buf_qk + (i / _num_hashes) * size + size,
                    _d_buf_qk_expand + i * size);
            }
        );
        // random_rotations [num_heads, num_hashes, head_size, num_bucket/2] ->
        //                  [bs, num_heads, num_hashes, head_size, num_bucket/2]
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size),
            [random_rotations, _d_buf_rrotations_expand,
            size=_all_head_size*_num_hashes*_num_bucket/2] __device__ (int i) {
                thrust::copy(
                    thrust::device,
                    random_rotations,
                    random_rotations + size,
                    _d_buf_rrotations_expand + i * size);
            }
        );
        // [bs, num_heads, num_hashes, seq_len, head_size] X
        // [bs, num_heads, num_hashes, head_size, num_bucket/2]
        // -> rotated_vecs [bs, num_heads, num_hashes, seq_len, num_bucket/2]
        cublasGemmStridedBatchedEx(
            _cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            _num_bucket / 2, _batch_seq_len, _head_size,
            &_one,
            _d_buf_rrotations_expand, _cublasAType, _num_bucket/2, _head_size * _num_bucket / 2,
            _d_buf_qk_expand, _cublasBType, _head_size, _batch_seq_len * _head_size,
            &_zero,
            _d_buf_rotated_vec, _cublasCType, _num_bucket / 2, _batch_seq_len * _num_bucket / 2,
            _batch_size * _num_heads * _num_hashes,
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // argmax mask offset
        // [bs, num_heads, num_hashes, seq_len, num_bucket/2]
        // -> [bs, num_heads, num_hashes, seq_len, num_bucket]
        // -> [bs, num_heads, num_hashes, seq_len]
        lsh_bucket_argmax_mask_offset_launcher(
            _d_buf_rotated_vec, atten_mask, _batch_size, _num_heads,
            _num_hashes, _batch_seq_len, _num_bucket,
            _d_buf_bucket);

        // ===== sort =====
        // sort buckets
        // [bs, num_heads, num_hashes * seq_len]
        // arange
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size * _num_heads),
            [_d_buf_sorted_bucket_idx, size=_num_hashes*_batch_seq_len] __device__ (int i) {
                thrust::sequence(
                    thrust::device,
                    _d_buf_sorted_bucket_idx + i * size,
                    _d_buf_sorted_bucket_idx + (i + 1) * size);
            }
        );
        // sort
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size * _num_heads),
            [_d_buf_sorted_bucket_idx, _d_buf_bucket,
            size=_num_hashes*_batch_seq_len] __device__ (int i) {
                thrust::stable_sort_by_key(
                    thrust::device,
                    _d_buf_bucket + i * size,
                    _d_buf_bucket + (i + 1) * size,
                    _d_buf_sorted_bucket_idx + i * size);
            }
        );
        // arange
        thrust::sequence(
            thrust::device,
            _d_buf_indices,
            _d_buf_indices + _num_hashes * _batch_seq_len);
        // scatter
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size * _num_heads),
            [_d_buf_undo_sorted_bucket_idx, _d_buf_sorted_bucket_idx, _d_buf_indices,
            size=_num_hashes*_batch_seq_len] __device__ (int i) {
                thrust::scatter(
                    thrust::device,
                    _d_buf_indices,
                    _d_buf_indices + size,
                    _d_buf_sorted_bucket_idx + i * size,
                    _d_buf_undo_sorted_bucket_idx + i * size);
            }
        );
        // sorted_bucket_idx_per_hash
        thrust::transform(
            thrust::device,
            _d_buf_sorted_bucket_idx,
            _d_buf_sorted_bucket_idx + _batch_size * _num_heads * _num_hashes * _batch_seq_len,
            _d_buf_sorted_bucket_idx_per_hash,
            thrust::placeholders::_1 % _batch_seq_len
        );

        // gather
        // qk/v vectors [bs, num_heads, seq_len, head_size]
        //_d_buf_sorted_bucket_idx_per_hash [bs, num_heads, num_hashes * seq_len]
        // -> [bs, num_heads, num_hashes*seq_len, head_size]
        lsh_gather_by_expansion_launcher(
            _d_buf_qk, _d_buf_sorted_bucket_idx_per_hash, _batch_size,
            _num_heads, _num_hashes, _batch_seq_len, _head_size,
            _d_buf_gather_qk);
        lsh_gather_by_expansion_launcher(
            _d_buf_v, _d_buf_sorted_bucket_idx_per_hash, _batch_size,
            _num_heads, _num_hashes, _batch_seq_len, _head_size,
            _d_buf_gather_v);

        // len norm key
        lsh_len_norm_launcher(
            _d_buf_gather_qk, _head_size,
            _batch_size * _all_head_size * _num_hashes * _batch_seq_len, _norm_scalar,
            _d_buf_atten_k);
        //_d_buf_atten_q = _d_buf_gather_qk;
        //_d_buf_atten_v = _d_buf_gather_v;

        // ===== atten =====
        // kv look adjacent
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, head_size] ->
        // [bs, num_heads, num_hashes*seq_len/chunk_len, N * chunk_len, head_size]
        look_adjacent_launcher(
            _d_buf_atten_k, _batch_size, _num_heads, _num_hashes * _batch_seq_len / _atten_lsh_chunk_len,
            _atten_lsh_chunk_len, _head_size,
            _atten_lsh_num_chunks_before, _atten_lsh_num_chunks_after,
            _d_buf_k_adjacent);
        look_adjacent_launcher(
            _d_buf_atten_v, _batch_size, _num_heads, _num_hashes * _batch_seq_len / _atten_lsh_chunk_len,
            _atten_lsh_chunk_len, _head_size,
            _atten_lsh_num_chunks_before, _atten_lsh_num_chunks_after,
            _d_buf_v_adjacent);

        // matmul qk
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, head_size] X
        // [bs, num_heads, num_hashes*seq_len/chunk_len, (N * chunk_len, head_size) ^ T] ->
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len,  N * chunk_len]
        // N = Num_bef + Num_aft + 1
        cublasGemmStridedBatchedEx(
            _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            N * _atten_lsh_chunk_len, _atten_lsh_chunk_len, _head_size,
            &_one,
            _d_buf_k_adjacent, _cublasAType, _head_size, N * _atten_lsh_chunk_len * _head_size,
            _d_buf_atten_q , _cublasBType, _head_size, _atten_lsh_chunk_len * _head_size,
            &_zero,
            _d_buf_qk_dots, _cublasCType, N * _atten_lsh_chunk_len, _atten_lsh_chunk_len * N * _atten_lsh_chunk_len,
            _batch_size * _num_heads * _num_hashes * _batch_seq_len / _atten_lsh_chunk_len, // batch count
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        // q idx, k idx
        // mask and self mask
        // softmax
        // matmul

        // ==== reverse sort =====
        // reverse
        // sum up hash rounds

        // out and merge

        thrust::copy(
            thrust::device,
            _d_buf_qk_dots,
            _d_buf_qk_dots + _batch_size * _num_heads * _num_hashes * _batch_seq_len * _head_size,
            x);
    }

    /**
     * @param x [bs, seq_len, hidden_size]
     * @param atten_mask [bs, seq_len]
     */
    void local_atten(T *x, const int *atten_mask) {
        T *_d_buf_temp = reinterpret_cast<T*>(_buffer);
        T *_d_buf_q = _d_buf_temp + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_k = _d_buf_q + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_v = _d_buf_k + _batch_size * _batch_seq_len * _all_head_size;
        int N = _atten_local_num_chunks_before + _atten_local_num_chunks_after + 1;
        T *_d_buf_k_adjacent = _d_buf_v + _batch_size * _batch_seq_len * _all_head_size;
        T *_d_buf_v_adjacent = _d_buf_k_adjacent + _batch_size * _batch_seq_len  * _all_head_size * N;
        int *_d_buf_mask_adjacent = reinterpret_cast<int*>(_d_buf_v_adjacent + _batch_size * _batch_seq_len  * _all_head_size * N);
        T *_d_buf_qk_dots = reinterpret_cast<T*>(_d_buf_mask_adjacent + _batch_size * _batch_seq_len * N);
        T *_d_buf_out_v = _d_buf_q;

        // ln
        layer_norm_launcher<T>(
            x,
            thrust::raw_pointer_cast(_d_atten_ln_weight.data()),
            thrust::raw_pointer_cast(_d_atten_ln_bias.data()),
            _eps,
            _hidden_size,
            _batch_size * _batch_seq_len * _hidden_size);

        // proj, split, transpose q k v
        // [bs, seq_len, hidden_size] -> 
        // [bs, seq_len, all_head_size] ->
        // [bs, num_heads, seq_len/chunk_len, chunk_len, head_size]
        _atten_in_proj(_d_atten_local_proj_q_weight, x, _d_buf_temp);
        _atten_split_transpose(_d_buf_temp, _d_buf_q, _atten_local_chunk_len);
        _atten_in_proj(_d_atten_local_proj_k_weight, x, _d_buf_temp);
        _atten_split_transpose(_d_buf_temp, _d_buf_k, _atten_local_chunk_len);
        _atten_in_proj(_d_atten_local_proj_v_weight, x, _d_buf_temp);
        _atten_split_transpose(_d_buf_temp, _d_buf_v, _atten_local_chunk_len);

        // look adjacent for k v
        // [bs, num_heads, seq_len/chunk_len, chunk_len, head_size] ->
        // [bs, num_heads, seq_len/chunk_len, (Num_bef + Num_aft + 1) * chunk_len, head_size]
        look_adjacent_launcher(
            _d_buf_k, _batch_size, _num_heads, _batch_seq_len / _atten_local_chunk_len,
            _atten_local_chunk_len, _head_size,
            _atten_local_num_chunks_before, _atten_local_num_chunks_after,
            _d_buf_k_adjacent);
        look_adjacent_launcher(
            _d_buf_v, _batch_size, _num_heads, _batch_seq_len / _atten_local_chunk_len,
            _atten_local_chunk_len, _head_size,
            _atten_local_num_chunks_before, _atten_local_num_chunks_after,
            _d_buf_v_adjacent);
        
        // [bs, n_chunks, chunk_len] ->
        // [bs, n_chunks, (Num_bef + Num_aft + 1) * chunk_len]
        look_adjacent_launcher(
            atten_mask, _batch_size, 1, _batch_seq_len / _atten_local_chunk_len,
            _atten_local_chunk_len, 1,
            _atten_local_num_chunks_before, _atten_local_num_chunks_after,
            _d_buf_mask_adjacent);

        // matmul q k
        // B [bs, num_heads, seq_len/chunk_len, chunk_len, head_size] X
        // A [bs, num_heads, seq_len/chunk_len, (N * chunk_len, head_size) ^ T] ->
        // C [bs, num_heads, seq_len/chunk_len, chunk_len,  N * chunk_len]
        // N = Num_bef + Num_aft + 1
        cublasGemmStridedBatchedEx(
            _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            N * _atten_local_chunk_len, _atten_local_chunk_len, _head_size,
            &_norm_scalar,
            _d_buf_k_adjacent, _cublasAType, _head_size, N * _atten_local_chunk_len * _head_size,
            _d_buf_q , _cublasBType, _head_size, _atten_local_chunk_len * _head_size,
            &_zero,
            _d_buf_qk_dots, _cublasCType, N * _atten_local_chunk_len, _atten_local_chunk_len * N * _atten_local_chunk_len,
            _batch_size * _num_heads * _batch_seq_len / _atten_local_chunk_len, // batch count
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // add mask
        local_atten_enc_mask_launcher(
            _d_buf_qk_dots, _d_buf_mask_adjacent, _batch_size, _num_heads,
            _batch_seq_len / _atten_local_chunk_len, _atten_local_chunk_len, N);

        // softmax
        // [bs, num_heads, seq_len/chunk_len, chunk_len, N * chunk_len]
        softmax_launcher<T>(
            _d_buf_qk_dots,
            N * _atten_local_chunk_len,
            _batch_size * _num_heads * _batch_seq_len * _atten_local_chunk_len * N);

        // matmul
        // B [bs, num_heads, seq_len/chunk_len, chunk_len, N * chunk_len] X
        // A [bs, num_heads, seq_len/chunk_len, N * chunk_len, head_size] ->
        // C [bs, num_heads, seq_len/chunk_len, chunk_len, head_size]
        // N = Num_bef + Num_aft + 1
        cublasGemmStridedBatchedEx(
            _cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            _head_size, _atten_local_chunk_len, N * _atten_local_chunk_len,
            &_one,
            _d_buf_v_adjacent, _cublasAType, _head_size, N * _atten_local_chunk_len * _head_size,
            _d_buf_qk_dots, _cublasBType, N * _atten_local_chunk_len, _atten_local_chunk_len * N * _atten_local_chunk_len,
            &_zero,
            _d_buf_out_v, _cublasCType, _head_size, _atten_local_chunk_len * _head_size,
            _batch_size * _num_heads * _batch_seq_len / _atten_local_chunk_len,
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // transpose back
        // [bs, num_heads, seq_len/chunk_len, chunk_len, head_size] ->
        // [bs, seq_len, all_head_size]
        _atten_merge_transpose(_d_buf_out_v, _d_buf_temp, _atten_local_chunk_len);

        // out dense
        // [bs, seq_len, all_head_size] -> [bs, seq_len, hidden_size]
        _atten_out_proj(_d_atten_output_proj_weight, _d_buf_temp, x);

    }

    /**
     * @param x [bs * seq_len, hidden_size]
     */
    void chunk_ffn(T *x)
    {
        // ffn buffer size : bs * ffn_chunk_size * ffn_size
        T *_d_buf_ffn = reinterpret_cast<T*>(_buffer);
        
        assert(_batch_seq_len % _ffn_chunk_size == 0);
        int num_chunks = _batch_seq_len / _ffn_chunk_size;
        for (int i = 0; i < num_chunks; i ++) {
            T *chunked_x = x + i * _batch_size * _ffn_chunk_size * _hidden_size;

            layer_norm_launcher<T>(
                chunked_x,
                thrust::raw_pointer_cast(_d_ffn_ln_weight.data()),
                thrust::raw_pointer_cast(_d_ffn_ln_bias.data()),
                _eps,
                _hidden_size,
                _batch_size * _ffn_chunk_size * _hidden_size);
            
            cublasGemmEx(
                _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                _ffn_size, _batch_size * _ffn_chunk_size, _hidden_size,
                &_one,
                thrust::raw_pointer_cast(_d_ffn_dense_weight.data()), _cublasAType, _hidden_size,
                chunked_x, _cublasBType, _hidden_size,
                &_zero,
                _d_buf_ffn, _cublasCType, _ffn_size,
                _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            
            bias_relu_launcher<T>(
                _d_buf_ffn,
                thrust::raw_pointer_cast(_d_ffn_dense_bias.data()),
                _ffn_size,
                _batch_size * _ffn_chunk_size * _ffn_size);
            
            cublasGemmEx(
                _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                _hidden_size, _batch_size * _ffn_chunk_size, _ffn_size,
                &_one,
                thrust::raw_pointer_cast(_d_ffn_output_weight.data()), _cublasAType, _ffn_size,
                _d_buf_ffn, _cublasBType, _ffn_size,
                &_zero,
                chunked_x, _cublasCType, _hidden_size,
                _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);


            add_bias_launcher<T>(
                chunked_x,
                thrust::raw_pointer_cast(_d_ffn_output_bias.data()),
                _hidden_size,
                _batch_size * _ffn_chunk_size * _hidden_size);

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
    const int _num_heads;
    const int _head_size;
    const int _all_head_size;
    const int _num_hashes;
    const int _num_bucket;
    const int _atten_lsh_chunk_len;
    const int _atten_lsh_num_chunks_before;
    const int _atten_lsh_num_chunks_after;
    const int _atten_local_chunk_len;
    const int _atten_local_num_chunks_before;
    const int _atten_local_num_chunks_after;
    const int _ffn_size;
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
        _num_heads(py2int(config["num_attention_heads"])),
        _head_size(py2int(config["attention_head_size"])),
        _all_head_size(_num_heads * _head_size),
        _num_hashes(py2int(config["num_hashes"])),
        _num_bucket(py2int(config["num_buckets"])),
        _atten_lsh_chunk_len(py2int(config["lsh_attn_chunk_length"])),
        _atten_lsh_num_chunks_before(py2int(config["lsh_num_chunks_before"])),
        _atten_lsh_num_chunks_after(py2int(config["lsh_num_chunks_after"])),
        _atten_local_chunk_len(py2int(config["local_attn_chunk_length"])),
        _atten_local_num_chunks_before(py2int(config["local_num_chunks_before"])),
        _atten_local_num_chunks_after(py2int(config["local_num_chunks_after"])),
        _ffn_size(py2int(config["feed_forward_size"])),
        _ffn_chunk_size(py2int(config["chunk_size_feed_forward"]))
    {
        _stream = 0;
        // cudaStreamCreate(&_stream);
        cublasCreate(&_cublasHandle);
        cublasSetStream(_cublasHandle, _stream);
        int local_N = _atten_local_num_chunks_before + _atten_local_num_chunks_after + 1;
        int lsh_N = _atten_lsh_num_chunks_before + _atten_lsh_num_chunks_after + 1;
        int buf_size = 
            sizeof(int) * _batch_size * _batch_seq_len + // padding_mask
            sizeof(T) * 2 * _batch_size * _batch_seq_len + _hidden_size + // encoder
            max(
                sizeof(T) * 4 * _batch_size * _batch_seq_len * _all_head_size +
                sizeof(T) * 2 * _batch_size * _batch_seq_len  * _all_head_size * local_N +
                sizeof(int) * _batch_size * _batch_seq_len * local_N +
                sizeof(T) * _batch_size * _num_heads * _batch_seq_len * local_N * _atten_local_chunk_len, // local atten

                sizeof(T) * 3 * _batch_size * _batch_seq_len * _all_head_size +
                sizeof(T) * 3 * _batch_size * _batch_seq_len * _all_head_size * _num_hashes +
                sizeof(T) * 2 * _batch_size * _batch_seq_len * _all_head_size * _num_hashes * lsh_N +
                sizeof(T) * _batch_size * _num_heads * _num_hashes * _batch_seq_len * lsh_N * _atten_lsh_chunk_len +
                max(sizeof(T) * _batch_size * _batch_seq_len * _all_head_size * _num_hashes +
                    sizeof(T) * _batch_size * _num_bucket / 2 * _all_head_size * _num_hashes +
                    sizeof(T) * _batch_size * _batch_seq_len * _num_heads * _num_bucket / 2 * _num_hashes +
                    sizeof(int) * _batch_size * _num_heads * _num_hashes * _batch_seq_len,

                    sizeof(int) * 3 * _batch_size * _num_heads * _num_hashes * _batch_seq_len +
                    sizeof(int) * _num_hashes * _batch_seq_len) // lsh atten
                // sizeof(T) * _batch_size * _ffn_chunk_size * _ffn_size // ffn
            );
        cudaMalloc(&_buffer, buf_size);
        _d_padding_mask = reinterpret_cast<int*>(_buffer);
        _d_buf_0 = reinterpret_cast<T*>(_d_padding_mask + _batch_size * _batch_seq_len);
        _d_buf_1 = _d_buf_0 + _batch_size * _batch_seq_len + _hidden_size;
        T *d_buf_layer = _d_buf_1 + _batch_size * _batch_seq_len + _hidden_size;
        // encoder
        int layer_num = py2int(config["num_hidden_layers"]);
        for (int layer_idx = 0; layer_idx < layer_num; layer_idx ++) {
            enc_layers.emplace_back(weights, config, d_buf_layer, layer_idx, _stream, _cublasHandle);
        }

    }
    ~ReformerModel() {
        // free buffer
        // cudaStreamDestroy(_stream);
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
