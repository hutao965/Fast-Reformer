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
    const T _mask_value = traits::mask_value;
    const T _self_mask_value = traits::self_mask_value;
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
    const T _one;
    const T _zero;
    const T _norm_scalar;
    T *_revnet_buffer;
    T *_atten_buffer;
    T *_ffn_buffer;

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
        _one(static_cast<T>(1.0f)),
        _zero(static_cast<T>(0.0f)),
        _norm_scalar(static_cast<T>(1/sqrt(_head_size)))
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

        _revnet_buffer = reinterpret_cast<T*>(buffer);
        _atten_buffer = _revnet_buffer + _batch_size * _batch_seq_len * _hidden_size;
        _ffn_buffer = _atten_buffer;
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
        atten_split_transpose_launcher(
            in, _batch_size, _batch_seq_len,
            chunk_len, _num_heads, _head_size,
            out);
    }

    // [bs, num_heads, seq_len/chunk_len, chunk_len, head_size] -> [bs, seq_len, all_head_size]
    void _atten_merge_transpose(const T *in, T *out, int chunk_len) {
        atten_merge_transpose_launcher(
            in, _batch_size, _batch_seq_len,
            chunk_len, _num_heads, _head_size,
            out);
    }

    void lsh_atten(T *x, const int *atten_mask, const T *random_rotations) {
        // num hashes
        T *_d_buf_temp = reinterpret_cast<T*>(_atten_buffer);
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
        T *_d_buf_logits = _d_buf_qk_dots + _batch_size * _num_heads * _num_hashes * _batch_seq_len * N * _atten_lsh_chunk_len;
        T *last = _d_buf_logits + _batch_size * _num_heads * _num_hashes * _batch_seq_len;

        T *_d_buf_qk_expand = last;
        T *_d_buf_rrotations_expand = _d_buf_qk_expand + _batch_size * _batch_seq_len * _all_head_size * _num_hashes;
        T *_d_buf_rotated_vec = _d_buf_rrotations_expand + _batch_size * _num_bucket / 2 * _all_head_size * _num_hashes;
        int *_d_buf_bucket = reinterpret_cast<int*>(_d_buf_rotated_vec + _batch_size * _num_bucket / 2 * _num_heads * _num_hashes * _batch_seq_len);
        // _d_buf_bucket + _batch_size * _num_heads * _num_hashes * _batch_seq_len);

        int *_d_buf_sorted_bucket_idx = reinterpret_cast<int*>(last);
        int *_d_buf_undo_sorted_bucket_idx = _d_buf_sorted_bucket_idx + _batch_size * _num_heads * _num_hashes * _batch_seq_len;
        int *_d_buf_q_idx = _d_buf_sorted_bucket_idx;
        int *_d_buf_kv_idx = _d_buf_undo_sorted_bucket_idx + _batch_size * _num_heads * _num_hashes * _batch_seq_len;
        // _batch_size * _num_heads * _num_hashes * _batch_seq_len * N

        T *_d_buf_rev_out = _d_buf_temp + _batch_size * _all_head_size * _num_hashes * _batch_seq_len;
        T *_d_buf_rev_logits = _d_buf_rev_out + _batch_size * _all_head_size * _num_hashes * _batch_seq_len;
        T *_d_buf_sum_hashes_res = _d_buf_rev_logits + _batch_size * _num_heads * _num_hashes * _batch_seq_len;


        // ln
        layer_norm_launcher(
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
        repeat_launcher(
            _d_buf_qk,
            _batch_size * _num_heads,
            _batch_seq_len * _head_size,
            _num_hashes,
            _d_buf_qk_expand);

        // random_rotations [num_heads, num_hashes, head_size, num_bucket/2] ->
        //                  [bs, num_heads, num_hashes, head_size, num_bucket/2]
        repeat_launcher(
            random_rotations,
            1,
            _all_head_size*_num_hashes*_num_bucket/2,
            _batch_size,
            _d_buf_rrotations_expand);


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
        // arange
        // [bs, num_heads, num_hashes * seq_len]
        arrange_last_launcher(
            _d_buf_sorted_bucket_idx,
            _num_hashes * _batch_seq_len,
            _batch_size * _num_heads * _num_hashes * _batch_seq_len);

        // sort
        // TODO cub sort
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

        // scatter undo_idx from arrange(num_hashes*seq_len)
        // sorted_idx %= seq_len
        lsh_scatter_undo_idx_launcher(
            _d_buf_sorted_bucket_idx, _d_buf_undo_sorted_bucket_idx,
            _batch_size, _num_heads, _num_hashes, _batch_seq_len);

        // gather
        // qk/v vectors [bs, num_heads, seq_len, head_size]
        //_d_buf_sorted_bucket_idx [bs, num_heads, num_hashes * seq_len]
        // -> [bs, num_heads, num_hashes*seq_len, head_size]
        lsh_gather_by_expansion_launcher(
            _d_buf_qk, _d_buf_sorted_bucket_idx, _batch_size,
            _num_heads, _num_hashes, _batch_seq_len, _head_size,
            _d_buf_gather_qk);
        lsh_gather_by_expansion_launcher(
            _d_buf_v, _d_buf_sorted_bucket_idx, _batch_size,
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
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, N * chunk_len]
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
        
        // mask
        // _d_buf_q_idx = _d_buf_sorted_bucket_idx [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len]
        // _d_buf_kv_idx [bs, num_heads, num_hashes*seq_len/chunk_len, N * chunk_len]
        look_adjacent_launcher(
            _d_buf_q_idx, _batch_size, _num_heads, _num_hashes * _batch_seq_len / _atten_lsh_chunk_len,
            _atten_lsh_chunk_len, 1,
            _atten_lsh_num_chunks_before, _atten_lsh_num_chunks_after,
            _d_buf_kv_idx);

        // atten mask + self mask
        lsh_enc_mask_launcher(
            _d_buf_qk_dots, _d_buf_q_idx, _d_buf_kv_idx, atten_mask,
            _mask_value, _self_mask_value, _batch_size, _num_heads, _num_hashes,
            _batch_seq_len, _atten_lsh_chunk_len, N);

        // softmax
        // _d_buf_qk_dots [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, N * chunk_len]
        // _d_buf_logits [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len]
        softmax_with_logits_launcher(
            _d_buf_qk_dots, _d_buf_logits, N * _atten_lsh_chunk_len,
            _batch_size * _num_heads * _num_hashes * _batch_seq_len * N * _atten_lsh_chunk_len);

        // matmul
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, N * chunk_len] X
        // [bs, num_heads, num_hashes*seq_len/chunk_len, N * chunk_len, head_size] ->
        // [bs, num_heads, num_hashes*seq_len/chunk_len, chunk_len, head_size]
        cublasGemmStridedBatchedEx(
            _cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            _head_size, _atten_lsh_chunk_len, N * _atten_lsh_chunk_len,
            &_one,
            _d_buf_v_adjacent, _cublasAType, _head_size, N * _atten_lsh_chunk_len * _head_size,
            _d_buf_qk_dots, _cublasBType, N * _atten_lsh_chunk_len, _atten_lsh_chunk_len * N * _atten_lsh_chunk_len,
            &_zero,
            _d_buf_temp, _cublasCType, _head_size, _atten_lsh_chunk_len * _head_size,
            _batch_size * _num_heads * _num_hashes * _batch_seq_len / _atten_lsh_chunk_len,
            _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // ==== reverse sort =====
        // gather out and logits (reverse sort)
        // _d_buf_undo_sorted_bucket_idx [bs, num_heads, num_hashes*seq_len]
        // _d_buf_temp [bs, num_heads, num_hashes*seq_len, head_size] ->
        // _d_buf_rev_out [bs, num_heads, num_hashes*seq_len, head_size]
        // _d_buf_logits [bs, num_heads, num_hashes*seq_len] ->
        // _d_buf_rev_logits [bs, num_heads, num_hashes*seq_len]
        lsh_undo_sort_launcher(
            _d_buf_undo_sorted_bucket_idx, _d_buf_temp, _d_buf_logits,
            _batch_size, _num_heads, _num_hashes, _batch_seq_len, _head_size,
            _d_buf_rev_out, _d_buf_rev_logits);

        // sum up hash
        // _d_buf_rev_out [bs, num_heads, num_hashes*seq_len, head_size]
        // _d_buf_rev_logits [bs, num_heads, num_hashes*seq_len] ->
        // _d_buf_sum_hashes_res [bs, num_heads, seq_len, head_size]
        sum_up_hashes_launcher(
            _d_buf_rev_out, _d_buf_rev_logits,
            _batch_size, _num_heads, _num_hashes,
            _batch_seq_len, _head_size,
            _d_buf_sum_hashes_res);

        // merge and transpose
        // _d_buf_sum_hashes_res [bs, num_heads, seq_len, head_size] ->
        // _d_buf_temp [bs, seq_len, all_head_size]
        atten_merge_transpose_launcher(
            _d_buf_sum_hashes_res, _batch_size, _batch_seq_len,
            1, _num_heads, _head_size,
            _d_buf_temp);
        
         // out dense
        // [bs, seq_len, all_head_size] -> [bs, seq_len, hidden_size]
        _atten_out_proj(_d_atten_output_proj_weight, _d_buf_temp, x);       
    }

    /**
     * @param x [bs, seq_len, hidden_size]
     * @param atten_mask [bs, seq_len]
     */
    void local_atten(T *x, const int *atten_mask) {
        T *_d_buf_temp = reinterpret_cast<T*>(_atten_buffer);
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
        layer_norm_launcher(
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
            _d_buf_qk_dots, _d_buf_mask_adjacent, _mask_value, _batch_size, _num_heads,
            _batch_seq_len / _atten_local_chunk_len, _atten_local_chunk_len, N);

        // softmax
        // [bs, num_heads, seq_len/chunk_len, chunk_len, N * chunk_len]
        softmax_launcher(
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
     * @param x [bs, seq_len, hidden_size]
     */
    void chunk_ffn(T *x)
    {
        // ffn buffer size : bs * ffn_chunk_size * ffn_size
        T *_d_buf_ffn = reinterpret_cast<T*>(_ffn_buffer);

        layer_norm_launcher(
            x,
            thrust::raw_pointer_cast(_d_ffn_ln_weight.data()),
            thrust::raw_pointer_cast(_d_ffn_ln_bias.data()),
            _eps,
            _hidden_size,
            _batch_size * _batch_seq_len * _hidden_size);

        int num_chunks = _batch_seq_len / _ffn_chunk_size;
        for (int i = 0; i < num_chunks; i ++) {
            T *chunked_x = x + i * _batch_size * _ffn_chunk_size * _hidden_size;

            cublasGemmEx(
                _cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                _ffn_size, _batch_size * _ffn_chunk_size, _hidden_size,
                &_one,
                thrust::raw_pointer_cast(_d_ffn_dense_weight.data()), _cublasAType, _hidden_size,
                chunked_x, _cublasBType, _hidden_size,
                &_zero,
                _d_buf_ffn, _cublasCType, _ffn_size,
                _cublasComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            bias_relu_launcher(
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
        }

        add_bias_launcher(
            x,
            thrust::raw_pointer_cast(_d_ffn_output_bias.data()),
            _hidden_size,
            _batch_size * _batch_seq_len * _hidden_size);
    }

    /**
     * @param pre_atten_output [bs, seq_len, hidden_size]
     * @param hiddens [bs, seq_len, hidden_size]
     * @param padding_mask [bs, seq_len]
     * @param random_rotations [num_heads, num_hashes, head_size, num_buckets/2]
     */
    void infer(
        T *pre_atten_output, T *hiddens, const int *padding_mask, const T *random_rotations)
    {
        int size = _batch_size * _batch_seq_len * _hidden_size;
        T *atten_output = reinterpret_cast<T*>(_revnet_buffer);
        thrust::copy(
            thrust::device,
            hiddens, hiddens + size,
            atten_output);
        
        if (_layer_type == "lsh") {
            lsh_atten(atten_output, padding_mask, random_rotations);
        }
        else { // local
            local_atten(atten_output, padding_mask);
        }

        // atten_out += pre_atten_output
        // pre_atten_output = atten_out
        add_launcher(atten_output, pre_atten_output, size);

        // hidden_states += ffn(atten_out)
        chunk_ffn(atten_output);
        add_launcher(hiddens, atten_output, size);
    }
};
template class ReformerLayer<FloatType::FP32>;


template<FloatType fp>
class ReformerModel {
private:
    using T = typename TypeTraits<fp>::DataType;
    cudaStream_t _stream;
    cublasHandle_t _cublasHandle;
    thrust::device_vector<T> _d_tok_embed_weight;
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
    const int _vocab_size;
    const int _pos_embds_dim_0;
    const int _pos_embds_dim_1;
    const int _pos_shape_0;
    const int _pos_shape_1;
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
    const T _eps;

    void *_buffer;
    int *_d_padding_mask; // [bs, seq_len]
    T *_d_buf_pre_atten_out; // [bs, seq_len, hidden_size]
    T *_d_buf_hiddens; // [bs, seq_len, hidden_size]

public:
    std::vector<ReformerLayer<fp>> enc_layers;
    ReformerModel(py::dict &weights, py::dict &config) :
        _batch_size(py2int(config["batch_size"])),
        _batch_seq_len(py2int(config["batch_seq_len"])),
        _hidden_size(py2int(config["hidden_size"])),
        _num_heads(py2int(config["num_attention_heads"])),
        _head_size(py2int(config["attention_head_size"])),
        _all_head_size(_num_heads * _head_size),
        _vocab_size(py2int(config["vocab_size"])),
        _pos_embds_dim_0(py2int(py::list(config["axial_pos_embds_dim"])[0])),
        _pos_embds_dim_1(py2int(py::list(config["axial_pos_embds_dim"])[1])),
        _pos_shape_0(py2int(py::list(config["axial_pos_shape"])[0])),
        _pos_shape_1(py2int(py::list(config["axial_pos_shape"])[1])),
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
        _eps(static_cast<T>(py2float(config["layer_norm_eps"])))
    {
        // TODO kernels set stream
        _stream = 0;
        // cudaStreamCreate(&_stream);
        cublasCreate(&_cublasHandle);
        cublasSetStream(_cublasHandle, _stream);
        
        // set buffer
        int local_N = _atten_local_num_chunks_before + _atten_local_num_chunks_after + 1;
        int lsh_N = _atten_lsh_num_chunks_before + _atten_lsh_num_chunks_after + 1;
        // TODO cal buf size from layers
        int buf_size = 
            sizeof(int) * _batch_size * _batch_seq_len + // padding_mask
            sizeof(T) * 2 * _batch_size * _batch_seq_len * _hidden_size + // encoder
            sizeof(T) * _batch_size * _batch_seq_len * _hidden_size + // revnet
            max(
                sizeof(T) * 4 * _batch_size * _batch_seq_len * _all_head_size +
                sizeof(T) * 2 * _batch_size * _batch_seq_len  * _all_head_size * local_N +
                sizeof(int) * _batch_size * _batch_seq_len * local_N +
                sizeof(T) * _batch_size * _num_heads * _batch_seq_len * local_N * _atten_local_chunk_len, // local atten

                sizeof(T) * 3 * _batch_size * _batch_seq_len * _all_head_size +
                sizeof(T) * 3 * _batch_size * _batch_seq_len * _all_head_size * _num_hashes +
                sizeof(T) * 2 * _batch_size * _batch_seq_len * _all_head_size * _num_hashes * lsh_N +
                sizeof(T) * _batch_size * _num_heads * _num_hashes * _batch_seq_len * lsh_N * _atten_lsh_chunk_len +
                sizeof(T) * _batch_size * _num_heads * _num_hashes * _batch_seq_len +
                max(sizeof(T) * _batch_size * _batch_seq_len * _all_head_size * _num_hashes +
                    sizeof(T) * _batch_size * _num_bucket / 2 * _all_head_size * _num_hashes +
                    sizeof(T) * _batch_size * _batch_seq_len * _num_heads * _num_bucket / 2 * _num_hashes +
                    sizeof(int) * _batch_size * _num_heads * _num_hashes * _batch_seq_len,

                    sizeof(int) * 2 * _batch_size * _num_heads * _num_hashes * _batch_seq_len +
                    sizeof(int) * _batch_size * _num_heads * _num_hashes * _batch_seq_len * lsh_N) // lsh atten
                // sizeof(T) * _batch_size * _ffn_chunk_size * _ffn_size // ffn
            );
        cudaMalloc(&_buffer, buf_size);
        _d_padding_mask = reinterpret_cast<int*>(_buffer);
        _d_buf_pre_atten_out = reinterpret_cast<T*>(_d_padding_mask + _batch_size * _batch_seq_len);
        _d_buf_hiddens = _d_buf_pre_atten_out + _batch_size * _batch_seq_len * _hidden_size;
        T *d_buf_layer = _d_buf_hiddens + _batch_size * _batch_seq_len * _hidden_size;

        // init layers
        int layer_num = py2int(config["num_hidden_layers"]);
        for (int layer_idx = 0; layer_idx < layer_num; layer_idx ++) {
            enc_layers.emplace_back(weights, config, d_buf_layer, layer_idx, _stream, _cublasHandle);
        }

        // set weights
        T *h_tok_embed_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("embeddings.word_embeddings.weight")]).request().ptr);
        T *h_pos_embed_weight_0 = static_cast<T*>(py::array_t<T>(weights[
            py::cast("embeddings.position_embeddings.weights.0")]).request().ptr);
        T *h_pos_embed_weight_1 = static_cast<T*>(py::array_t<T>(weights[
            py::cast("embeddings.position_embeddings.weights.1")]).request().ptr);
        T *h_enc_ln_weight = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layer_norm.weight")]).request().ptr);
        T *h_enc_ln_bias = static_cast<T*>(py::array_t<T>(weights[
            py::cast("encoder.layer_norm.bias")]).request().ptr);
        
        _d_tok_embed_weight.assign(h_tok_embed_weight, h_tok_embed_weight + _vocab_size * _hidden_size);
        _d_pos_embed_weight_0.assign(h_pos_embed_weight_0, h_pos_embed_weight_0 + _pos_shape_0 * _pos_embds_dim_0);
        _d_pos_embed_weight_1.assign(h_pos_embed_weight_1, h_pos_embed_weight_1 + _pos_shape_1 * _pos_embds_dim_1);
        _d_enc_ln_weight.assign(h_enc_ln_weight, h_enc_ln_weight + 2 * _hidden_size);
        _d_enc_ln_bias.assign(h_enc_ln_bias, h_enc_ln_bias + 2 * _hidden_size);
    }
    ~ReformerModel() {
        // cudaStreamDestroy(_stream);
        cublasDestroy(_cublasHandle);
        cudaFree(_buffer);
    }

    /**
     * @param input_ids [bs, seq_len]
     * @param random_rotations [num_heads, num_hashes, head_size, num_buckets/2]
     * @param pad_id [bs, seq_len, hidden_size]
     * @param output [bs, seq_len, 2 * hidden_size]
     */
    void infer_one_batch(
        const int *input_ids, const T* random_rotations, int pad_id,
        T *output)
    {
        // embedding
        // ids -> hiddens, padding_mask
        encoder_embedding_launcher(
            input_ids,
            thrust::raw_pointer_cast(_d_tok_embed_weight.data()),
            thrust::raw_pointer_cast(_d_pos_embed_weight_0.data()),
            thrust::raw_pointer_cast(_d_pos_embed_weight_1.data()),
            _pos_embds_dim_0, _pos_embds_dim_1, _pos_shape_0, _pos_shape_1,
            _batch_size, _batch_seq_len, _hidden_size, pad_id, 0,
            _d_buf_hiddens,
            _d_padding_mask);
        thrust::copy(
            thrust::device,
            _d_buf_hiddens,
            _d_buf_hiddens + _batch_size * _batch_seq_len * _hidden_size,
            _d_buf_pre_atten_out
        );

        // enc
        for (auto &layer : enc_layers) {
            layer.infer(
                _d_buf_pre_atten_out,
                _d_buf_hiddens,
                _d_padding_mask,
                random_rotations);
        }

        // cat
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(_batch_size * _batch_seq_len),
            [_d_buf_pre_atten_out=_d_buf_pre_atten_out,
            _d_buf_hiddens=_d_buf_hiddens, output,
            _hidden_size=_hidden_size] __device__ (int i) {
                thrust::copy(
                    thrust::device,
                    _d_buf_pre_atten_out + i * _hidden_size,
                    _d_buf_pre_atten_out + (i + 1) * _hidden_size,
                    output + 2 * i * _hidden_size);
                thrust::copy(
                    thrust::device,
                    _d_buf_hiddens + i * _hidden_size,
                    _d_buf_hiddens + (i + 1) * _hidden_size,
                    output + (2 * i + 1) * _hidden_size);
            }
        );
        
        // ln
        layer_norm_launcher<T>(
            output,
            thrust::raw_pointer_cast(_d_enc_ln_weight.data()),
            thrust::raw_pointer_cast(_d_enc_ln_bias.data()),
            _eps, 2 * _hidden_size,
            _batch_size * _batch_seq_len * 2 * _hidden_size);

    }
};
template class ReformerModel<FloatType::FP32>;


} // namespace FastReformer
