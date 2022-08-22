#pragma once
#include "utils.cuh"

namespace FastReformer {

template<typename T>
void encoder_embedding_launcher(
    const int *input_ids, const T *tok_embd_weights,
    const T *pos_embd_weight_0, const T *pos_embd_weight_1,
    int pos_embds_dim_0, int pos_embds_dim_1, int pos_shape_0, int pos_shape_1,
    int batch_size, int batch_seq_len, int hidden_size,
    int pad_id, int start_idx_pos_encodings,
    T *output, int *pad_mask);

template<typename T>
void layer_norm_launcher(
    T *input, const T *weight, const T *bias,
    T eps, int norm_size, int size);

template<typename T>
void bias_relu_launcher(
    T *input, const T *bias,
    int hidden_size, int size);

template<typename T>
void add_bias_launcher(
    T *input, const T *bias,
    int hidden_size, int size);

template<typename T>
void softmax_launcher(
    T *input, int reduce_size, int size);

template<typename T>
void atten_split_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output);

template<typename T>
void atten_merge_transpose_launcher(
    const T *input, int batch_size, int seq_len,
    int chunk_len, int num_heads, int head_size,
    T *output);

template<typename T>
void look_adjacent_launcher(
    const T *input, int batch_size, int num_heads, int n_chunks,
    int chunk_len, int head_size, int before, int after,
    T *output);

template<typename T>
void local_atten_enc_mask_launcher(
    T *qk_dots, const int *mask, T mask_value, int batch_size, int num_heads,
    int n_chunks, int chunk_len, int N);

template<typename T>
void repeat_launcher(
    const T *in, int dim0, int dim1, int repeat_num,
    T *out);

template<typename T>
void lsh_bucket_argmax_mask_offset_launcher(
    const T *in, const int *atten_mask,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int num_bucket,
    int *out);

template<typename T>
void block_unsigned_radix_sort_launcher(T *in, int *idx, int grid, int N);

void lsh_scatter_undo_idx_launcher(
    int *sorted_idx, int *undo_sorted_idx,
    int batch_size, int num_heads, int num_hashes, int seq_len);

template<typename T>
void lsh_gather_by_expansion_launcher(
    const T *in, const int *idx, int batch_size,
    int num_heads, int num_hashes, int seq_len, int head_size,
    T *out);

template<typename T>
void lsh_len_norm_launcher(
    const T *in, int norm_size, int size, T norm_scalar, T *out);

template<typename T>
void lsh_enc_mask_launcher(
    T *qk_dots, const int *q_idx, const int *k_idx, const int *atten_mask,
    T mask_value, T self_mask_value, int batch_size, int num_heads, int num_hashes,
    int seq_len, int chunk_len, int N);

template<typename T>
void softmax_with_logits_launcher(
    T *input, T *logits, int reduce_size, int size);

template<typename T>
void lsh_undo_sort_launcher(
    const int *undo_sort_idx, const T *vec, const T *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    T *rev_vec, T *rev_logits);

template<typename T>
void sum_up_hashes_launcher(
    const T *in, const T *logits,
    int batch_size, int num_heads, int num_hashes,
    int seq_len, int head_size,
    T *out);

template<typename T>
void add_launcher(T *first, T *second, int size);

} // namespace FastReformer
