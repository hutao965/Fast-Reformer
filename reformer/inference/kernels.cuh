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
    T *output, T *pad_mask);

template<typename T>
void layer_norm_launcher(
    const T *input, const T *weight, const T *bias,
    T eps, int norm_size, int size, T *output);

template<typename T>
void bias_relu_launcher(
    const T *input, const T *bias,
    int hidden_size, int size, T *output);

template<typename T>
void softmax_launcher(
    const T *input, int reduce_size, int size,
    T *output);


// template<typename T>
// void split_launcher();


// template<typename T>
// void merge_launcher();


} // namespace FastReformer
