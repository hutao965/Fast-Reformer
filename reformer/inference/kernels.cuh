#pragma once
#include "utils.cuh"

namespace FastReformer {

template<typename T>
void encoder_embedding_launcher();

template<typename T>
void layer_norm_launcher();

template<typename T>
void rev_layer_norm_launcher();

template<typename T>
void split_launcher();

template<typename T>
void softmax_launcher();

template<typename T>
void merge_launcher();

template<typename T>
void bias_relu_launcher();

} // namespace FastReformer
