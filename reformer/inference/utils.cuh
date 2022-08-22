#pragma once

#include <cassert>
#include <vector>
#include <memory>
#include <math.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace FastReformer {

// traits
// fp16

enum class FloatType{FP16, FP32};

template<FloatType T>
class TypeTraits;

template<>
class TypeTraits<FloatType::FP32> {
public:
    using DataType = float;
    static const cudaDataType_t cublasComputeType = CUDA_R_32F;
    static const cudaDataType_t cublasAType = CUDA_R_32F;
    static const cudaDataType_t cublasBType = CUDA_R_32F;
    static const cudaDataType_t cublasCType = CUDA_R_32F;
    static constexpr float mask_value = -1e9f;
    static constexpr float self_mask_value = -1e5f;
};

template<>
class TypeTraits<FloatType::FP16> {
public:
    using DataType = __half;
    static const cudaDataType_t cublasComputeType = CUDA_R_16F;
    static const cudaDataType_t cublasAType = CUDA_R_16F;
    static const cudaDataType_t cublasBType = CUDA_R_16F;
    static const cudaDataType_t cublasCType = CUDA_R_16F;
    static constexpr float mask_value = -1e4f;
    static constexpr float self_mask_value = -1e3f;
};


// allocate __constant__ or reigster directly
inline std::string py2str(const py::handle &handle) {
    return std::string(py::str(handle));
}

inline int py2int(const py::handle &handle) {
    return std::stoi(std::string(py::str(handle)));
}

inline float py2float(const py::handle &handle) {
    return std::stof(std::string(py::str(handle)));
}



} // namespace FastReformer