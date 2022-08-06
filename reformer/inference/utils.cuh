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
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace FastReformer {

// const __constant__ 
// traits
// fp16

enum class FloatType{FP16, FP32};

template<FloatType T>
class TypeTraits;

template<>
class TypeTraits<FloatType::FP32> {
public:
    using DataType = float;
    static cudaDataType_t const cublasComputeType = CUDA_R_32F;
    static cudaDataType_t const cublasAType = CUDA_R_32F;
    static cudaDataType_t const cublasBType = CUDA_R_32F;
    static cudaDataType_t const cublasCType = CUDA_R_32F;
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