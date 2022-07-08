add_library(kernels STATIC kernels.cu)
add_library(models STATIC models.cu)
# target_include_directories(kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(models INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(models pybind11::headers)
# target_link_libraries(
#     models PRIVATE
#     CUDA::cublas_static
#     CUDA::cublasLt_static
# )

# pybind11_add_module(
#     infer MODULE
#     infer.cu
# )
# target_link_libraries(infer PUBLIC models)