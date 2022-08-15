add_library(kernels STATIC kernels.cu)
add_library(models STATIC models.cu)
target_link_libraries(
    models PRIVATE
    kernels
    ${CUDA_cublas_LIBRARY}
)
