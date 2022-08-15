pybind11_add_module(
    testkernels MODULE
    test_kernels.cu
)
target_link_libraries(testkernels PUBLIC kernels)
pybind11_add_module(
    testmodels MODULE
    test_models.cu
)
target_link_libraries(
    testmodels PUBLIC
    models
    #cuda.lib
)