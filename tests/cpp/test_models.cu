#include "../../reformer/inference/models.cuh"

using namespace FastReformer;

template<FloatType fp>
class TestModels {
private:
    using T = typename TypeTraits<fp>::DataType;
    std::shared_ptr<ReformerModel<fp>> _model;
    const int _hidden_size;
public:
    TestModels(py::dict &weights, py::dict &config) :
        _model(std::make_shared<ReformerModel<fp>>(weights, config)),
        _hidden_size(py2int(config["hidden_size"])) {}

    py::array_t<T> test_Reformer(py::array_t<int> &input_ids, int pad_id,
                                 int batch_size, int batch_seq_len) {
        int input_size = batch_size * batch_seq_len;
        int output_size = batch_size * batch_seq_len * _hidden_size;
        auto output_hiddens = py::array_t<T>(output_size);

        int *h_p_input_ids = static_cast<int*>(input_ids.request().ptr);
        T *h_p_output = static_cast<T*>(output_hiddens.request().ptr);
        int *d_p_input_ids;
        T *d_p_output;
        cudaMalloc(&d_p_input_ids, input_size * sizeof(int));
        cudaMalloc(&d_p_output, output_size * sizeof(T));
        cudaMemcpy(d_p_input_ids, h_p_input_ids, input_size * sizeof(int), cudaMemcpyHostToDevice);
        _model->infer_one_batch(d_p_input_ids, d_p_output,
                                pad_id, batch_size, batch_seq_len);
        cudaMemcpy(h_p_output, d_p_output, output_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_p_input_ids);
        cudaFree(d_p_output);
        return output_hiddens;
    }

    py::array_t<T> test_Reformer_layer(py::array_t<T> &prev_output, py::array_t<T> &hiddens,
                                       py::array_t<T> &padding_mask, int batch_size, int batch_seq_len,
                                       int layerId) {
        auto layer = _model->encoder->layers[layerId];
        int mask_size = batch_size * batch_seq_len;
        int output_size = batch_size * batch_seq_len * _hidden_size;
        auto output_hiddens = py::array_t<T>(output_size);

        T *h_p_padding_mask = static_cast<T*>(padding_mask.request().ptr);
        T *h_p_prev_output = static_cast<T*>(prev_output.request().ptr);
        T *h_p_hiddens = static_cast<T*>(hiddens.request().ptr);
        T *h_p_output = static_cast<T*>(output_hiddens.request().ptr);
        T *d_p_padding_mask;
        T *d_p_prev_output, *d_p_hiddens, *d_p_output;
        cudaMalloc(&d_p_padding_mask, mask_size * sizeof(T));
        cudaMalloc(&d_p_prev_output, output_size * sizeof(T));
        cudaMalloc(&d_p_hiddens, output_size * sizeof(T));
        cudaMalloc(&d_p_output, output_size * sizeof(T));
        cudaMemcpy(d_p_padding_mask, h_p_padding_mask, mask_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p_prev_output, h_p_prev_output, output_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p_hiddens, h_p_hiddens, output_size * sizeof(T), cudaMemcpyHostToDevice);
        layer->infer(d_p_prev_output, d_p_hiddens, d_p_padding_mask, d_p_output,
                        batch_size, batch_seq_len);
        cudaMemcpy(h_p_output, d_p_output, output_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_p_padding_mask);
        cudaFree(d_p_prev_output);
        cudaFree(d_p_hiddens);
        cudaFree(d_p_output);
        return output_hiddens;
    }
};


PYBIND11_MODULE(testmodels, m) {
    m.attr("__name__") = "testmodels";
    py::class_<TestModels<FloatType::FP32>>(m, "TestModels_fp32")
        .def(py::init<py::dict &, py::dict &>())
        .def("test_Reformer", &TestModels<FloatType::FP32>::test_Reformer,
             py::return_value_policy::reference_internal)
        .def("test_Reformer_layer", &TestModels<FloatType::FP32>::test_Reformer_layer,
             py::return_value_policy::reference_internal);
}
