#include "../../reformer/inference/kernels.cuh"

using namespace FastReformer;

template<FloatType fp>
class TestKernels {
private:
    using T = typename TypeTraits<fp>::DataType;
    py::dict _weights;
    py::dict _config;
    const int _hidden_size;
public:
    TestKernels(py::dict &weights, py::dict &config) :
        _weights(weights),
        _config(config),
        _hidden_size(py2int(config["hidden_size"])) {}

    py::tuple test_encoder_embedding(py::array_t<int> &input_ids, int pad_id,
                                     int batch_size, int batch_seq_len) {
        int mask_size = batch_size * batch_seq_len;
        int output_size = batch_size * batch_seq_len * _hidden_size;
        auto output = py::array_t<T>(output_size);
        auto padding_mask = py::array_t<int>(mask_size);

        // from py weights -> 
        py::array_t<T>(_weights["embeddings.word_embeddings.weight"]);
        encoder_embedding_launcher<T>();

        return py::make_tuple(output, padding_mask);
    }
    
    py::array_t<T> test_layer_norm(py::array_t<T> &hiddens) {
        return nullptr;
    }
    py::array_t<T> test_rev_layer_norm() {
        return nullptr;
    }
    py::array_t<T> test_softmax(py::array_t<T> &query_key_dots, int batch_size,
                                int batch_seq_len) {

        
        return nullptr;
    }
    py::array_t<T> test_bias_relu() {
        return nullptr;
    }

    py::array_t<T> test_chunk_ffn() {
        return nullptr;
    }

};



PYBIND11_MODULE(testkernels, m) {
    m.attr("__name__") = "testkernels";
    py::class_<TestKernels<FloatType::FP32>>(m, "TestKernels_fp32")
        .def(py::init<py::dict &, py::dict &>())
        .def("test_encoder_embedding", &TestKernels<FloatType::FP32>::test_encoder_embedding,
             py::return_value_policy::reference_internal);
}