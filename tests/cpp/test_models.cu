#include "../../reformer/inference/models.cu"

using namespace FastReformer;

template<FloatType fp>
class TestModels {
private:
    using T = typename TypeTraits<fp>::DataType;
    ReformerModel<fp> _model;
    const int _hidden_size;
public:
    TestModels(py::dict &weights, py::dict &config) :
        _model(weights, config),
        _hidden_size(py2int(config["hidden_size"])) {}

    py::array_t<T> test_chunk_ffn(py::array_t<T> &atten_out) {
        int size = atten_out.size();
        auto output = py::array_t<T>(size);
        T *h_atten_out = static_cast<T*>(atten_out.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<T> d_atten_out(h_atten_out, h_atten_out + size);
        _model.enc_layers[0].chunk_ffn(thrust::raw_pointer_cast(d_atten_out.data()));
        thrust::copy(d_atten_out.begin(), d_atten_out.end(), h_output);
        return output;
    }

    py::array_t<T> test_local_atten(
        py::array_t<T> &hidden_states, py::array_t<int> &atten_mask)
    {
        int size = hidden_states.size();
        int mask_size = atten_mask.size();
        auto output = py::array_t<T>(size);
        T *h_hidden_states = static_cast<T*>(hidden_states.request().ptr);
        int *h_atten_mask = static_cast<int*>(atten_mask.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<T> d_hidden_states(h_hidden_states, h_hidden_states + size);
        thrust::device_vector<int> d_atten_mask(h_atten_mask, h_atten_mask + mask_size);
        _model.enc_layers[0].local_atten(
            thrust::raw_pointer_cast(d_hidden_states.data()),
            thrust::raw_pointer_cast(d_atten_mask.data()));
        thrust::copy(d_hidden_states.begin(), d_hidden_states.end(), h_output);
        return output;
    }

    py::array_t<T> test_lsh_atten(
        py::array_t<T> &hidden_states, py::array_t<int> &atten_mask,
        py::array_t<T> &random_rotations)
    {
        int size = hidden_states.size();
        auto output = py::array_t<T>(size);
        T *h_hidden_states = static_cast<T*>(hidden_states.request().ptr);
        int *h_atten_mask = static_cast<int*>(atten_mask.request().ptr);
        T *h_random_rotations = static_cast<T*>(random_rotations.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<T> d_hidden_states(h_hidden_states, h_hidden_states + size);
        thrust::device_vector<int> d_atten_mask(h_atten_mask, h_atten_mask + atten_mask.size());
        thrust::device_vector<T> d_random_rotations(h_random_rotations, h_random_rotations + random_rotations.size());
        _model.enc_layers[1].lsh_atten(
            thrust::raw_pointer_cast(d_hidden_states.data()),
            thrust::raw_pointer_cast(d_atten_mask.data()),
            thrust::raw_pointer_cast(d_random_rotations.data()));
        thrust::copy(d_hidden_states.begin(), d_hidden_states.end(), h_output);
        return output;
    }

    py::tuple test_Reformer_enc_layer(
        py::array_t<T> &pre_atten_output, py::array_t<T> &hiddens, py::array_t<int> &padding_mask,
        py::array_t<T> &random_rotations, int layer_id)
    {
        int size = hiddens.size();
        auto output = py::array_t<T>(size);
        auto atten_output = py::array_t<T>(size);
        T *h_pre_atten_output = static_cast<T*>(pre_atten_output.request().ptr);
        T *h_hiddens = static_cast<T*>(hiddens.request().ptr);
        int *h_padding_mask = static_cast<int*>(padding_mask.request().ptr);
        T *h_random_rotations = static_cast<T*>(random_rotations.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        T *h_atten_output = static_cast<T*>(atten_output.request().ptr);
        thrust::device_vector<T> d_pre_atten_output(h_pre_atten_output, h_pre_atten_output + size);
        thrust::device_vector<T> d_hiddens(h_hiddens, h_hiddens + size);
        thrust::device_vector<int> d_padding_mask(h_padding_mask, h_padding_mask + padding_mask.size());
        thrust::device_vector<T> d_random_rotations(h_random_rotations, h_random_rotations + random_rotations.size());
        _model.enc_layers[layer_id].infer(
            thrust::raw_pointer_cast(d_pre_atten_output.data()),
            thrust::raw_pointer_cast(d_hiddens.data()),
            thrust::raw_pointer_cast(d_padding_mask.data()),
            thrust::raw_pointer_cast(d_random_rotations.data()));
        thrust::copy(d_hiddens.cbegin(), d_hiddens.cend(), h_output);
        thrust::copy(d_pre_atten_output.cbegin(), d_pre_atten_output.cend(), h_atten_output);
        return py::make_tuple(output, atten_output);
    }

    py::array_t<T> test_Reformer(
        py::array_t<int> &input_ids, py::array_t<T> &random_rotations,
        int pad_id)
    {
        int output_size = input_ids.size() * _hidden_size * 2;
        auto output = py::array_t<T>(output_size);
        int *h_input_ids = static_cast<int*>(input_ids.request().ptr);
        T *h_random_rotations = static_cast<T*>(random_rotations.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<int> d_input_ids(h_input_ids, h_input_ids + input_ids.size());
        thrust::device_vector<T> d_random_rotations(h_random_rotations, h_random_rotations + random_rotations.size());
        thrust::device_vector<T> d_output(output_size);
        _model.infer_one_batch(
            thrust::raw_pointer_cast(d_input_ids.data()),
            thrust::raw_pointer_cast(d_random_rotations.data()),
            pad_id,
            thrust::raw_pointer_cast(d_output.data()));
        thrust::copy(d_output.cbegin(), d_output.cend(), h_output);
        return output;
    }
};


PYBIND11_MODULE(testmodels, m) {
    m.attr("__name__") = "testmodels";
    py::class_<TestModels<FloatType::FP32>>(m, "TestModels_fp32")
        .def(py::init<py::dict &, py::dict &>())
        .def("test_chunk_ffn", &TestModels<FloatType::FP32>::test_chunk_ffn,
             py::return_value_policy::reference_internal)
        .def("test_local_atten", &TestModels<FloatType::FP32>::test_local_atten,
             py::return_value_policy::reference_internal)
        .def("test_lsh_atten", &TestModels<FloatType::FP32>::test_lsh_atten,
             py::return_value_policy::reference_internal)
        .def("test_Reformer_enc_layer", &TestModels<FloatType::FP32>::test_Reformer_enc_layer,
             py::return_value_policy::reference_internal)
        .def("test_Reformer", &TestModels<FloatType::FP32>::test_Reformer,
             py::return_value_policy::reference_internal);
}
