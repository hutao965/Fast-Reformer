#include "../../reformer/inference/kernels.cuh"

using namespace FastReformer;

template<FloatType fp>
class TestKernels {
private:
    using T = typename TypeTraits<fp>::DataType;
    py::dict _weights;
    py::dict _config;
public:
    TestKernels(py::dict &weights, py::dict &config) :
        _weights(weights),
        _config(config) {}

    py::tuple test_encoder_embedding(py::array_t<int> &input_ids, int pad_id, int start_idx_pos_encodings) {
        int batch_size = input_ids.shape(0);
        int batch_seq_len = input_ids.shape(1);
        int hidden_size = py2int(_config["hidden_size"]);
        auto pos_embds_dims = py::list(_config["axial_pos_embds_dim"]);
        int pos_embds_dim_0 = py2int(pos_embds_dims[0]);
        int pos_embds_dim_1 = py2int(pos_embds_dims[1]);
        auto pos_shapes = py::list(_config["axial_pos_shape"]);
        int pos_shape_0 = py2int(pos_shapes[0]);
        int pos_shape_1 = py2int(pos_shapes[1]);
        int mask_size = batch_size * batch_seq_len;
        int output_size = batch_size * batch_seq_len * hidden_size;
        auto output = py::array_t<T>(output_size);
        auto padding_mask = py::array_t<int>(mask_size);

        int *h_input_ids = static_cast<int*>(input_ids.request().ptr);
        T *h_tok_embd_weights = static_cast<T*>(
            py::array_t<T>(_weights["embeddings.word_embeddings.weight"]).request().ptr);
        T *h_pos_embd_weight_0 = static_cast<T*>(
            py::array_t<T>(_weights["embeddings.position_embeddings.weights.0"]).request().ptr);
        T *h_pos_embd_weight_1 = static_cast<T*>(
            py::array_t<T>(_weights["embeddings.position_embeddings.weights.1"]).request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        int *h_pad_mask = static_cast<int*>(padding_mask.request().ptr);

        thrust::device_vector<int> d_input_ids(h_input_ids, h_input_ids + mask_size);
        thrust::device_vector<T> d_tok_embd_weights(h_tok_embd_weights, h_tok_embd_weights + py2int(_config["vocab_size"]) * hidden_size);
        thrust::device_vector<T> d_pos_embd_weight_0(h_pos_embd_weight_0, h_pos_embd_weight_0 + pos_shape_0 * pos_embds_dim_0);
        thrust::device_vector<T> d_pos_embd_weight_1(h_pos_embd_weight_1, h_pos_embd_weight_1 + pos_shape_1 * pos_embds_dim_1);
        thrust::device_vector<T> d_output(output_size);
        thrust::device_vector<int> d_pad_mask(mask_size);

        encoder_embedding_launcher<T>(
            thrust::raw_pointer_cast(d_input_ids.data()),
            thrust::raw_pointer_cast(d_tok_embd_weights.data()),
            thrust::raw_pointer_cast(d_pos_embd_weight_0.data()),
            thrust::raw_pointer_cast(d_pos_embd_weight_1.data()),
            pos_embds_dim_0, pos_embds_dim_1, pos_shape_0, pos_shape_1,
            batch_size, batch_seq_len, hidden_size, pad_id, start_idx_pos_encodings,
            thrust::raw_pointer_cast(d_output.data()),
            thrust::raw_pointer_cast(d_pad_mask.data()));

        thrust::copy(d_output.cbegin(), d_output.cend(), h_output);
        thrust::copy(d_pad_mask.cbegin(), d_pad_mask.cend(), h_pad_mask);

        return py::make_tuple(output, padding_mask);
    }
    
    py::array_t<T> test_encoder_layer_norm(py::array_t<T> &input, int norm_size) {
        int size = input.size();
        auto output = py::array_t<T>(size);
        T *h_input = static_cast<T*>(input.request().ptr);
        T *h_ln_weight = static_cast<T*>(
            py::array_t<T>(_weights["encoder.layer_norm.weight"]).request().ptr);
        T *h_ln_bias = static_cast<T*>(
            py::array_t<T>(_weights["encoder.layer_norm.bias"]).request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        T eps = static_cast<T>(py2float(_config["layer_norm_eps"]));

        thrust::device_vector<T> d_input(h_input, h_input + size);
        thrust::device_vector<T> d_ln_weight(h_ln_weight, h_ln_weight + norm_size);
        thrust::device_vector<T> d_ln_bias(h_ln_bias, h_ln_bias + norm_size);
        layer_norm_launcher<T>(
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_ln_weight.data()),
            thrust::raw_pointer_cast(d_ln_bias.data()),
            eps, norm_size, size);
        thrust::copy(d_input.cbegin(), d_input.cend(), h_output);
        return output;
    }

    py::array_t<T> test_bias_relu(py::array_t<T> &input, py::array_t<T> &bias) {
        int size = input.size();
        int hidden_size = bias.size();
        auto output = py::array_t<T>(size);
        T *h_input = static_cast<T*>(input.request().ptr);
        T *h_bias = static_cast<T*>(bias.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<T> d_input(h_input, h_input + size);
        thrust::device_vector<T> d_bias(h_bias, h_bias + hidden_size);
        bias_relu_launcher<T>(
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_bias.data()),
            hidden_size, size);
        thrust::copy(d_input.cbegin(), d_input.cend(), h_output);
        return output;
    }

    py::array_t<T> test_softmax(py::array_t<T> &input)
    {
        int size = input.size();
        int reduce_size = input.shape(2);
        auto output = py::array_t<T>(size);
        T *h_input = static_cast<T*>(input.request().ptr);
        T *h_output = static_cast<T*>(output.request().ptr);
        thrust::device_vector<T> d_input(h_input, h_input + size);
        softmax_launcher<T>(
            thrust::raw_pointer_cast(d_input.data()),
            reduce_size, size);
        thrust::copy(d_input.cbegin(), d_input.cend(), h_output);
        return output;
    }


    

};



PYBIND11_MODULE(testkernels, m) {
    m.attr("__name__") = "testkernels";
    py::class_<TestKernels<FloatType::FP32>>(m, "TestKernels_fp32")
        .def(py::init<py::dict &, py::dict &>())
        .def("test_encoder_embedding", &TestKernels<FloatType::FP32>::test_encoder_embedding,
             py::return_value_policy::reference_internal)
        .def("test_encoder_layer_norm", &TestKernels<FloatType::FP32>::test_encoder_layer_norm,
             py::return_value_policy::reference_internal)
        .def("test_softmax", &TestKernels<FloatType::FP32>::test_softmax,
             py::return_value_policy::reference_internal)
        .def("test_bias_relu", &TestKernels<FloatType::FP32>::test_bias_relu,
             py::return_value_policy::reference_internal);
}