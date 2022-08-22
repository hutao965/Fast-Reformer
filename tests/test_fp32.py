import sys
import unittest
import numpy as np
import torch
import transformers
from transformers import (
    ReformerConfig,
    ReformerModel
)

sys.path.append("../bin/Release")
sys.path.append("../bin")
import testmodels
import testkernels


# TODO half
class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)

    def setUp(self):
        self._set_seed(567)
        self.config = ReformerConfig.from_pretrained("config.json")
        hf_model_cpu = ReformerModel(self.config)
        test_weight_dict = {k:v.numpy() for k,v in hf_model_cpu.state_dict().items()}
        self.hf_model = hf_model_cpu.cuda()
        self.hf_model.eval()
        self.test_models = testmodels.TestModels_fp32(
            test_weight_dict, self.config.__dict__)
        self.test_kernels = testkernels.TestKernels_fp32(
            test_weight_dict, self.config.__dict__)
        self._vocab_size = self.config.vocab_size
        self._batch_size = self.config.batch_size
        self._batch_seq_len = self.config.batch_seq_len
        self._pad_id = self.config.pad_token_id

    def tearDown(self):
        del self.config
        del self.hf_model
        del self.test_models
        del self.test_kernels

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _assert_allclose(self, fast, hf, rtol=1e-5, atol=0):
        np.testing.assert_allclose(
            fast.reshape(-1),
            hf.reshape(-1),
            rtol=rtol,
            atol=atol
        )

    def _random_input_ids(self):
        shape = (self._batch_size, self._batch_seq_len)
        input_ids = torch.randint(10, self._vocab_size - 10, shape)
        padding_mask = torch.full(shape, 1, dtype=torch.int)
        length = torch.randint(self._batch_seq_len//2, self._batch_seq_len, (self._batch_size,))
        for i in range(self._batch_size):
            input_ids[i][length[i]:] = self._pad_id
            padding_mask[i][length[i]:] = 0
        return input_ids, padding_mask

    def _random_hiddens(self, shape):
        return torch.randn(shape)

    def _random_rotation(self):
        # the same random rotations as huggingface reformer
        torch.manual_seed(self.config.hash_seed)
        random_rotations = torch.randn(
            (
                self.config.num_attention_heads,
                self.config.attention_head_size,
                self.config.num_hashes,
                self.config.num_buckets // 2
            ),
            device="cuda"
        )
        return np.swapaxes(random_rotations.detach().cpu().numpy(), 1, 2).copy()

    def test_encoder_embedding(self):
        start_idx_pos_encodings = \
            self.config.axial_pos_shape[0] * self.config.axial_pos_shape[1] // 2 + \
            self.config.axial_pos_shape[1] // 2 - 1
        # start_idx_pos_encodings = 0
        input_ids, padding_mask = self._random_input_ids()
        hf_result = self.hf_model.embeddings(
            input_ids.cuda(),
            start_idx_pos_encodings=start_idx_pos_encodings
        ).detach().cpu().numpy()

        fast_result, fast_padding_mask = self.test_kernels.test_encoder_embedding(
            input_ids.numpy(),
            self._pad_id,
            start_idx_pos_encodings
        )
        self._assert_allclose(fast_result, hf_result)
        self._assert_allclose(fast_padding_mask, padding_mask.numpy())

    def test_encoder_layer_norm(self):
        hiddens = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, 2 * self.config.hidden_size)
        )
        hf_result = self.hf_model.encoder.layer_norm(
            hiddens.cuda()
        ).detach().cpu().numpy()
        fast_result = self.test_kernels.test_encoder_layer_norm(
            hiddens,
            2 * self.config.hidden_size
        )
        self._assert_allclose(fast_result, hf_result)

    def test_bias_relu(self):
        hiddens = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        bias = self._random_hiddens(
            (self.config.hidden_size,)
        )
        hf_result = torch.nn.functional.relu(hiddens + bias).numpy()
        fast_result = self.test_kernels.test_bias_relu(
            hiddens, bias
        )
        self._assert_allclose(fast_result, hf_result)

    def test_softmax(self):
        query_key_dots = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self._batch_seq_len)
        )
        # logsumexp seems not stable, it will be
        # torch.logsumexp(tensor([-1e9, -1e9, -1e9, -1e9]), dim=-1, keepdim=True) ->
        # tensor([-1e9])
        hf_result = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        hf_result = torch.exp(query_key_dots - hf_result)
        fast_result = self.test_kernels.test_softmax(query_key_dots)
        self._assert_allclose(fast_result, hf_result.numpy())

    def test_chunk_ffn(self):
        atten_output = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        hf_result = self.hf_model.encoder.layers[0].feed_forward(
            atten_output.cuda()
        ).detach().cpu().numpy()
        fast_result = self.test_models.test_chunk_ffn(
            atten_output.numpy()
        )
        self._assert_allclose(fast_result, hf_result, atol=1e-2)

    def test_local_atten(self):
        _, padding_mask = self._random_input_ids()
        hidden_states = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        hf_result = self.hf_model.encoder.layers[0].attention(
            hidden_states.cuda(),
            attention_mask=padding_mask.cuda()
        ).hidden_states.detach().cpu().numpy()

        hf_result = hf_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
                    padding_mask.unsqueeze(-1).numpy()

        fast_result = self.test_models.test_local_atten(
            hidden_states.numpy(),
            padding_mask.numpy()
        )
        fast_result = fast_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
                      padding_mask.unsqueeze(-1).numpy()
        self._assert_allclose(fast_result, hf_result, atol=1e-4)

    def test_lsh_atten(self):
        _, padding_mask = self._random_input_ids()
        hidden_states = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        hf_result = self.hf_model.encoder.layers[1].attention(
            hidden_states.cuda(),
            attention_mask=padding_mask.cuda()
        ).hidden_states.detach().cpu().numpy()

        fast_result = self.test_models.test_lsh_atten(
            hidden_states.numpy(),
            padding_mask.numpy(),
            self._random_rotation()
        )
        fast_result = fast_result[:hf_result.size]
        self._assert_allclose(fast_result, hf_result, atol=1e-2)

    def test_layer(self):
        _, padding_mask = self._random_input_ids()
        prev_atten_out = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        hiddens = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        for layer_id in [0, 1]:
            hf_result = self.hf_model.encoder.layers[layer_id](
                prev_atten_out.cuda(),
                hiddens.cuda(),
                padding_mask.cuda()
            )
            fast_result = self.test_models.test_Reformer_enc_layer(
                prev_atten_out.numpy(),
                hiddens.numpy(),
                padding_mask.numpy(),
                self._random_rotation(),
                layer_id
            )
            self._assert_allclose(
                fast_result[0],
                hf_result.hidden_states.detach().cpu().numpy(),
                atol=1e-2
            )
            self._assert_allclose(
                fast_result[1],
                hf_result.attn_output.detach().cpu().numpy(),
                atol=2e-2
            )

    def test_model(self):
        input_ids, padding_mask = self._random_input_ids()
        hf_result = self.hf_model(
            input_ids.cuda(),
            padding_mask.cuda()
        )[0].detach().cpu().numpy()
        fast_result = self.test_models.test_Reformer(
            input_ids.numpy(),
            self._random_rotation(),
            self._pad_id
        )
        self._assert_allclose(
            fast_result,
            hf_result,
            atol=5e-2
        )


if __name__ == '__main__':
    unittest.main()