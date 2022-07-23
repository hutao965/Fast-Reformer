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
import testmodels
import testkernels


# TODO half
class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self._set_seed(123)
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

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _assert_allclose(self, fast, hf):
        # print(fast.reshape(-1)[:100])
        # print(hf.reshape(-1)[:100])
        np.testing.assert_allclose(
            fast.reshape(-1),
            hf.reshape(-1),
            rtol=1e-5,
            atol=0
        )

    def _random_input_ids(self):
        shape = (self._batch_size, self._batch_seq_len)
        input_ids = torch.randint(10, self._vocab_size - 10, shape)
        padding_mask = torch.full(shape, 1., dtype=torch.float)
        length = torch.randint(self._batch_seq_len//2, self._batch_seq_len, (self._batch_size,))
        for i in range(self._batch_size):
            input_ids[i][length[i]:] = self._pad_id
            padding_mask[i][length[i]:] = 0.
        return input_ids, padding_mask

    def _random_hiddens(self, shape):
        return torch.randn(shape)

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

    # def test_local_atten(self):
    #     pass
    # def test_lsh_atten(self):
    #     pass

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
        hf_result = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        hf_result = torch.exp(query_key_dots - hf_result)
        fast_result = self.test_kernels.test_softmax(query_key_dots)
        self._assert_allclose(fast_result, hf_result.numpy())

    # def test_chunk_ffn(self):
    #     atten_output = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, self.config.hidden_size)
    #     )
    #     hf_result = self.hf_model.encoder.layers[0].feed_forward(
    #         atten_output.cuda()
    #     ).detach().cpu().numpy()
    #     fast_result = self.test_kernels.test_chunk_ffn(
    #         atten_output.numpy()
    #     )
    #     self._assert_allclose(fast_result, hf_result)

    # def test_layer(self):
    #     # TODO num hashes, buckets
    #     prev_hiddens, _ = self.random_input_hiddens()
    #     hiddens, atten_mask = self._random_hiddens()
    #     for layerId in [0, 1]:
    #         hf_result = self.hf_model.encoder.layers[layerId](
    #             prev_hiddens.cuda(),
    #             hiddens.cuda(),
    #             atten_mask.cuda()
    #         ).cpu().numpy()
    #         fast_result = self.test_models.test_Reformer_layer(
    #             prev_hiddens,
    #             hiddens,
    #             atten_mask
    #         )
    #         self._assert_allclose(fast_result, hf_result)


    # def test_model(self):
    #     # random input
    #     input_ids, atten_mask = self._random_input_ids()
    #     hf_result = self.hf_model(input_ids.cuda(), atten_mask.cuda()).cpu().numpy()
    #     fast_result = self.test_models.test_Reformer(
    #         input_ids.numpy(),
    #         self._pad_id)
    #     self._assert_allclose(fast_result, hf_result)


if __name__ == '__main__':
    unittest.main()