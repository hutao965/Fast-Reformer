import sys
import unittest
import numpy as np
import torch
import transformers
from transformers import (
    ReformerTokenizer,
    ReformerConfig,
    ReformerModel
)

sys.path.append("../bin/Release")
import testmodels
import testkernels


# TODO half
class Test(unittest.TestCase):
    def __init__(self):
        self._set_seed(123)
        self.config = ReformerConfig()
        self.tokenizer = ReformerTokenizer.from_pretrianed()
        hf_model_cpu = ReformerModel(self.config)
        self.hf_model = hf_model_cpu.cuda()
        self.test_models = testmodels.TestModels_fp32(
            hf_model_cpu.state_dict(), self.config.__dict__
        )
        self.test_kernels = testkernels.TestKernels_fp32(
            hf_model_cpu.state_dict(), self.config.__dict__
        )
        self._vocab_size = self.config.vocab_size
        self._batch_size = 32
        self._batch_seq_len = 256
        self._pad_id = self.config.pad_token_id

    def _set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _assert_allclose(fast, hf):
        np.testing.assert_allclose(
            fast.reshape(-1),
            hf.reshape(-1),
            rtol=1e-5,
            atol=0
        )
        # relative absolute erro

    def _random_input_ids(self):
        shape = (self._batch_size, self._batch_seq_len)
        input_ids = torch.randint(10, self._vocab_size - 10, shape)
        padding_mask = torch.full(shape, 1., dtype=torch.float)
        length = torch.randint(self._batch_seq_len//2, self._batch_seq_len, (self._batch_size,))
        for i in range(self._batch_size):
            input_ids[i][length[i]:] = 0
            padding_mask[i][length[i]:] = 0.
        return input_ids, padding_mask

    def _random_hiddens(self, shape):
        return torch.randn(shape)

    def test_enc_embedding(self):
        input_ids, padding_mask = self._random_input_ids()
        hf_result = self.hf_model.embeddings(input_ids.cuda()).cpu().numpy()
        fast_result, fast_padding_mask = self.test_kernels.test_encoder_embedding(
            input_ids, self._pad_id, self._batch_size, self._batch_seq_len
        )
        self._assert_allclose(fast_result, hf_result)
        self._assert_allclose(fast_padding_mask, padding_mask)

    def test_layer_norm(self):
        pass
    # def test_local_atten(self):
    #     pass
    # def test_lsh_atten(self):
    #     pass
    def test_softmax(self):
        query_key_dots = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self._batch_seq_len)
        )
        hf_result = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        hf_result = torch.exp(query_key_dots - hf_result)
        fast_result = self.test_kernels.test_softmax(
            query_key_dots, self._batch_size, self._batch_seq_len
        )
        self._assert_allclose(fast_result, hf_result)


    def test_chunk_ffn(self):
        
        pass

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
    #             atten_mask,
    #             self._batch_size,
    #             self._batch_seq_len,
    #             layerId
    #         )
    #         self._assert_allclose(fast_result, hf_result)


    # def test_model(self):
    #     # random input
    #     input_ids, atten_mask = self._random_input_ids()
    #     hf_result = self.hf_model(input_ids.cuda(), atten_mask.cuda()).cpu().numpy()
    #     fast_result = self.test_models.test_Reformer(input_ids.numpy(),
    #         self._pad_id, self._batch_size, self._batch_seq_len)
    #     self._assert_allclose(fast_result, hf_result)


if __name__ == '__main__':
    unittest.main()