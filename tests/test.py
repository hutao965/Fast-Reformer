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

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _assert_allclose(self, fast, hf, rtol=1e-5, atol=0):
        # print(hf.shape)
        # a = hf.size
        # diff = (fast.reshape(-1) - hf.reshape(-1))
        # print(diff[a//4*3-50:a//4*3+50])
        # print(diff[:100])
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

    # def test_encoder_embedding(self):
    #     start_idx_pos_encodings = \
    #         self.config.axial_pos_shape[0] * self.config.axial_pos_shape[1] // 2 + \
    #         self.config.axial_pos_shape[1] // 2 - 1
    #     # start_idx_pos_encodings = 0
    #     input_ids, padding_mask = self._random_input_ids()
    #     hf_result = self.hf_model.embeddings(
    #         input_ids.cuda(),
    #         start_idx_pos_encodings=start_idx_pos_encodings
    #     ).detach().cpu().numpy()

    #     fast_result, fast_padding_mask = self.test_kernels.test_encoder_embedding(
    #         input_ids.numpy(),
    #         self._pad_id,
    #         start_idx_pos_encodings
    #     )
    #     self._assert_allclose(fast_result, hf_result)
    #     self._assert_allclose(fast_padding_mask, padding_mask.numpy())

    # def test_encoder_layer_norm(self):
    #     hiddens = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, 2 * self.config.hidden_size)
    #     )
    #     hf_result = self.hf_model.encoder.layer_norm(
    #         hiddens.cuda()
    #     ).detach().cpu().numpy()
    #     fast_result = self.test_kernels.test_encoder_layer_norm(
    #         hiddens,
    #         2 * self.config.hidden_size
    #     )
    #     self._assert_allclose(fast_result, hf_result)

    # def test_bias_relu(self):
    #     hiddens = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, self.config.hidden_size)
    #     )
    #     bias = self._random_hiddens(
    #         (self.config.hidden_size,)
    #     )
    #     hf_result = torch.nn.functional.relu(hiddens + bias).numpy()
    #     fast_result = self.test_kernels.test_bias_relu(
    #         hiddens, bias
    #     )
    #     self._assert_allclose(fast_result, hf_result)

    # def test_softmax(self):
    #     query_key_dots = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, self._batch_seq_len)
    #     )
    #     # logsumexp seems not stable, it will be
    #     # torch.logsumexp(tensor([-1e9, -1e9, -1e9, -1e9]), dim=-1, keepdim=True) ->
    #     # tensor([-1e9])
    #     hf_result = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
    #     hf_result = torch.exp(query_key_dots - hf_result)
    #     fast_result = self.test_kernels.test_softmax(query_key_dots)
    #     self._assert_allclose(fast_result, hf_result.numpy())

    # def test_chunk_ffn(self):
    #     atten_output = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, self.config.hidden_size)
    #     )
    #     hf_result = self.hf_model.encoder.layers[0].feed_forward(
    #         atten_output.cuda()
    #     ).detach().cpu().numpy()
    #     fast_result = self.test_models.test_chunk_ffn(
    #         atten_output.numpy()
    #     )
    #     self._assert_allclose(fast_result, hf_result, atol=5e-3)

    # def test_local_atten(self):
    #     _, padding_mask = self._random_input_ids()
    #     hidden_states = self._random_hiddens(
    #         (self._batch_size, self._batch_seq_len, self.config.hidden_size)
    #     )
    #     hf_result = self.hf_model.encoder.layers[0].attention(
    #         hidden_states.cuda(),
    #         attention_mask=padding_mask.cuda()
    #     ).hidden_states.detach().cpu().numpy()

    #     hf_result = hf_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
    #                 padding_mask.unsqueeze(-1).numpy()

    #     fast_result = self.test_models.test_local_atten(
    #         hidden_states.numpy(),
    #         padding_mask.numpy()
    #     )
    #     fast_result = fast_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
    #                   padding_mask.unsqueeze(-1).numpy()
    #     self._assert_allclose(fast_result, hf_result, atol=1e-4)

    def test_lsh_atten(self):
        _, padding_mask = self._random_input_ids()
        hidden_states = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        # hf_result = self.hf_model.encoder.layers[1].attention(
        #     hidden_states.cuda(),
        #     attention_mask=padding_mask.cuda()
        # ).hidden_states.detach().cpu().numpy()

        atten = self.hf_model.encoder.layers[1].attention.self_attention
        x = self.hf_model.encoder.layers[1].attention.layer_norm(hidden_states.cuda())
        query_key_vectors = atten.query_key(x)
        value_vectors = atten.value(x)
        query_key_vectors = atten._split_hidden_size_dim(
            query_key_vectors, atten.num_attention_heads, atten.attention_head_size
        )
        value_vectors = atten._split_hidden_size_dim(
            value_vectors, atten.num_attention_heads, atten.attention_head_size
        )
        rotation_size = atten.num_buckets
        num_buckets = atten.num_buckets
        vectors = query_key_vectors.detach()
        torch.manual_seed(atten.hash_seed)
        rotations_shape = (atten.num_attention_heads, vectors.shape[-1], atten.num_hashes, rotation_size // 2)
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        buckets = torch.argmax(rotated_vectors, dim=-1)
        buckets_mask = padding_mask.cuda().to(torch.uint8)[:, None, None, :].expand(buckets.shape)
        buckets = torch.where(
            buckets_mask, buckets, torch.tensor(num_buckets, dtype=torch.long, device=buckets.device)
        )
        offsets = torch.arange(atten.num_hashes, device=vectors.device)
        offsets = (offsets * (num_buckets + 1)).view((1, 1, -1, 1))
        offsets = offsets.expand((self._batch_size, atten.num_attention_heads) + offsets.shape[-2:])
        buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        sorted_bucket_idx, undo_sorted_bucket_idx = atten._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
            self._batch_seq_len, buckets, atten.num_hashes
        )
        sorted_bucket_idx_per_hash = sorted_bucket_idx % self._batch_seq_len
        query_key_vectors = atten._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, atten.num_hashes)
        value_vectors = atten._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, atten.num_hashes)
        query_key_vectors = atten._split_seq_length_dim_to(
            query_key_vectors,
            -1,
            atten.chunk_length,
            atten.num_attention_heads,
            atten.attention_head_size,
        )
        value_vectors = atten._split_seq_length_dim_to(
            value_vectors,
            -1,
            atten.chunk_length,
            atten.num_attention_heads,
            atten.attention_head_size,
        )
        key_vectors = atten._len_and_dim_norm(query_key_vectors)
        # atten
        key_vectors = atten._look_adjacent(key_vectors, atten.num_chunks_before, atten.num_chunks_after)
        value_vectors = atten._look_adjacent(value_vectors, atten.num_chunks_before, atten.num_chunks_after)
        query_key_dots = torch.matmul(query_key_vectors, key_vectors.transpose(-1, -2))

        hf_result = query_key_dots.reshape(-1).detach().cpu().numpy()
        # hf_result = hf_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
        #             padding_mask.unsqueeze(-1).numpy()
        
        torch.manual_seed(self.config.hash_seed)
        fast_result = self.test_models.test_lsh_atten(
            hidden_states.numpy(),
            padding_mask.numpy(),
            np.swapaxes(random_rotations.detach().cpu().numpy(), 1, 2).copy()
        )
        fast_result = fast_result[:hf_result.size]
        # fast_result = fast_result.reshape(self._batch_size, self._batch_seq_len, self.config.hidden_size) * \
        #               padding_mask.unsqueeze(-1).numpy()
        self._assert_allclose(fast_result, hf_result, atol=1e-3)




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