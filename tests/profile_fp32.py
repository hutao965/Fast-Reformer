import sys
import numpy as np
import torch
import transformers
from transformers import (
    ReformerConfig,
    ReformerModel
)
import torch.cuda.profiler as profiler

sys.path.append("../bin/Release")
sys.path.append("../bin")
import testmodels

class Profile():
    def __init__(self):
        self._set_seed(567)
        self.config = ReformerConfig.from_pretrained("config.json")
        hf_model_cpu = ReformerModel(self.config)
        test_weight_dict = {k:v.numpy() for k,v in hf_model_cpu.state_dict().items()}
        self.hf_model = hf_model_cpu.cuda()
        self.hf_model.eval()
        self.test_models = testmodels.TestModels_fp32(
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


    def profile_model(self):
        input_ids, padding_mask = self._random_input_ids()
        random_rotations = self._random_rotation()
        with profiler.profile():
            hf_result = self.hf_model(
                input_ids.cuda(),
                padding_mask.cuda()
            )
        with profiler.profile():
            fast_result = self.test_models.test_Reformer(
                input_ids.numpy(),
                random_rotations,
                self._pad_id
            )

    def profile_local_atten(self):
        _, padding_mask = self._random_input_ids()
        hidden_states = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        with profiler.profile():
            hf_result = self.hf_model.encoder.layers[0].attention(
                hidden_states.cuda(),
                attention_mask=padding_mask.cuda()
            )
        with profiler.profile():
            fast_result = self.test_models.test_local_atten(
                hidden_states.numpy(),
                padding_mask.numpy()
            )

    def profile_lsh_atten(self):
        _, padding_mask = self._random_input_ids()
        hidden_states = self._random_hiddens(
            (self._batch_size, self._batch_seq_len, self.config.hidden_size)
        )
        random_rotation = self._random_rotation()
        
        with profiler.profile():
            hf_result = self.hf_model.encoder.layers[1].attention(
                hidden_states.cuda(),
                attention_mask=padding_mask.cuda()
            )
        with profiler.profile():
            fast_result = self.test_models.test_lsh_atten(
                hidden_states.numpy(),
                padding_mask.numpy(),
                random_rotation
            )

if __name__ == "__main__":
    p = Profile()
    p.profile_model()
    