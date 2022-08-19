# Fast-Reformer
[Reformer](https://arxiv.org/abs/2001.04451) encoder (inference only) with cuda implementation  
for my cuda practice  
  
The test target is [huggingface reformer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py)  
  
```bash
pip install transformers
git clone --recursive https://github.com/hutao965/Fast-Reformer.git
cd Fast-Reformer
sh make.sh
sh test.sh
```

# Goal
- [x] reformer model
  - [x] axial embedding
  - [x] chunk ffn
  - [x] local atten
  - [x] lsh atten
    - [x] block radix sort
- [x] unit test
  - [x] pybind
- [ ] profiling
  - [x] fp32 speed up 38.63%
  - [ ] fp16

# Ref
- [lightseq](https://github.com/bytedance/lightseq)
- [block radix sort in cub](https://github.com/NVIDIA/cub/blob/main/cub/block/block_radix_sort.cuh)
