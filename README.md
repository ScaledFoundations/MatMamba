# MatMamba
![MatMamba](assets/blog1.jpg)

## About
MatMamba is a general sequence processing architecture based on [Mamba2](https://github.com/state-spaces/mamba). It introduces a nested [Matryoshka](https://arxiv.org/abs/2205.13147) structure in a Mamba2 block. We jointly train a few chosen granularities to get a single model from which we can flexibly extract a large number of nested submodels for adaptive inference based on the available deployment compute.

## Setup
To install the `matmamba` package and set up a fresh conda environment with all necessary dependencies, run the following script:

```bash
bash scripts/setup_env.sh
```

## Usage

Like a Transformer and Mamba2, a MatMamba2 block takes in a tensor of shape `(batch_size, seq_len, d_model)` and returns a tensor of the same shape. Based on the available compute, we can use a specific number of dimensions (and heads) internally.

```python
from matmamba import MatMamba2

matmamba_block = MatMamba2(
    d_model=512,
    d_state=128,
).cuda()
b, l, d = 8, 1024, 512
x = torch.randn((b, l, d)).cuda()

# Without any optional args/config, the block is a regular Mamba2 block
y1 = matmamba_block(x)
assert y1.shape == (b, l, d)

# If we want a number of dims as a fraction of `d_model`, we can use the `mrl_level` 
# An `mrl_level` of 2 means that `d_model/2` dims will be used
y2 = matmamba_block(x, mrl_level=2)

# `y2` is also (b, l, d), but only half the dims are used internally
assert y2.shape == (b, l, d)

# We can also manually specify the number of dims for each layer using `mixnmatch_dims` 
# For example, if we want to use exactly 64 dims:
matmamba_block.mixnmatch = True
matmamba_block.mixnmatch_dims = 64

y3 = matmamba_block(x)
assert y3.shape == (b, l, d)

# Set mixnmatch to False to revert to the default behavior
matmamba_block.mixnmatch = False
matmamba_block.mixnmatch_dims = matmamba_block.d_model

```

## Pretrained Models

You can find all pretrained models (MatMamba-Vision and MatMamba-LM) from the paper on Hugging Face in the [MatMamba collection](https://huggingface.co/collections/scaledfoundations/matmamba-670701480fa415dc2de60453).

| Model Name       | Training Dataset | d_model | Training Granularities | Link to Weights                                                                 |
|------------------|------------------|---------|------------------------|---------------------------------------------------------------------------------|
| MatMamba-Vision-35M  | ImageNet         | 512     | 512, 256, 128, 64                | [weights](https://huggingface.co/scaledfoundations/MatMamba-Vision-35M-ImageNet/tree/main) |
| MatMamba-Vision-135M  | ImageNet         | 1024     | 1024, 512, 256, 128               | [weights](https://huggingface.co/scaledfoundations/MatMamba-Vision-670M-ImageNet/tree/main) |
| MatMamba-LM-130M  | FineWeb         | 768     | 768, 384, 192, 96               | [weights](https://huggingface.co/scaledfoundations/MatMamba-LM-130M-FineWeb/tree/main) |
| MatMamba-LM-370M  | FineWeb         | 1024     | 1024, 512, 256, 128               | [weights](https://huggingface.co/scaledfoundations/MatMamba-LM-370M-FineWeb/tree/main) |
| MatMamba-LM-790M | FineWeb         | 1536     | 1536, 768, 384, 192               | [weights](https://huggingface.co/scaledfoundations/MatMamba-LM-790M-FineWeb/tree/main) |
| MatMamba-LM-1.4B | FineWeb         | 2048     | 2048, 1024, 512, 256               | [weights](https://huggingface.co/scaledfoundations/MatMamba-LM-1.4B-FineWeb/tree/main) |

### MatMamba-Vision
![MatMamba-Vision](assets/blog2.jpg)

### MatMamba-LM

## Citation

If you use this code, or otherwise find our work valuable, please cite:

```
@article{author2023matmamba,
    title={MatMamba: A Matryoshka State Space Model},
    author={Shukla, Abhinav and Vemprala, Sai, and Kusupati, Aditya, and Kapoor, Ashish},
    journal={arXiv preprint arXiv:2410.06718},
    year={2024}
}
```