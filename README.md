# MatMamba
![MatMamba](assets/blog1.jpg)

## About
MatMamba is a general sequence processing architecture based on Mamba2. It introduces a nested Matryoshka structure in a Mamba2 block. We jointly train a few chosen granularities to get a single model from which we can flexibly extract a large number of nested submodels for adaptive inference based on the available deployment compute.

## Setup
To install the matmamba package and set up a fresh conda environment with all necessary dependencies, run the following script:

```bash
bash scripts/setup_env.sh
```

## Usage

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

You can find all pretrained models (MatMamba-Vision and MatMamba-LM) from the paper on Hugging Face: [MatMamba on Hugging Face](https://huggingface.co/collections/scaledfoundations/matmamba-670701480fa415dc2de60453)

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