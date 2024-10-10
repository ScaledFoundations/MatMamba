# sudo apt-get update && sudo apt-get upgrade -y

# Install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/etc/profile.d/conda.sh

conda create -y -n matmamba python=3.9
source ~/miniconda3/bin/activate matmamba

# Install the mamba_ssm package
pip install causal-conv1d einops
git clone https://github.com/Abhinav95/mamba
cd mamba
pip install -e .
cd ..
python -c "import mamba_ssm"

pip install timm
pip install tiktoken

pip install -e .