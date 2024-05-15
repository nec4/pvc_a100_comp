# A100 vs. PVC Benchmarking on CIFAR10 and Reddit Datasets
===

## What is this repository?
---

A collection of python training/inference/submission scripts and accompanying
analysis notebooks for comparing some tasks on Intel Max Series 1550 ("Ponte
Vecchio" or "PVC") GPUs with NVIDIA A100 GPUs. 

## What do I need?
---

### A100
Typical CUDA-accelerated PyTorch and PyTorch Geometric python environment.

```
pyg                       2.4.0           py39_torch_2.1.0_cu118    pyg
pytorch                   2.1.1           py3.9_cuda11.8_cudnn8.7.0_0    pytorch
pytorch-cluster           1.6.3           py39_torch_2.1.0_cu118    pyg
pytorch-cuda              11.8                 h7e8668a_5    pytorch
pytorch-lightning         2.1.0              pyhd8ed1ab_0    conda-forge
pytorch-mutex             1.0                        cuda    pytorch
pytorch-scatter           2.1.2           py39_torch_2.1.0_cu118    pyg
pytorch-sparse            0.6.18          py39_torch_2.1.0_cu118    pyg
pytorch-spline-conv       1.2.2           py39_torch_2.1.0_cu118    pyg
torchaudio                2.1.1                py39_cu118    pytorch
torchmetrics              0.11.4           py39h2f386ee_1
torchtriton               2.1.0                      py39    pytorch
torchvision               0.16.1               py39_cu118    pytorch
```

CUDA 11.8 and Intel MPI 2021.11 was used. Each node is supoprted by 1x Ice Lake
8360Y.

### PVC
setup:
```
python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

python -m pip install oneccl_bind_pt==2.1.100+xpu -f https://pytorch-extension.intel.com/release-whl/stable/xpu/us/oneccl-bind-pt/

python -m pip install torch_geometric torch_sparse torch_scatter torch_cluster torch_spline_conv
```

relevant pacakges:
```
oneccl-bind-pt            2.1.100+xpu              pypi_0    pypi
intel-extension-for-pytorch 2.1.10+xpu               pypi_0    pypi
torch                     2.1.0a0+cxx11.abi          pypi_0    pypi
torch-cluster             1.6.3                    pypi_0    pypi
torch-geometric           2.6.0                    pypi_0    pypi
torch-geometric-benchmark 0.1.0                     dev_0    <develop>
torch-scatter             2.1.2                    pypi_0    pypi
torch-sparse              0.6.18                   pypi_0    pypi
torchaudio                2.1.0a0+cxx11.abi          pypi_0    pypi
torchvision               0.16.0a0+cxx11.abi          pypi_0    pypi
```

Intel oneAPI 2024.0.0 was used with Intel MPI 2021.11 and oneCCL 2021.12. Each
node is supported by 2x Intel(R) Xeon(R) Platinum 8480L (Sapphire Rapids).

## Other details
---
At the time of the repo creation, XPU support for `torch_geometric`
backends (`torch_sparse`, `torch_scatter`) was not available. Hopefully this
changes in the future :smile: .

