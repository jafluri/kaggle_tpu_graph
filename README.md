# TPU Graph 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repo contains the code used for the [3rd place solution](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456377) of the 
[Google - Fast or Slow? Predict AI Model Runtime](https://www.kaggle.com/competitions/predict-ai-model-runtime) 
competition on Kaggle.

The final submission was a combination of different networks trained at different stages 
during the development of this repo. For completeness, all submissions have separate branches. 
However, the state of the repo during the development was sometimes quite messy. The codes of the submissions 
follow all the same pattern, but use slightly different network architectures, conventions etc. 
Experiments show that the main branch achieves the same or better results for the layout and is therefore recommended.
Note that the Kendall's Tau score is a very noisy evaluation metric. Training the same network in the same way 
can lead to a difference of 0.1 in the score. One has to be careful when comparing different networks and always 
train them multiple times.

The main branch can only be used for the layout collection. The tile collection has its own branch and does not need 
any special preparation.

### Installation

This package was developed and tested with Python 3.11. You might want to install [`pytorch`](https://pytorch.org/) and [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) 
manually with pre-build wheels using the same CUDA version. Otherwise, the installation is in theory as easy as

```bash
pip install -e .
```

#### Dependencies

This repo depends on `pytorch` which should be installed with GPU support.

## Usage

The package uses the data of the competition and has two scripts in the `scripts` folder. 
The first one `add_features.py` extracts additional features from the data and adds some derived features. 
In detail, it does:
- Extract additional features from the protocol buffer files
- Add positional encodings using [RWPE](https://arxiv.org/pdf/2110.07875.pdf) with the asymmetric and symmetric 
  adjacency matrices
- Logs features with large dynamic range
- Creates new features from the 30 dimensional features in the form of `x % 128` and `(x // 128) / 10`, where `x` 
  is the original feature. This is done because of the register size of the TPU.
- Add a virtual output node to the graph that connects all nodes with outputs.

It requires pointers to directories containing the protocol buffer files and the npz files. You can have a look at the 
full signature with `python scripts/add_features.py --help`. Note that the RWPE can take a while and use a lot of 
memory for the larger graph.

The second script `train.py` trains a network on the data. You can have a look at the signature with 
`python scripts/train.py --help`. The script requires a path to a directory containing the npz files generated 
with `add_features.py`. After every epoch, validation and test set are evaluated and saved along with the
model. If there are multiple GPUs available, one can specify the number of GPUs to use. Note that the backend and 
master port are hard-coded in the script and might need to be changed. 

## Development

### Hooks

Pre-commit hooks come in any color you'd like

```bash
pre-commit install -c .hooks/.pre-commit-config.yaml
```

### TODO

- Clean up, better docs