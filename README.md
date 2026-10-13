# Symmetry Discovery from Offline Reinforcement Learning Datasets

This repository contains code for discovering Lie group symmetries from offline RL
datasets and validating them as genuine dynamics symmetries of the environment.

## Methods

We provide implementations of four symmetry discovery approaches:

| Method | Description |
|--------|-------------|
| **TMT-NN** | Tangent manifold method using a score/denoiser network to estimate the orbit tangent space |
| **TMT-KNN** | Tangent manifold method using k-nearest neighbours and local PCA for tangent estimation |
| **LieSD** | Jacobian-based tangent space discovery via a learned dynamics model |
| **LieGAN** | Adversarial generator discovery — learns Lie algebra generators whose orbits are indistinguishable from data |

A discovered generator can be validated by checking whether a plain forward-dynamics
model trained with a soft equivariance penalty built from that generator (rather than a
hard architectural constraint) actually generalizes to orientations held out of training —
see "Dynamics-model equivariance check" in [`RL/README.md`](RL/README.md).

## Repository structure

```
RL/                               # Symmetry discovery + dynamics-model validation
  train_tmt_nn.py                 # TMT-NN training
  train_tmt_knn.py                # TMT-KNN training
  train_liesd.py                  # LieSD training
  train_liegan.py                 # LieGAN training
  train_dynamics_model.py         # Soft equivariance-penalty dynamics model (Reacher)
  eval_dynamics_model_orientation.py  # Held-out-orientation evaluation for the above
  generate_datasets.py            # Dataset generation
  dataset_loader.py                # Shared dataset loading
  representations.py              # Ground-truth generators and evaluation metrics
  jax_rl/
    networks/                     # Network implementations for all four methods
    agents/bc/                    # _taylor_expm (used by the equivariance penalty)
    envs/                         # Custom environments
```

## Installation

```bash
git clone https://github.com/mschoene/tmt.git
cd tmt

# Create and activate the conda environment (installs JAX, Flax, Gymnasium+MuJoCo,
# Minari, EMLP, PyTorch, and the other dependencies listed in RL/environment.yml)
conda env create -f RL/environment.yml
conda activate jax-rl
```

`RL/environment.yml` installs the CPU build of JAX by default; for a CUDA/GPU build
matching your driver, follow https://docs.jax.dev/en/latest/installation.html instead.

Alternatively, `RL/requirements.txt` lists the same Python dependencies for a plain
`pip install -r RL/requirements.txt` into an existing Python 3.10 environment. All
scripts are run directly from within `RL/` (no package installation step needed).

## Usage

See [`RL/README.md`](RL/README.md) for detailed instructions on dataset generation,
running each discovery method, and the dynamics-model equivariance check.

## Acknowledgements

- `emlp` (Finzi et al.) supplies the representation-theory primitives (`V`, `T`, `SO`, ...)
  used throughout `representations.py` to define each environment's ground-truth symmetry
  group and representations, used for diagnostic comparison against discovered generators.
- The dataset/replay-buffer scaffolding in `jax_rl/` is adapted from
  **jax-rl** — Kostrikov, 2021.
  [https://github.com/ikostrikov/jax-rl](https://github.com/ikostrikov/jax-rl)
