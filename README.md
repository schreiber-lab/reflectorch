# Reflectorch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![YAML](https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515)](https://yaml.org/)

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python version](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://www.python.org/)
![Repos size](https://img.shields.io/github/repo-size/schreiber-lab/reflectorch)
[![Jupyter Book Documentation](https://jupyterbook.org/badge.svg)](https://reflectorch.readthedocs.io)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


**Reflectorch** is a machine learning Python package for the analysis of X-ray and neutron reflectometry data via based on Pytorch. It provides functionality for the fast simulation of reflectometry curves on the GPU, customizable setup of the physical parameterization model and neural network architecture via YAML configuration files, and prior-aware training of neural networks as described in our paper [Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge](https://arxiv.org/abs/2307.05364).

## Installation

**Reflectorch** can be installed either from [PyPi](https://pypi.org/project/reflectorch/) via *pip* or from [conda-forge](https://anaconda.org/conda-forge/reflectorch/) via *conda*:

```bash
pip install reflectorch
```

or

```bash
conda install -c conda-forge reflectorch
```

Users with Nvidia GPUs need to additionally install Pytorch with CUDA support corresponding to their hardware and operating system according to the instructions from the [Pytorch website](https://pytorch.org/get-started/locally/)

## Get started

We provide an interactive Google Colab notebook for exploring the basic functionality of the package: [![Explore reflectorch in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vmunteanu/reflectorch/blob/master/explore_reflectorch.ipynb)<br>

## Documentation

The documentation is built with [Jupyter Book](https://jupyterbook.org/) and [Sphinx](https://www.sphinx-doc.org) and is hosted at [reflectorch.readthedocs.io](https://reflectorch.readthedocs.io).

## Citation
If you find our work useful in your research, please cite:
```
@misc{munteanu2023neural,
      title={Neural network analysis of neutron and X-ray reflectivity data: Incorporating prior knowledge for tackling the phase problem}, 
      author={Valentin Munteanu and Vladimir Starostin and Alessandro Greco and Linus Pithan and Alexander Gerlach and Alexander Hinderhofer and Stefan Kowarik and Frank Schreiber},
      year={2023},
      eprint={2307.05364},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```