


## About

**Reflectorch** is a machine learning Python package for the analysis of X-ray and neutron reflectometry data via based on Pytorch. It provides functionality for the fast simulation of reflectometry curves on the GPU, customizable setup of the physical parameterization model and neural network architecture via YAML configuration files, and prior-aware training of neural networks as described in our paper [Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge](https://arxiv.org/abs/2307.05364).

## Installation

**Reflectorch** can be installed either via *pip* or *conda*:

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