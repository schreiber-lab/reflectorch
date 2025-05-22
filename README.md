# Reflectorch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![YAML](https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515)](https://yaml.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD700.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/valentinsingularity/reflectivity)

[![Python version](https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://www.python.org/)
![CI workflow status](https://github.com/schreiber-lab/reflectorch/actions/workflows/ci.yml/badge.svg)
![Repos size](https://img.shields.io/github/repo-size/schreiber-lab/reflectorch)
<!-- [![CodeFactor](https://www.codefactor.io/repository/github/schreiber-lab/reflectorch/badge)](https://www.codefactor.io/repository/github/schreiber-lab/reflectorch) -->
[![Jupyter Book Documentation](https://jupyterbook.org/badge.svg)](https://jupyterbook.org/)
[![Documentation Page](https://img.shields.io/badge/Documentation%20Page-%23FFDD33.svg?style=flat&logo=read-the-docs&logoColor=black)](https://schreiber-lab.github.io/reflectorch/)
<!-- [![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) -->


**Reflectorch** is a machine learning Python package for the analysis of X-ray and neutron reflectometry data, written by [Vladimir Starostin](https://github.com/StarostinV/) & [Valentin Munteanu](https://github.com/valentinsingularity) at the University of Tübingen. It provides functionality for the fast simulation of reflectometry curves on the GPU, customizable setup of the physical parameterization model and neural network architecture via YAML configuration files, and prior-aware training of neural networks as described in our paper [Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge](https://doi.org/10.1107/S1600576724002115).

## Installation

**Reflectorch** can be installed from [![PyPi](https://img.shields.io/badge/PyPi-3776AB.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/reflectorch/) via ``pip``:

<!-- or from [![conda-forge](https://img.shields.io/badge/conda--forge-44A833.svg?style=flat&logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/reflectorch/) via ``conda``: -->

```bash
pip install reflectorch
```

<!-- or

```bash
conda install -c conda-forge reflectorch
``` -->

Alternatively, one can clone the entire Github repository and install the package in editable mode:

```bash
git clone https://github.com/schreiber-lab/reflectorch.git
pip install -e .
```

For development purposes, the package can be installed together with the optional dependencies for building the distribution, testing and documentation:

```bash
git clone https://github.com/schreiber-lab/reflectorch.git
pip install -e .[tests,docs,build]
```

Users with Nvidia **GPU**s need to additionally install **Pytorch with CUDA support** corresponding to their hardware and operating system according to the instructions from the [Pytorch website](https://pytorch.org/get-started/locally/)

## Get started

[![Documentation Page](https://img.shields.io/badge/Documentation%20Page-%23FFDD33.svg?style=flat&logo=read-the-docs&logoColor=black)](https://schreiber-lab.github.io/reflectorch/)
 The full documentation of the package, containing tutorials and the API reference, was built with [Jupyter Book](https://jupyterbook.org/) and [Sphinx](https://www.sphinx-doc.org) and it is hosted at the address: [https://schreiber-lab.github.io/reflectorch/](https://schreiber-lab.github.io/reflectorch/).

[![Interactive Notebook](https://img.shields.io/badge/Interactive%20Notebook-%23F9AB00.svg?style=flat&logo=google-colab&logoColor=black)](https://colab.research.google.com/drive/1rf_M8S_5kYvUoK0-9-AYal_fO3oFl7ck?usp=sharing)
We provide an interactive Google Colab notebook for exploring the basic functionality of the package: [![Explore reflectorch in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rf_M8S_5kYvUoK0-9-AYal_fO3oFl7ck?usp=sharing)<br>

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD700.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/valentinsingularity/reflectivity)
Configuration files and the corresponding pretrained model weights are hosted on Huggingface: [https://huggingface.co/valentinsingularity/reflectivity](https://huggingface.co/valentinsingularity/reflectivity).

<!-- [![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/)
Docker images for reflectorch *will* be hosted on Dockerhub. -->

## Contributing
If you'd like to contribute to the package, please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Citation
If you find our work useful in your research, please cite as follows:
```
@Article{Munteanu2024,
  author    = {Munteanu, Valentin and Starostin, Vladimir and Greco, Alessandro and Pithan, Linus and Gerlach, Alexander and Hinderhofer, Alexander and Kowarik, Stefan and Schreiber, Frank},
  journal   = {Journal of Applied Crystallography},
  title     = {Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge},
  year      = {2024},
  issn      = {1600-5767},
  month     = mar,
  number    = {2},
  volume    = {57},
  doi       = {10.1107/s1600576724002115},
  publisher = {International Union of Crystallography (IUCr)},
}
```