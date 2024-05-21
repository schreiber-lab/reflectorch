---
title: 'reflectorch: a deep learning package for X-ray and neutron reflectometry'
tags:
  - Python
  - Pytorch
  - machine learning
  - surface scattering physics
  - reflectometry
authors:
  - name: Valentin Munteanu
    orcid: 0000-0002-5897-3863
    affiliation: 1
    equal-contrib: true
  - name: Vladimir Starostin
    orcid: 0000-0003-4533-6256
    affiliation: 1
    equal-contrib: true
  - name: Alexander Hinderhofer
    orcid: 0000-0001-8152-6386
    affiliation: 1
    affiliation: 1
  - name: Alexander Gerlach
    orcid: 0000-0003-1787-1868
    affiliation: 1
  - name: Dmitry Lapkin
    orcid: 0000-0000-0000-0000
  - name: Frank Schreiber
    orcid: 0000-0003-3659-6718
    affiliation: 1
    corresponding: true
affiliations:
 - name: University of Tübingen, Institute for Applied Physics, Auf der Morgenstelle 10, 72076 Tübingen, Germany
   index: 1

date: 10 April 2024
bibliography: paper.bib

---

# Summary

We introduce `reflectorch`, a Python package which facilitates the full machine learning pipeline for the data domain of X-ray and neutron reflectivity. Firstly, the package allows the choice of different parameterizations of the scattering length density profile of a thin film and the sampling of the ground truth physical parameters from user-defined ranges. Secondly, the package provides functionality for the fast simulation of reflectivity curves on the GPU using a vectorized implementation of the Abelès matrix formalism [@Abeles1950] and the augmentation of the theoretical curves with noise informed by experimental considerations. The architecture of the neural network as well as the training callbacks and hyperparameters can be easily customized from YAML configurtation files. Notably, our implementation makes use of a special training procedure introduced in our publication [@Munteanu2024], in which prior boundaries for the target parameters are provided alongside the reflectivity curve as an additional input to the neural network.


# Statement of need

`Reflectorch` is a Python package for machine learning based analysis of reflectometry data. The package is built on top of the Pytorch deep learning framework. 


The core

`Reflectorch` was designed to be used by researchers both at their home institutes and at synchrotron facilities.

...Utilized in closed-loop experiments at the beamline... [@Pithan2023]

# General workflow

Several types of parameterizations of the scattering length density profile are implemented: the (default) box model parameterization without absorption (the parameter types being layer thicknesses, interlayer roughnesses and real valued layer SLDs), the box model parameterization with absorption (with imaginary valued layer SLDs as an additional parameter type) and a parameterization for multilayers with repeating unit. The parameters are represented as an instance of the `Params` class (or its subclasses). The `PriorSampler` is responsible 

# Related Work

There are several well-established packages designed for the classical analysis of X-ray and neutron reflectivity data such as `GenX` [@GlaviGenX], `refnx` [@NelsonRefnx] and `refl1d` [@Maranville2020]. While several machine learning approaches pertaining to X-ray or neutron reflectometry have been proposed in various publications, `mlreflect` [@Greco2022Neural] is the only other properly packaged and documented software. Previously developed in our research group, `mlreflect` is built in the Tensorflow deep learning framework and has limited functionality compared to `reflectorch`. Still, it has been successfully adopted for usage by people in the scattering community such as in the publication [@Schumi-Marecek2024]. 


<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }' -->

# Acknowledgements

The research was part of a project (VIPR 05D23VT1 ERUMDATA) funded by the German Federal Ministry for Science and Education (BMBF). This work was partly supported by the consortium DAPHNE4NFDI in the context of the work of the NFDI e.V., funded by the German Research Foundation.

# References