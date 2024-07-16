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
    affiliation: 2
    equal-contrib: true
  - name: Alexander Hinderhofer
    orcid: 0000-0001-8152-6386
    affiliation: 1
    affiliation: 1
  - name: Alexander Gerlach
    orcid: 0000-0003-1787-1868
    affiliation: 1
  - name: Dmitry Lapkin
    orcid: 0000-0003-0680-8740
  - name: Frank Schreiber
    orcid: 0000-0003-3659-6718
    affiliation: 1
    corresponding: true
affiliations:
  - name: University of Tübingen, Institute of Applied Physics, Auf der Morgenstelle 10, 72076 Tübingen, Germany
    index: 1
  - name: University of Tübingen, Cluster of Excellence "Machine learning - new perspectives for science", Maria-von-Linden-Straße 6, 72076 Tübingen, Germany
    index: 2

date: 1 June 2024
bibliography: paper.bib

---

# Summary

We introduce `reflectorch`, a Python package which facilitates the full machine learning pipeline for the data domain of X-ray and neutron reflectometry. Firstly, the package allows the choice of different parameterizations of the scattering length density profile of a thin film, or generally, a layered structure, and the sampling of the ground truth physical parameters from user-defined ranges. Secondly, the package provides functionality for the fast simulation of reflectometry curves on the GPU using a vectorized implementation of the Abelès matrix formalism [@Abeles1950] and the augmentation of the theoretical curves with noise informed by experimental considerations. The architecture of the neural network as well as the training callbacks and hyperparameters can be easily customized from YAML configuration files. Notably, our implementation makes use of a special training procedure introduced in our publication [@Munteanu2024], in which prior boundaries for the target parameters are provided alongside the reflectivity curve as an additional input to the neural network.


# Statement of need

X-ray and neutron reflectometry (XRR and NR) are widely-used and indispensable experimental techniques for elucidating the structure of interfaces and thin films, i.e. layered systems. Experimentalists, having measured a reflectometry curve, are faced with the task of obtaining the physical parameters corresponding to an assumed parameterization of the scattering length density (SLD) profile along the depth of the investigated sample. The inverse problem is non-trivial and generally ambiguous due to the lack of phase information and experimental limitations. Recent years have seen an increased interest in the fast analysis of reflectometry data using machine learning techniques, as such methods could be adapted for real-time investigations at large scale facilities (i.e. synchrotron and neutron sources), potentially enabling closed loop experiments [@Pithan2023]. 

Our contribution to the open-source scientific software community is `reflectorch`, a deep learning package built in the Pytorch framework, which posseses a modular, object-oriented and customizable design. Reflectorch enables the user to simulate reflectivity curves in a fast and vectorized manner which takes advantage of the computing power of modern GPUs. The neural network architecture as well as the training callbacks and hyperparameters can be easily customized by editing YAML configuration files. As a result of the special training procedure, which incorporates minimum and maximum prior bounds for the parameters as described in [@Munteanu2024], the user is able to make use of prior knowledge about the investigated sample at inference time.  

# General workflow

Several types of parameterizations of the SLD profile are available, being implemented as subclasses of the `ParametricModel` class: 
1. the (default) box model parameterization without absorption (the parameter types being layer thicknesses, interlayer roughnesses and real valued layer SLDs), 
2. the box model parameterization with absorption (the values of the imaginary part of the layer SLDs being an additional parameter type) 
3. parameterization for multilayers with a repeating unit (as described in [@Munteanu2024]).

The parameters are represented as an instance of the `BasicParams` class, which encapsulates the parameter values and their prior bounds, also taking care of scaling these values to neural-network friendly ranges. The prior sampler (a subclass of `PriorSampler`) is responsible for sampling the values of the parameters and their prior bounds from their predefined ranges, in which a subclass of `SamplingStrategy` can be used to further restrict the values of some parameters with respect to others (such as the interlayer roughness not exceeding a fraction of the thicknesses of the adjacent layers). Based on the sampled parameters and on the momentum transfer (q) values generated by a subclass of `QGenerator`, batches of reflectivity curves are simulated and augmented with experimentally-informed noise provided by a subclass of `IntensityNoiseGenerator` and scaled to a neural network-friendly range by a subclass of `CurvesScaler` (which is also responsible for restoring the scaled curves to their original ranges). 

The trainer encapsulates the data loader, the Pytorch optimizer, the neural network, and other training hyperparameters, allowing the seamless training of the model and the saving of the resulting model weights and of the history of losses and learning rates. The neural network consists of an embedding network for the reflectivity curves and a multilayer perceptron. The architecture can be easily customized. Different embedding networks for the reflectivity curves are also available. Callback objects can be used to control the training process, such as scheduling the learning rate or periodically saving the model weights. 

Finally, the `EasyInferenceModel` class serves as a wrapper around trained models, simplifying the inference step. It also provides functionality for automatically downloading model weights and configuration files not locally available from a remote Huggingface repository.

# Related Work

There are several well-established packages designed for the conventional analysis of X-ray and neutron reflectometry data such as `GenX` [@GlaviGenX], `refnx` [@NelsonRefnx] and `refl1d` [@Maranville2020]. While several machine learning approaches pertaining to X-ray or neutron reflectometry have been proposed in various publications, `mlreflect` [@Greco2022Neural] is the only other properly packaged and documented software so far to the best of our knowledge. Previously developed in our group, `mlreflect` is built in the Tensorflow deep learning framework and has limited functionality compared to `reflectorch` (specifically, the SLD profile is limited to a single film). Still, it has been successfully adopted for use in the scattering community such as in the publication [@Schumi-Marecek2024]. 


# Acknowledgements

The research was part of a project (VIPR 05D23VT1 ERUMDATA) funded by the German Federal Ministry for Science and Education (BMBF). This work was partly supported by the consortium DAPHNE4NFDI in the context of the work of the NFDI e.V., funded by the German Research Foundation. Supported by the Cluster of Excellence “Machine Learning – New Perspectives for Science” funded by the German Research Foundation under Germany’s Excellence Strategy – reference Number EXC 2064/1 - project number 390727645.

# References


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