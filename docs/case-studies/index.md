# Case studies

## Elicitation

- [The human growth model](human-growth.ipynb): a prior for the Preece-Baines model, from five simulated experts.

## PyMC models from Bayesian Cognitive Modeling

These case studies use the models of the book *Bayesian Cognitive Modeling:
A Practical Course* (Lee & Wagenmakers, 2013). Each case study takes one
model and:

- translates the PyMC model with `elicito.adapter.pymc`,
- checks that the TensorFlow samples match the samples of `pm.draw`,
- elicits a prior from two or three experts, and fits the model with PyMC.

- [Memory retention](memory-retention.ipynb) (chapter 10.1)
- [Signal detection theory](signal-detection-theory.ipynb) (chapter 11.2)
- [Psychophysical functions](psychophysical-functions.ipynb) (chapter 12.1)
- [Extrasensory perception](extrasensory-perception.ipynb) (chapter 13.3)
- [Multinomial processing trees](multinomial-processing-trees.ipynb) (chapter 14.1)
- [The BART model of risk taking](bart-risk-taking.ipynb) (chapter 16.1)
- [The GCM model of categorization](gcm-categorization.ipynb) (chapter 17.2)
- [Heuristic decision-making](heuristic-decision-making.ipynb) (chapter 18.2)
- [Number concept development](number-concept-development.ipynb) (chapter 19.1)

## Source and licence

The PyMC3 models and the data come from the
[pymc-resources](https://github.com/pymc-devs/pymc-resources/tree/main/BCM/CaseStudies)
repository (commit `c24940c`). That repository uses the MIT licence,
Copyright (c) 2017 Repository Contributors. The folder `data/` keeps the
[licence text](data/LICENSE).

Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
Practical Course*. Cambridge University Press.
