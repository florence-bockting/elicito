# Case studies

## Elicitation

- [The human growth model](human-growth.ipynb): a prior for the Preece-Baines model, from five simulated experts.

## PyMC translation checks

These case studies use the models of the book *Bayesian Cognitive Modeling:
A Practical Course* (Lee & Wagenmakers, 2013). Each notebook:

- ports the PyMC3 version of a model to the current PyMC,
- translates the model with `elicito.adapter.pymc`,
- compares the TensorFlow samples with the samples of `pm.draw`.

The notebooks check the translation only. They do not elicit a prior.

- [Signal detection theory](signal-detection-theory.ipynb) (chapter 11)
- [The BART model of risk taking](bart-risk-taking.ipynb) (chapter 16)
- [Psychophysical functions](psychophysical-functions.ipynb) (chapter 12)
- [Memory retention](memory-retention.ipynb) (chapter 10)
- [The GCM model of categorization](gcm-categorization.ipynb) (chapter 17)
- [The SIMPLE model of memory](simple-memory.ipynb) (chapter 15)
- [Heuristic decision-making](heuristic-decision-making.ipynb) (chapter 18)
- [Extrasensory perception](extrasensory-perception.ipynb) (chapter 13)
- [Multinomial processing trees](multinomial-processing-trees.ipynb) (chapter 14)
- [Number concept development](number-concept-development.ipynb) (chapter 19)

## Source and licence

The PyMC3 models and the data come from the
[pymc-resources](https://github.com/pymc-devs/pymc-resources/tree/main/BCM/CaseStudies)
repository (commit `c24940c`). That repository uses the MIT licence,
Copyright (c) 2017 Repository Contributors. The folder `data/` keeps the
[licence text](data/LICENSE).

Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
Practical Course*. Cambridge University Press.
