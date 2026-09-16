# Case studies

These case studies use the models of the book *Bayesian Cognitive Modeling:
A Practical Course* (Lee & Wagenmakers, 2013). Each notebook:

- ports the PyMC3 version of a model to the current PyMC,
- translates the model with `elicito.adapter.pymc`,
- compares the TensorFlow samples with the samples of `pm.draw`.

The notebooks check the translation only. They do not elicit a prior.

- [Signal detection theory](signal-detection-theory.py) (chapter 11)
- [The BART model of risk taking](bart-risk-taking.py) (chapter 16)
- [Psychophysical functions](psychophysical-functions.py) (chapter 12)
- [Memory retention](memory-retention.py) (chapter 10)
- [The GCM model of categorization](gcm-categorization.py) (chapter 17)
- [The SIMPLE model of memory](simple-memory.py) (chapter 15)
- [Heuristic decision-making](heuristic-decision-making.py) (chapter 18)
- [Extrasensory perception](extrasensory-perception.py) (chapter 13)
- [Multinomial processing trees](multinomial-processing-trees.py) (chapter 14)

## Source and licence

The PyMC3 models and the data come from the
[pymc-resources](https://github.com/pymc-devs/pymc-resources/tree/main/BCM/CaseStudies)
repository (commit `c24940c`). That repository uses the MIT licence,
Copyright (c) 2017 Repository Contributors. The folder `data/` keeps the
[licence text](data/LICENSE).

Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
Practical Course*. Cambridge University Press.
