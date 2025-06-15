---8<--- "README.md:description"
# `elicito`: A Python package for expert prior elicitation

**elicito** supports the implementation of expert prior elicitation methods.
The goal of *expert prior elicitation* is to specify prior distributions for parameters in a
Bayesian model that accurately reflect the expectations of a domain expert.

With its modulare framework `elicito` supports

+ a wide range of generative models (statistical model describing the data-generating process),
+ different types of expert knowledge (which information is extracted from the domain expert?)
+ various types of elicitation techniques (how the information is extracted from the domain expert)
+ loss functions (criterion used to match expert knowledge with simulated model quantities)

## Conceptual background
The general workflow underlying **elicito** closely resembles the approach of prior
predictive checks: We simulate from the joint model $p(\theta, y)$ and
assess how well the resulting prior predictions align with the expert’s expectations. If
there is a discrepancy between the expert’s expectations and the model simulations,
the prior specification needs to be adjusted accordingly.
The general workflow of our framework can be summarized as follows:

1. Define the generative model : Define the generative model including dimensionality
and parameterization of prior distribution(s). (Setup stage)
2. Identify variables and elicitation techniques for querying expert knowledge: Select
the set of variables to be elicited from the domain expert (target quantities) and
determine which elicitation techniques to use for querying the selected variables
from the expert (elicited statistics). (Setup stage)
3. Elicit statistics from expert and simulate corresponding predictions from the gener-
ative model : Sample from the generative model and perform all necessary computa-
tional steps to generate model predictions (model-elicited statistics) corresponding
to the set of expert-elicited statistics. (Elicitation stage)
4. Evaluate consistency between expert knowledge and model predictions: Evaluate the
discrepancy between the model- and expert-elicited statistics via a multi-objective
loss function. (Fitting stage)
5. Adjust prior to align model predictions more closely with expert knowledge: Use
mini-batch stochastic gradient descent to adjust the prior so as to reduce the loss.
(Fitting stage)

### Main References

+ [Software Paper] Bockting F. & Bürkner PC (2025). elicito: A Python package for expert-prior elicitation.
[PDF is coming soon]()
+ [Methods Paper] Bockting F., Radev ST, Bürkner PC (2024). Simulation-based prior knowledge elicitation
for parametric Bayesian models. *Scientific Report, 14*(1), 17330. [PDF](https://www.nature.com/articles/s41598-024-68090-7)
+ [Methods Paper] Bockting F., Radev ST, Bürkner PC (2025). Expert-elicitation method for non-parametric joint priors using
normalizing flows. *Statistics and Computing*. [PDF](https://arxiv.org/abs/2411.15826)

## Implementation (in a nutshell)
The primary user interface of **elicito** is the `Elicit` class, through which the user can specify
the entire elicitation procedure. The arguments of the `Elicit` class are designed to capture all
necessary information required to implement an elicitation method.
A brief overview of these arguments is provided below:

+ `model`: Defines the generative model used in the elicitation procedure.
+ `parameters`: Specifies assumptions regarding the prior distributions over model parameters,
    including (hyper)parameter constraints, dimensionality, and parametric form.
+ `targets`: Defines the elicited statistics in terms of target quantities and corresponding
  elicitation techniques. Also specifies the discrepancy measure and weight used for the
    associated loss component.
+ `expert`: Provides the expert information that serves as the basis for the learning criterion.
+ `optimizer`: Specifies the optimization algorithm to be used, along with its
    hyperparameters (e.g., learning rate).
+ `trainer`: Configures the overall training procedure, including settings such as the random
    seed, number of epochs, sample size, and batch size.
+ `initializer`: Defines the initialization strategy for the hyperparameters used to
    instantiate the simulation-based optimization process.
+ `networks`: Specifies the architecture of the deep generative model; required only when
    using non-parametric prior distributions.

By configuring these core components, **elicito** supports a wide range of elicitation
methods, including both structural and predictive approaches, univariate and
multivariate as well as parametric and nonparametric prior distributions.

## Getting started
+ See introductory tutorials for learning

    + independent, parametric prior distributions [Notebook](tutorials/getting-started-param.py)
    + joint, non-parametric prior distributions [Notebook](tutorials/getting-started-deep.py)

+ Browse through the

    + package **API** to get familiar with the functionalities
    + [How-to-guides](how-to-guides/index.md) to see further use cases and explanations

Please don't hesitate to raise an [issue](https://github.com/florence-bockting/elicito/issues) if you get stuck, encounter an error, or have a question.
We are happy to hear from you.
