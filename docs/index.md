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
Given a generative model and a set of initial hyperparameters defining the prior distributions,
the model can be run in forward mode to simulate elicited summaries by computing the predefined
target quantities and summary statistics. These simulated summaries are then compared with the
expert-elicited summaries obtained during the expert-elicitation stage. An iterative optimization
scheme is employed to update the hyperparameters of the parametric prior distributions so as to
minimize the discrepancy between simulated and expert-elicited summaries. In other words, the
objective is to identify the vector of hyperparameters that yields the closest alignment between
simulated and expert-elicited summaries.

![conceptual workflow](graphics/conceptual-workflow.png)

The core logic of the expert prior elicitation method proposed in Bockting et al. (2024) can be summarized in a five-step workflow:

/// note | Core logic of method underlying elicito
1. *Define the generative model*: Specify the generative model, including the functional form
of the data distribution and the parametric family of prior distributions.
2. *Define target quantities and elicitation techniques*: Select the set of target quantities
and determine the elicitation techniques to query the expert (cf. elicited summaries).
3. *Simulate elicited summaries*: Draw samples from the generative model and compute the
corresponding set of simulated elicited summaries.
4. *Evaluate discrepancy between simulated and expert-elicited summaries*: Assess the
discrepancy between the simulated and expert-elicited summaries using a multi-objective loss
function.
5. *Adjust prior hyperparameters to minimize discrepancy*: Apply an optimization scheme to
update the prior hyperparameters such that the loss function is minimized.
///


### Main References

+ [Software Paper] Bockting F. & Bürkner PC (2025). elicito: A Python package for expert-prior elicitation. arXiv.
[Preprint](https://arxiv.org/pdf/2506.16830)
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
