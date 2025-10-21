# Simulation-based expert prior elicitation
## High-level idea
In Bockting et al. (2024), we propose an expert prior elicitation (EPE) method that builds upon recent advances in the field of **predictive prior elicitation** by Manderson (2023), da Silva et al (2023), and Hartmann and Agiashvili (2020). Our approach extends this line of research by introducing a **hybrid framework** that allows target quantities to be specified in both the parameter space and the observable space. Furthermore, the method supports custom specifications of generative models, target quantities, and elicitation techniques, enabled by its modular design and simulation-based approach.

/// note | EPE method - Core logic
The core logic of our EPE method can be summarized in a five-step workflow:

1. **Define the generative model**: Specify the generative model, including the functional form
of the data distribution and the parametric family of prior distributions.
2. **Define target quantities and elicitation techniques**: Select the set of target quantities
and determine the elicitation techniques to query the expert (cf. elicited summaries).
3. **Simulate elicited summaries**: Draw samples from the generative model and compute the
corresponding set of simulated elicited summaries.
4. **Evaluate discrepancy between simulated and expert-elicited summaries**: Assess the
discrepancy between the simulated and expert-elicited summaries using a multi-objective loss
function.
5. **Adjust prior hyperparameters to minimize discrepancy**: Apply an optimization scheme to
update the prior hyperparameters such that the loss function is minimized.
///

The following figure provides a **conceptual visualization** of this worklow within the [EPE process](EPE-process.md).

![conceptual workflow](../graphics/conceptual-workflow.png)

## Formalization
The definition of the generative model $\mathcal{M}$ comprises the data distribution, $p(y \mid \theta)$, and the specification of parametric prior distribution families, $p(\theta \mid \lambda)$. In the formulation presented in the subsequent Equation, we additionally introduce a vector of derived model parameters, $\eta$, and derived model outcomes, $z$, which are defined as transformations of the model parameters, $f_j^\text{par}(\theta_j)$, and the model outcomes, $f^\text{dat}(y)$, respectively:
\begin{align}\label{eq: generative-model}
\begin{split}
\left.
\begin{aligned}
    \theta &\sim p(\theta \mid \lambda) \\
    \eta_j &=f_j^\text{par}(\theta_j)
\end{aligned}
\right\} & \quad \text{parameter space} \\
\left.
\begin{aligned}
    y &\sim p(y \mid b,\eta) \\
    z &= f^\text{dat}(y)
\end{aligned}
\right\} & \quad \text{observable space} \\
\end{split}
\end{align}
with $\eta = (\eta_1,\ldots,\eta_J)$ and $b$ denotes a vector of potential covariates within a regression context.
Consequently, given a set of hyperparameter values $\lambda$, the generative model can be executed in forward mode to draw a total of $S$ samples ($s=1,\ldots,S$) for the parameters ($\theta^{(s)},\eta^{(s)}$) and observables ($y^{(s)},z^{(s)}$) specified by $\mathcal{M}$.

### Target quantities
Besides the generative model, it is necessary to formalize the selected set of \emph{target quantities}. A target quantity is defined as a random variable $T \in \mathfrak{T}$ distributed according to $p_T$
\begin{equation}
T \sim p_T,
\end{equation}
where $T$ may refer to any quantity in the parameter space, the observable space, or any transformation thereof, i.e., $\mathfrak{T} = \phi_\Theta(\Theta) \cup \phi_{\mathcal{Y}}(\mathcal{Y})$ with $\phi_\Theta, \phi_{\mathcal{Y}}$ denoting arbitrary transformations of the respective spaces. Using the previously introduced notation, the different types of target quantities can be represented as
\begin{align*}\label{eq: target-quantities}
    T&: \theta \sim p(\theta \mid \lambda) \quad\text{(parameter)} &T: y \sim p(y \mid b,\lambda) &\quad\text{(outcome)}\\
    T&:\eta \sim p(\eta \mid \lambda) \quad\text{(deriv. parameter)}  &T: z \sim p(z \mid b, \lambda) &\quad\text{(deriv. outcome)}
\end{align*}
Instead of directly employing the density $p_T$, our simulation-based approach involves sampling from the generative model yielding realizations drawn from $p_T$
\begin{equation}
\{t^{(s)}\}_{s=1}^S \sim p_T(t \mid \lambda).
\end{equation}
Typically, a \emph{set} of $P$ target quantities is defined, resulting in a vector of samples for each defined target quantity, $\{t_p^{(s)}\}_{s\in \mathbb N, p\in \mathbb N}$.

### Elicitation techniques
Given the set of target quantities, the next step is to formalize \emph{how} these quantities are queried from an expert, that is, to define the corresponding \emph{elicitation techniques}. For each of the $P$ target quantities $Q_p$ elicitation techniques $f_{p,q}$ are specified with $q=1,\ldots, Q_p$. An elicitation technique applied to a target quantity yields in an elicited summary
\begin{equation}
    E_m = f_{p,q}\left(\{t_p^{(s)}\}_{s=1}^S\right) \quad \text{for } p=1,\ldots,P.
\end{equation}
The index $m=1,\ldots,M$ arises from the combination of target quantities and the elicitation techniques specified for each target quantity. For instance, consider two target quantities, $\{t_1^{(s)}\}$ and $\{t_2^{(s)}\}$. Suppose the median, $\text{MED}(\{t_1^{(s)}\})$, is elicited for $\{t_1^{(s)}\}$, while the 25th, 50th, and 75th percentiles, $Q_{25}(\{t_2^{(s)}\})$, $Q_{50}(\{t_2^{(s)}\})$, and $Q_{75}(\{t_2^{(s)}\})$, are elicited for $\{t_2^{(s)}\}$. In this case, the total number of elicited summaries is $M = 4$.
An elicited summary $E_m$ may take the form of a scalar, for example, when the median or a quantile of a target quantity is elicited, or a vector. The final set of elicited summaries is denoted by $\{E_m\}_{m=1}^M$.

### Discrepancy measure
Once the generative model $\mathcal M$, target quantities $\{t_p^{(s)}\}$, and elicitation techniques $f_{p,q}$ are specified, the corresponding samples can be generated in forward mode for a given $\lambda$. This procedure results in the set of simulated summaries $\{E_m\}_{m=1}^M$.
Subsequently, the discrepancy between each pair of simulated and expert-elicited summaries is computed via a discrepancy measure $\mathcal D_m$, which may vary across the elicited summaries $\{E_m\}_{m=1}^M$. For clarity, we introduced a slightly adjusted notation to differentiate between simulated data (i.e., the simulated elicited summaries), denoted by $\tilde E_m$, and observed data (i.e., the expert-elicited summaries), denoted by $\dot E_m$. We further use $\tilde{E}_m(\lambda)$ to explicitly show the functional dependence of the simulated summaries on the hyperparameters $\lambda$, which are subject to the optimization procedure. The discrepancy between the simulated and expert-elicited summaries can then be denoted by
\begin{equation}
    \mathcal D_{m'}(\tilde{E}_{m'}(\lambda), \dot{E}_{m'})
\end{equation}
where $\mathcal D_{m'}$ denotes the chosen discrepancy measure. A new index, $m'=1,\ldots, M'$, is introduced to account for the possibility of concatenating multiple elicited summaries into a single representation. For example, revisiting the case described above, the three elicited quantiles for $\{t_2^{(s)}\}$ may be concatenated into a single vector. In this case, the total number of loss components is reduced, yielding $m' = 1,2$. However, the impact of concatenating elicited statistics on the optimization results is not yet fully understood and warrants further investigation.

### Multi-objective loss function
To obtain a single loss value, the individual discrepancy measures, $\mathcal D_{m'}$ (also referred to as \emph{loss components}), are combined via a \emph{multi-objective loss function} denoted by $\mathcal L(\lambda)$. As aggregation method, a weighted sum is employed, with weights $\gamma_{m'}$ assigned to each discrepancy measure $\mathcal D_{m'}$
\begin{equation}\label{eq: weighted-sum}
    \mathcal L(\lambda) = \sum_{{m'}=1}^{M'} \gamma_{m'}\cdot\mathcal D_{m'}(\dot{E}_{m'}(\lambda), \tilde{E}_{m'}).
\end{equation}
Under this loss formulation, the optimization objective is to determine the hyperparameters $\lambda^*$ that minimize the multi-objective loss, thereby reducing the overall discrepancy between the simulated and expert-elicited summaries
\begin{equation}\label{eq: minimization-objective}
    \lambda^* := \arg \min_\lambda \mathcal L(\lambda).
\end{equation}

### Gradient-based optimization
The current procedure for solving the introduced objective utilizes an iterative approach. In each iteration (or epoch), a set of simulated elicited summaries is derived based on the current hyperparameter vector $\lambda^{\text{epoch}}$. The discrepancy between these simulated and the expert-elicited summaries is then computed and aggregated to yield a total loss value, which subsequently drives the update of the hyperparameters $\lambda$
\begin{align}
\lambda^{\text{epoch}} = \lambda^{\text{epoch}-1} - \delta \frac{\partial \mathcal{L}}{\partial \lambda},
\end{align}
where $\delta$ denotes the learning rate. The updated hyperparameters are then used to simulate a new set of summaries, which enters the next loss computation. This updating procedure continues until a convergence criterion is satisfied. To implement this optimization procedure, we employ mini-batch stochastic gradient descent with automatic differentiation, facilitated by the reparameterization trick \citep{kingma2015variational}.
