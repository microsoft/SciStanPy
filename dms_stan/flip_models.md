FLIPv3 Models
=============
This file describes the models applied to the different datasets in FLIPv3.

## TrpB
The TrpB landscapes are constructed by measuring the counts of different variants over a series of timesteps before fitting a growth curve to that data for each variant. The parameters of the fit curves define a measure of variant fitness.

### Data
There are two measured quantities in the TrpB datasets:
1. The counts, which we define as the tensor $\left\{C \in \Z^{R \times T \times K} \vert c_{rtk} \ge 0\right\}$, where $R$ gives the number of replicates, $T$ gives the number of timepoints, $K$ gives the size of the library, $r$ is the replicate index, $t$ is the timepoint index, and $k$ is the variant index.
2. The time, which we define as the vector $\left\{x \in \R^T \vert x_t \ge 0.0\right\}$.

### Parameters
We model counts as being drawn from either a multinomial, binomial, or Poisson distribution. In that order, these are parametrized as

$$
\begin{align}
C_{rt} &\sim \textrm{Multinomial}(\theta_{rt})\\
c_{rtk} &\sim \textrm{Binomial}(\theta_{rtk})\\
c_{rtk} &\sim \textrm{Poisson}(\lambda_{rtk}),
\end{align}
$$

where the $\theta_{rt}$ parameter of the multinomial distribution is a simplex of length $K$ that describes the relative abundances of all variants in replicate $r$ and timepoint $t$, and $\theta_{rtk}$ and $\lambda_{rtk}$ are both scalars that describe a single variant $k$ in replicate $r$ at timepoint $t$. Note that $RT$ distributions must be learned when modeling counts as multinomial and $RTK$ distributions must be learned when modeling counts as Binomial or Poisson. See [`count_models.md`](count_models.md) for a more detailed description of the different models.

Using the parameters inferred from the counts as the dependent variable and the timepoints as an independent variable, a model of competitive growth is inferred. This model is parametrized as below regardless of the model used to infer counts:

$$
\begin{align}
\phi_{rtk} \sim \textrm{Normal}(\textrm{F}(x_t; \rho_{k}), \sigma_k),
\end{align}
$$

where $\phi_{rtk}$ is an unnormalized version of the parameters describing the count distribution, $\textrm{F}(x_t; \rho_{k})$ is some function that maps from time $x_t$ to the expected unnormalized count parameter for a given variant, $\rho_{k}$ is the set of parameters that define model $\mathrm{F}$ for variant $k$, and $\sigma_k$ measures the error in the fit for variant $k$. There will be $K$ such distributions inferred during sampling, one for each variant. See [`growth_models.md`](growth_models.md) for a description of the different growth models tested and their parametrizations (i.e., $\rho_k$).

Returning briefly to $\phi_{rtk}$: because Stan uses sampling to infer parameters, we cannot guarantee (nor, for that matter, expect) that the values of $\phi_{rtk}$ represent true relative proportions of variants, as we are explicitly assuming in the parametrization of the multinomial distribution and which represents powerful prior information for the parametrizations of the binomial and Poisson distributions. Thus, $\phi_{rtk}$ must be transformed into the appropriate normalized parameter for the count distributions by dividing by the total sum over the variant dimension. For instance, for the binomial distribution,

$$
\theta_{rtk} = \frac{\phi_{rtk}}{\sum_{k=1}^K\phi_{rtk}}
$$

Care must be taken with this step to keep the model identifiable, as $\theta_{rtk}$ has become invariant to the scale of $\phi_{rtk}$. Descriptive priors on the $\rho_k$ and $\sigma_k$ can ensure identifiability, however.

####