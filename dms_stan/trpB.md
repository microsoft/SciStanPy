TrpB Generative Models
=====================
This file describes the distributions available for modeling DMS datasets in DMS Stan.

## The Multinomial Distribution

The counts resulting from next generation sequencing are best approximated as draws from a multinomial distribution. We can imagine the process of sequencing as drawing with replacement (assuming the population of molecules in the sequenced sample is much larger than the number of counts returned, which is valid) from a "bag" of molecules where each variant is represented by some subpopulation of those molecules; the number of counts we "draw" will be a function of the relative proportions of those subpopulations, and the relative proportions will be a function of the fitness of the variants.

The probability mass function for the multinomial distribution is parametrized as below in DMS Stan:

$$
P(\mathbf{c} \vert \mathbf{\theta}, N) = \frac{N!}{c_1!c_2!\cdots c_k!}\theta_1^{c_1}\theta_2^{c_2}\cdots\theta_k^{c_k},
$$

where $N$ is the total number of counts, $c_k$ is the number of counts observed for variant $k$, $\theta_k$ is the proportion of the population made up of variant $k$, and $\mathbf{\theta}$ is a simplex of all $\theta_k$.

## The Binomial Distributrion

The multinomial distribution is just a generalization of the binomial distribution, which is used for modeling the number of times an event is observed among $N$ trials. The probability mass function for the binomial distribution is parametrized very similarly to the multinomial distribution:

$$
P(c_k \vert \theta_k, N) = \frac{N!}{c_k!(N-c_k)!} \theta_k^{c_k} (1 - \theta_k)^{N-c_k},
$$

where $\theta_k$ is the probability of an event happening and $c_k$ is the number of observed instances of that event among $N$ trials.

The binomial distribution differs from the multinomial in that it does not consider the joint probability of all variants in the population, treating the distribution of counts seen for each variant independently. In other words, there is no requirement that $\sum_{i=1}^k\theta_i = 1$ for the binomial distribution as there is for the multinomial.

## The Poisson Distribution
In the limit of $N \rightarrow \infty$ and $\theta_k \rightarrow 0$, the binomial distribution can be approximated as a Poisson distribution. In DMS studies, we typically are evaluating many variants (meaning $\theta_k$ should be low, at least for early timepoints) with high throughput next generation sequencing (meaning $N$ should be high). The Poisson approximation of the binomial distribution may thus be valid.

In DMS Stan, the probability mass function for the Poisson distribution is parametrized as follows:

$$
P(c_k \vert \lambda_k ) = \frac{\lambda_k^{c_k}}{c_k!}\textrm{e}^{-\lambda_k}
$$

The parameter $\lambda$ approximates $N\theta_k$.

