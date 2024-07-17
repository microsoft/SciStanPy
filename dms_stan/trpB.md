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

The multinomial distribution is just a generalization of the binomial distribution, which is used for modeling the number of times an event is observed among $N$ trials. The binomial distribution is parametrized very similarly to the multinomial distribution:

$$
P(c_k | \theta_k, N) = \frac{N!}{c_k!(N-c_k)!} \theta_k^c (1 - \theta_k)^{N-c_k},
$$

where $\theta_k$ is the probability of an event happening and $c_k$ is the number of observed instances of that event among $N$ trials.

The binomial distribution differs from the multinomial in that it does not consider the joint probability of all variants in the population, treating the distribution of counts seen for each variant independently. In other words, there is no requirement that $\sum_{i=1}^k\theta_i = 1$ for the binomial distribution as there is for the multinomial.

In this formulation, $\frac{N!}{c!(N-c)!}$ gives the number of possible ways we could have observed the event $c$ times and not observed the event $N-c$ times -- the term $1-\theta$ gives the probability of not observing the event.

We might think of the number of counts resulting for a *single* variant from a next generation sequencing run as being binomially distributed. We can imagine the process of sequencing as drawing with replacement (assuming the population of molecules in the sequenced sample is much larger than the number of counts returned, which is valid) from a "bag" of molecules where a given variant $k$ is represented by some subpopulation of those molecules; the number of counts we "draw" will be a function of proportion of this subpopulation among all other populations. Thus, the probability of having $c$ observations of variant $k$ will be given by $\theta$ and the probability of having $N-c$ observations of any other variant will be $1-\theta$.

We can generalize from the single variant case to $V$ variants using the rules of conditional probability. Specifically, we can break the joint probability distribution $P(\mathbf{c} | \mathbf{\theta}, N) = P(c_1, c_2, \cdots, c_V \vert \theta_1, \theta_2, \cdots, \theta_V, N)$ into

$$
\begin{align}
P(\mathbf{c} | \mathbf{\theta}, N) &= P(c_1 \vert \theta_1, N)P(c_2 \vert \theta_2, N, c_1)...P(c_V \vert \theta_V, N, c_1, c_2, \cdots, c_{V-1}) \\
P(\mathbf{c} | \mathbf{\theta}, N) &= P(c_1 \vert \theta_1, N)P(c_2 \vert \theta_2, N - c_1)...P(c_V \vert \theta_V, N - c_1 - c_2 - \cdots  c_{V-1}) ,
\end{align}
$$

where we are assuming that all $\theta_k$ are independent from one another. Expanding the product gives us

$$
\begin{align}
P(\mathbf{c} | \mathbf{\theta}, N) = &\left(\frac{N!}{c_1!(N-c_1)!}\right) \theta_1^{c_1} (1 - \theta_1)^{N-c_1}
\left(\frac{(N - c_1)!}{c_2!(N-c_1-c_2)!}\right) \theta_2^{c_2} (1 - \theta_2)^{N-c_1-c_2}
\cdots \\
&\left(\frac{(N - \sum_{k=1}^{V-1}c_k)!}{c_V!(N-\sum_{k=1}^Vc_k)!}\right) \theta_V^{c_V} (1 - \theta_V)^{N-\sum_{k=1}^Vc_k}.
\end{align}
$$

The term $(N-c_k)!$ in the denominator of variant $k$ cancels out the equivalent term in the numerator of variant $k + 1$, the the above reduces to


$$
\begin{align}
P(\mathbf{c} | \mathbf{\theta}, N) = &\left(\frac{N!}{c_1!}\right) \theta_1^{c_1} (1 - \theta_1)^{N-c_1}
\left(\frac{1}{c_2!}\right) \theta_2^{c_2} (1 - \theta_2)^{N-c_1-c_2}
\cdots \\
&\left(\frac{1}{c_V!(N-\sum_{k=1}^Vc_k)!}\right) \theta_V^{c_V} (1 - \theta_V)^{N-\sum_{k=1}^Vc_k} \\
= &\left(\frac{N!}{c_1!c_2! \cdots c_V!(N-N)!}\right) \theta_V^{c_V} (1 - \theta_V)^{N-N} \prod_{k=1}^{V-1} \theta_k^{c_k} (1 - \theta_k)^{N-\sum_{i=1}^kc_i} \\
= &\left(\frac{N!}{c_1!c_2! \cdots c_V!}\right) \theta_V^{c_V}\prod_{k=1}^{V-1} \theta_k^{c_k} (1 - \theta_k)^{N-\sum_{i=1}^kc_i}.
\end{align}
$$

Recalling that, given $\{c_i, \theta_i\}_{i < k}$, the term $\theta_k^{c_k}$ gives the probability of observing variant $k$ and the term $(1 - \theta_k)^{N - \sum_{i=1}^kc_i}$ gives us the probability of observing any other variant $j > k$, we know that

$$
(1 - \theta_k)^{N - \sum_{i=1}^kc_i} = \theta_{k+1}^{c_{k+1}}(1 - \theta_{k+1})^{N - \sum_{i=1}^{k+1}c_i}.
$$




Thus, it follos that

$$
\frac{(1 - \theta_k)^{N - \sum_{i=1}^kc_i}}{} = \theta_{k+1}^{c_{k+1}}(1 - \theta_{k+1})^{N - \sum_{i=1}^{k+1}c_i}.
$$

which can be rearranged to give

the binomial distribution to a distribution of multi-variant case by recognizing that the probability of seeing any given distribution of counts for multiple variants is just

$$
P(c_1 \vert \theta_1, N)P(c_2 \vert \theta_2, N - c_1)P(c_3 \vert \theta_3, N - c_1 - c_2)\cdots P\left(c_k \vert \theta_k, N - \sum_{i=1}^{k-1}c_i\right)
$$

or, equivalently

$$
P(c_1 \vert \theta_1, N)P(c_2 \vert \theta_2, N, c_1)P(c_3 \vert \theta_3, N, c_1, c_2)...P(c_k \vert \theta_k, N, c_1, c_2, \cdots, c_{k-1}) = P(\mathbf{c} | \mathbf{\theta}, N),
$$

where $\mathbf{c}$ is a vector of all counts observed for all variants and $\mathbf{\theta}$ is a simplex giving the proportion of the population of all variants made up by each variant.

Written out fully, this product is

We can expand the combinatorial section of the above product as follows

$$
\begin{align}
\left(\frac{N!}{c_1!(N - c_1)!}\right)\left(\frac{(N - c_1)!}{c_2!(N - c_1 - c_2)!}\right)\left(\frac{(N - c_1 - c_2)!}{c_3!(N - c_1 - c_2 - c_3)!}\right) \cdots \\ \left(\frac{(N - \sum_{i=1}^{k-2}c_i)!}{c_{k-1}!(N - \sum_{i=1}^{k-1}c_i)!}\right)\left(\frac{(N - \sum_{i=1}^{k-1}c_i)!}{c_{k}!(N - \sum_{i=1}^{k}c_i)!}\right)
\end{align}
$$

to see that many terms cancel out

$$
\begin{align}
\left(\frac{N!}{c_1!\sout{(N - c_1)!}}\right)\left(\frac{\sout{(N - c_1)!}}{c_2!\sout{(N - c_1 - c_2)!}}\right)\left(\frac{\sout{(N - c_1 - c_2)!}}{c_3!\sout{(N - c_1 - c_2 - c_3)!}}\right) \cdots \\ \left(\frac{\sout{(N - \sum_{i=1}^{k-2}c_i)!}}{c_{k-1}!\sout{(N - \sum_{i=1}^{k-1}c_i)!}}\right)\left(\frac{\sout{(N - \sum_{i=1}^{k-1}c_i)!}}{c_{k}!(N - \sum_{i=1}^{k}c_i)!}\right),
\end{align}
$$

yielding

$$
\begin{align}
\frac{N!}{c_1!c_2! \cdots c_k!(N - \sum_{i=1}^kc_i)!} = \frac{N!}{c_1!c_2! \cdots c_k!(N - N)!} = \frac{N!}{c_1!c_2! \cdots c_k!}.
\end{align}
$$

Expanding the probability portion of the earlier-described product gives us

$$
\begin{align}
\prod_{k=1}^V\theta_k^{c_k}(1 - \theta_k)^{N-\sum_{i=1}^kc_i}\\
\end{align}
$$

where $V$ is the total number of unique populations of variants. This product can be dramatically

We know from the definition of the the binomial distribution that the term $\theta_k^{c_k}$ is the probability of observing $c_k$ counts of variant $k$ and the term $(1-\theta_k)^{N - \sum_{i=1}^kc_i}$ is the probability of observing any other variant given given the fact that all variants with $i < k$ have already been observed in the population. That is, in symbolic terms, we know the following:

$$
\begin{align}
(1-\theta_k)^{N - \sum_{i=1}^kc_i} = \theta_{k+1}
\end{align}
$$

The counts are a function of the true proportion of DNA molecule that represents a specific variant from the population of all other DNA molecules) occurs





We might also model the distribution of each variant as a Poisson distribution, which approximates the binomial distribution in the limit of $

