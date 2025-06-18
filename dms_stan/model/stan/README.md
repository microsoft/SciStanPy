This module contains custom Stan code needed to run some functionality of DMS Stan. In this Readme, the derivations of the appropriate log-probability functions are recorded.

# Distributions
Custom distributions in DMS Stan are derived from inbuilt Stan distributions with a change of variables. We must thus calculate the adjusted probability distributions, taking into account the impact changing variables has on probability density. For univariate distributions, the change of variables formula goes by

$$
P_y(y) = P_x\left(f^{-1}(y)\right)\left|\frac{d}{dy}f^{-1}(y)\right|,
$$

where $f(x) = y$ defines the change of variables formula and $f^{-1}(y)$ describes the inverse.

In the multivariate case where we have vector $\mathbf{x}$ transformed to vector $\mathbf{y}$, the absolute value of the derivative above generalizes to the determinant of the Jacobian matrix. In this case, we thus have

$$
P_{\mathbf{y}}(\mathbf{y}) = P_{\mathbf{x}}\left(f^{-1}(\mathbf{y})\right)\left|\text{det }J_{f^{-1}}(\mathbf{y})\right|.
$$

In the following subsections, we refer to the change of variables function $f$ as the "transform" and the inverse function $f^{-1}$ as the "inverse transform".


## Exponential Dirichlet
Any vector $\mathbf{y}$ whose exponential is Dirichlet-distributed is Exponential-Dirichlet distributed. This distribution can be particularly useful when working with extremely high dimensional Dirichlet distributions--the type seen in, e.g., deep mutational scanning data--where small values can result in inefficient sampling.

The transform and inverse-transform relative to Dirichlet-distributed simplex $\mathbf{x}$ are as follows, respectively:

$$
\begin{align}
f(\mathbf{x}) &= \ln{\mathbf{x}} = \mathbf{y} \\
f^{-1}(\mathbf{y}) &= \text{e}^{\mathbf{y}} = \mathbf{x}.
\end{align}
$$

The Jacobian correction of the inverse function can be calculated as follows:

$$
\begin{align}
J_{f^{-1}} =
\begin{bmatrix}
\frac{\partial x_1}{\partial y_1} & \dotsm & \frac{\partial x_1}{\partial y_K} \\
\vdots & \ddots & \vdots \\
\frac{\partial x_K}{\partial y_1} & \dotsm & \frac{\partial x_K}{\partial y_K}
\end{bmatrix}
=\begin{bmatrix}
\text{e}^{y_1} & 0 & \dotsm & 0 \\
0 & \text{e}^{y_2} & \dotsm & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dotsm & \text{e}^{y_K}
\end{bmatrix} \\
\end{align}
$$

The determinant of a diagonal matrix is just the product of the diagonal elements, so the Jacobian adjustment is

$$
\begin{align}
\left|\text{det }J_{f^{-1}}(\mathbf{y})\right| = \prod_{k=1}^K\text{e}^{y_k}
\end{align}
$$

Putting everything together, the probability for the exponential-Dirichlet distribution is

$$
\begin{align}
P_{\mathbf{y}}(\mathbf{y} | \mathbf{\alpha}) = P_{\mathbf{x}}(\text{e}^\mathbf{y} | \mathbf{\alpha}) \prod_{k=1}^K\text{e}^{y_k},
\end{align}
$$

which, on the log scale, gives us

$$
\begin{align}
\ln{\left(P_{\mathbf{y}}(\mathbf{y} | \mathbf{\alpha})\right)} &= \ln{\left(P_{\mathbf{x}}(\text{e}^\mathbf{y} | \mathbf{\alpha}) \prod_{k=1}^K\text{e}^{y_k}\right)} \\

&= \ln{\left(\frac{1}{\Beta(\mathbf{\alpha})} \prod_{k=1}^{K} (\text{e}^{y_k})^{\alpha_k - 1}\right)} + \ln{\left(\prod_{k=1}^K\text{e}^{y_k}\right)} \\

&= \sum_{k=1}^K\ln{\left(\text{e}^{y_k(\alpha_k - 1)}\right)} - \ln{(\Beta(\mathbf{\alpha}))} + \sum_{k=1}^K \ln{\text{e}^{y_k}} \\

&= \sum_{k=1}^K y_k\alpha_k - y_k + y_k - \ln{(\Beta(\mathbf{\alpha}))} \\

&= \sum_{k=1}^K y_k\alpha_k - \ln{(\Beta(\mathbf{\alpha}))}
\end{align}
$$

Importantly, note that when $\mathbf{\alpha}$ has no prior, the term $\ln{(\Beta(\mathbf{\alpha}))}$ is constant for all samples, so can be excluded from the computation, leading to improved sampling efficiency.

# Constraint Transforms
For the [exponential-Dirichlet](#exponential-dirichlet) distribution, the vector $\mathbf{y}$ must obey the constraint $\sum_{k=1}^K \text{e}^{y_k} = 1$. This is because, for the standard Dirichlet distribution, $\mathbf{x}$ is a simplex and $x_k = \text{e}^{y_k}$.

Unfortunately, Stan does not have such a constrained data type inbuilt, and we would rather avoid the additional computation and possible stability issues that would come with simply taking the log of the inbuilt simplex constrained type. Thus, we need a custom constrained type that will produce a "log-simplex."

The following sections cover proposed inverse functions that map from an unconstrained space $\mathbf{z}$ to a constrained space $\mathbf{y}$.

## Log-Softmax

Note that the following derivation is inspired by work performed by [Modrak et al.](https://arxiv.org/pdf/2211.02383) (See Section 5).

Starting with the standard softmax function, we have

$$
x_k = \frac{\text{e}^{z_k}}{\sum_{k'=1}^K\text{e}^{z_{k'}}},
$$

which forces

$$
\sum_{k=1}^K x_k = 1.
$$

Thus, if, as we do for the derivation of the [exponential-Dirichlet](#exponential-dirichlet), we define $\mathbf{x} = \text{e}^\mathbf{y}$, we enforce the constraint $\sum_{k=1}^K \text{e}^{y_k} = 1$ and obtain the below:

$$
\text{e}^{y_k} = \frac{\text{e}^{z_k}}{\sum_{k'=1}^K\text{e}^{z_{k'}}}.
$$

Solving for $y_k$ gives us

$$
\begin{align}
y_k &= \ln{\left(\frac{\text{e}^{z_k}}{\sum_{k'=1}^K\text{e}^{z_{k'}}}\right)} \\

&= \ln(\text{e}^{z_k}) - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\right)} \\

&= z_k - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\right)}.
\end{align}
$$

So we can use the above equation to transform from our unconstrained space $\mathbf{z}$ to the constrained space $\mathbf{y}$ over which the exponential-Dirichlet distribution is defined.

There is a problem, however, with the above transformation, particularly as it applies to calculating the Jacobian correction: as defined, this is not an invertible function, as any constant shift to $z$ results in an equivalent output:

$$
\begin{align}
y_k' &= z_k + C - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'} + C}\right)} \\

&= z_k + C - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\text{e}^{C}\right)} \\

&= z_k + C - \ln{\left(\text{e}^{C}\sum_{k'=1}^K\text{e}^{z_{k'}}\right)} \\

&= z_k + C - \ln {\text{e}^{C}} - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\right)} \\

&= z_k + C - C - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\right)} \\

&= z_k - \ln{\left(\sum_{k'=1}^K\text{e}^{z_{k'}}\right)} \\

&= y_k \\
\end{align}
$$

This identifiability problem can be fixed by forcing the value of one element of $\mathbf{z}$ to be 0. This can be any position in the vector, but we choose the last for simplicity ($z_K = 0$):

$$
\begin{align}
y_k &= z_k - \ln{\left(\text{e}^0 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)} \\

&= z_k - \ln{\left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)} \\
\end{align}
$$

Adding a constant term to all $z_k$, $k < K$ will now yield a different solution, as we have no way of factoring out the resulting scaling factor of the sum from the log.

Now that we have the inverse transformation set, we can work on the Jacobian adjustment. Note that, because we have the constraint $\sum_{k=1}^K \text{e}^{y_k} = 1$, it follows that any distribution over $\mathbf{y}$ can be written in terms of the first $K - 1$ elements. This is because we can write $y_K$ in terms of the other vector components:

$$
\begin{align}
\sum_{k=1}^K \text{e}^{y_k} &= 1 \\

\text{e}^{y_K} + \sum_{k=1}^{K-1} \text{e}^{y_k} &= 1 \\

\text{e}^{y_K} &= 1 - \sum_{k=1}^{K-1} \text{e}^{y_k} \\

y_K &= \ln{\left(1 - \sum_{k=1}^{K-1} \text{e}^{y_k}\right)}.
\end{align}
$$

As a result, $y_K$ does not factor into our Jacobian adjustment calculation--all probability distributions are fully parametrized by the first $K-1$ elements, so our Jacobian is a $K-1$ by $K-1$ square matrix considering the inverse transformation $g^{-1}(\mathbf{z}) = \mathbf{y}$, where both $\mathbf{y}$ and $\mathbf{z}$ have length $K-1$ and for a given $y_k$ we have

$$
y_k = z_k - \ln{\left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)}.
$$

Now to derive this Jacobian:

$$
\begin{align}
\frac{\partial y_k}{\partial z_k} &= 1 - \left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)^{-1}\text{e}^{z_k} \\

\frac{\partial y_k}{\partial z_{j \ne k}} &= -\left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)^{-1}\text{e}^{z_j} \\
\end{align}
$$

For ease of notation, we will define $s = - \left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)^{-1}$, which gives us


$$
\begin{align}
\frac{\partial y_k}{\partial z_k} &= 1 + s\text{e}^{z_k} \\

\frac{\partial y_k}{\partial z_{j \ne k}} &= s\text{e}^{z_j} \\

J_{g^{-1}} &=
\begin{bmatrix}
1 + s\text{e}^{z_1} & s\text{e}^{z_1} & \dotsm & s\text{e}^{z_1} \\
s\text{e}^{z_2} & 1 + s\text{e}^{z_2} & \dotsm & s\text{e}^{z_2} \\
\vdots & \vdots & \ddots &\vdots \\
s\text{e}^{z_{K-1}} & s\text{e}^{z_{K-1}} & \dotsm & 1 + s\text{e}^{z_{K-1}}
\end{bmatrix}.
\end{align}
$$

The Jacobian can be factored into a column vector multiplied by a row vector added to the identity matrix:

$$
\begin{align}
J_{g^{-1}} = I + \mathbf{c}\mathbf{r},
\end{align}
$$

with

$$
\begin{align}
\mathbf{c} &=
    \begin{bmatrix}
    \text{e}^{z_1} \\
    \text{e}^{z_2} \\
    \vdots \\
    \text{e}^{z_{K-1}} \\
    \end{bmatrix}\\

\mathbf{r} &=
    \begin{bmatrix}
    s & s & \dotsm & s
    \end{bmatrix}
    .
\end{align}
$$

By the Matrix Determinant Lemma, we know that

$$
\text{det}(J_{g^{-1}}) = \text{det}(I + \mathbf{c}\mathbf{r}) = 1 + \mathbf{r}\mathbf{c}
$$

which can be expanded to

$$
\begin{align}
    \text{det}(J_{g^{-1}}) &= 1 + \mathbf{r}\mathbf{c} \\

    &= 1 + \sum_{k=1}^{K-1} s\text{e}^{z_k} \\

    &= 1 - \sum_{k=1}^{K-1} \frac{\text{e}^{z_k}}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}}. \\
\end{align}
$$

The summation in the term above is the sum over the first $K-1$ outputs from the softmax function where we set $z_K = 1$. We know that

$$
\frac{1}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}} +\sum_{k=1}^{K-1} \frac{\text{e}^{z_k}}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}}  = 1
$$

and, consequently, that

$$
\frac{1}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}} = 1 - \sum_{k=1}^{K-1} \frac{\text{e}^{z_k}}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}},
$$

which gives us

$$
\text{det}(J_{g^{-1}}) = \frac{1}{1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}}.
$$

In log form, we have

$$
\ln{\left(\text{det}(J_{g^{-1}})\right)} = -\ln{\left(1 +\sum_{k'=1}^{K - 1}\text{e}^{z_{k'}}\right)} = y_K.
$$

## Stick Breaking (in progress)

Because $\mathbf{x}$ is a simplex, $\mathbf{y}$ must obey the constraint $\sum_{k=1}^K \text{e}^{y_i} = 1$. Unfortunately, Stan does not have such a constrained data type inbuilt, and we cannot simply take the log of the inbuilt simplex constrained type, as passing through this intermediate linear space defeats the point of operating in the log space in the first place. We thus need to calculate the modified probability density function starting from unconstrained space.

Assume, for a moment, however, that we do want to pass through the intermediate linear space. This would proceed through a series of inverse transforms as follows:

$$
\begin{align}
\mathbf{x} &= \text{e}^{\mathbf{y}} = f^{-1}(\mathbf{y}) \\
\mathbf{y} &= \ln{\mathbf{z}} = g^{-1}(\mathbf{z}) \\
\mathbf{z} &= h^{-1}(\mathbf{a}),
\end{align}
$$

where $h^{-1}(\mathbf{z})$ is some function that takes unconstrained vector $\mathbf{a}$ and converts it into simplex $\mathbf{z}$. To avoid moving through the linear space, we want to find some mapping that goes directly from $\mathbf{a}$ to $\mathbf{y}$, avoiding any direct calculation of $\mathbf{z}$. In other words, we want to write all of our equations in terms of $\ln{\mathbf{z}}$

For $h^{-1}(\mathbf{a})$, we will use the [stick breaking algorithm](https://mc-stan.org/docs/reference-manual/transforms.html#simplex-transform.section) that produces the simplex constrained type within Stan. This algorithm takes the following form:

$$
\begin{align}
h^{-1}(\mathbf{a}) &=
\begin{cases}
\left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)j(a_k) & \text{if } k < K \\
1 - \sum_{k'=1}^{K-1}z_{k'} & \text{if } k =K \\
\end{cases},
\end{align}
$$

where $K$ is the length of $\mathbf{z}$ and $\mathbf{a}$ has length $K-1$. $\mathbf{a}$ has one less element than $\mathbf{z}$ because the final element of $\mathbf{z}$ is fully determined by its first $K-1$ elements.

$$
\begin{align}
j(a_k) &= \text{logit}^{-1}(a_k - \ln({K - k})) \\
&= \left(1 + \text{e}^{-a_k + \ln({K-k})}\right)^{-1} \\
&= \left(1 + (K -k)\text{e}^{-a_k}\right)^{-1}.
\end{align}
$$

Putting all of the above together into the multivariate change of variables formula, we get the following for modeling the probability of our exponential Dirichlet in terms of unconstrained vector $\mathbf{a}$:

$$
\begin{align}
P_{\mathbf{a}}(\mathbf{a}) = P_{\mathbf{x}}\left(f^{-1}(g^{-1}(h^{-1}(\mathbf{a})))\right)\left|\text{det }J_{f^{-1}}(\mathbf{a})\right|.
\end{align}
$$

Because Stan works with log probabilities, what we actually want to solve for is the following:

$$
\begin{align}
\ln{\left(P_{\mathbf{a}}(\mathbf{a})\right)} = \ln{\left(P_{\mathbf{x}}\left(f^{-1}(g^{-1}(h^{-1}(\mathbf{a})))\right)\right)} + \ln{\left(\left|\text{det }J_{f^{-1}}(\mathbf{a})\right|\right)}.
\end{align}
$$

We'll run through the derivations of each additive component independently. First, the easier one. To begin, we cancel variables where possible and make some substitutions:

$$
\begin{align}
\ln{\left(P_{\mathbf{x}}\left(f^{-1}(g^{-1}(h^{-1}(\mathbf{a})))\right)\right)} &= \ln{\left(P_{\mathbf{x}}\left(\text{e}^{\ln(h^{-1}(\mathbf{a}))}\right)\right)} \\
&= \ln{\left(P_{\mathbf{x}}\left(h^{-1}(\mathbf{a})\right)\right)} \\
&= \ln{\left(\frac{1}{\Beta(\boldsymbol{\alpha})}\prod_{k=1}^K z_k^{\alpha_k -1}\right)} \\
&= -\ln{\left(\Beta(\boldsymbol{\alpha})\right)} +\sum_{k=1}^K\ln{\left(z_k^{\alpha_k -1}\right)} \\
&= -\ln{\left(\Beta(\boldsymbol{\alpha})\right)} +\sum_{k=1}^K(\alpha_k -1)\ln{z_k}
\end{align}
$$

Remembering that we defined $\mathbf{y} = \ln{\mathbf{z}}$, the above can also be written as

$$
\ln{\left(P_{\mathbf{x}}\left(f^{-1}(g^{-1}(h^{-1}(\mathbf{a})))\right)\right)} = -\ln{\left(\Beta(\boldsymbol{\alpha})\right)} +\sum_{k=1}^K(\alpha_k -1)y_k
$$

By unraveling the calculating of $\mathbf{y}$, we can see that we do not need to go through a linear intermediate space to calculate the above:

$$
\begin{align}
y_1 &= \ln{z_1} = \ln{\left(j(a_1)\right)} \\
y_2 &= \ln{z_2} = \ln{(1 - j(a_1))} + \ln{(j(a_2))} \\
y_3 &= \ln{z_3} = \ln{(1 - j(a_1) - j(a_2)) } + \ln{(j(a_3))} \\
&\dots \\
y_
{K-1} &= \ln{z_{K-1}} = \ln{\left(1 - \sum_{k'=1}^{K-2}j(a_{k'})\right) } + \ln{(j(a_{K-1}))} \\
y_K &= \ln{z_{K}} = \ln{\left(1 - \sum_{k'=1}^{K-1}j(a_{k'})\right) }\\
\end{align}
$$

Note that the log of $j(a_k)$ can be written as:

$$
\begin{align}
\ln{j(a_k)} &= -\ln{\left(1 + (K -k)\text{e}^{-a_k}\right)} \\
&= -\ln{\left((K-k)\left(\frac{1}{K-k} + \text{e}^{-a_k}\right)\right)} \\
&= \ln{(K-k)} - \ln{\left(\frac{1}{K-k} + \text{e}^{-a_k}\right)} \\
&= \ln{(K-k)} - \text{logSumExp}(-\ln{(K-k)}, -a_k),
\end{align}
$$

which will be numerically stable.

Now for the Jacobian adjustment portion of the log probability. All partials making up the Jacobian matrix can be expanded via the chain rule to give

$$
\begin{align}
\frac{\partial x}{\partial a} &= \frac{\partial x}{\partial y}\frac{\partial y}{\partial z}\frac{\partial z}{\partial a} \\
&= \text{e}^y\frac{1}{z}\frac{\partial z}{\partial a}.
\end{align}
$$

And, because $y = \ln{z}$, we also have $\text{e}^y = z$, so


$$
\begin{align}
\frac{\partial x}{\partial a} &= \text{e}^y\frac{1}{\text{e}^y}\frac{\partial z}{\partial a} \\
\frac{\partial x}{\partial a} &= \frac{\partial z}{\partial a}.
\end{align}
$$

For the diagonal elements, we have

$$
\begin{align}
\frac{\partial z_k}{\partial a_k} &= \left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)\frac{\partial}{\partial a_k}j(a_k) \\
&= \left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)(-1)\left(1 + (K -k)\text{e}^{-a_k}\right)^{-2}(K-k)\text{e}^{-a_k}(-1) \\
&= \left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)\left(1 + (K -k)\text{e}^{-a_k}\right)^{-2}(K-k)\text{e}^{-a_k}
\end{align}
$$

For the upper-triangular part of the Jacobian matrix, we have

$$
\begin{align}
\frac{\partial z_k}{\partial a_{k'' > k}} = 0,
\end{align}
$$

as $z_k$ only has dependence on $\mathbf{a}$ for $k' <= k$. This means that the determinant of the Jacobian is just the product of the diagonal elements:

$$
\left|\text{det }J_{f^{-1}}(\mathbf{a})\right| = \prod_{k=1}^{K-1}\frac{\partial z_k}{\partial a_k}.
$$

Because the Jacobian correction will be in log space, we can instead write the above as a sum:

$$
\ln{\left(\left|\text{det }J_{f^{-1}}(\mathbf{a})\right|\right)} = \sum_{k=1}^{K-1}\ln{\frac{\partial z_k}{\partial a_k}}.
$$

If we expand out $\ln{\frac{\partial z_k}{\partial a_k}}$ as

$$
\begin{align}
\ln{\frac{\partial z_1}{\partial a_1}} &= \ln{\left(\left(1 + (K - 1)\text{e}^{-a_1}\right)^{-2}(K-1)\text{e}^{-a_1}\right)} \\
&= -2\ln{\left(1 + (K - 1)\text{e}^{-a_1}\right)} + \ln{(K-1)} + \ln{\text{e}^{-a_1}} \\
&= \ln{(K-1)} -2\ln{\left(1 + (K - 1)\text{e}^{-a_1}\right)} - a_1 \\
&= \ln{(K-1)} -2\ln{\left((K-1)((K - 1)^{-1} + \text{e}^{-a_1})\right)} - a_1 \\
&= \ln{(K-1)} -2\left(\ln{(K-1)} + \ln{\left((K - 1)^{-1} + \text{e}^{-a_1}\right)}\right) - a_1 \\
&= \ln{(K-1)} -2\ln{(K-1)} - 2\ln{\left((K - 1)^{-1} + \text{e}^{-a_1}\right)} - a_1 \\
&= -\ln{(K-1)} - 2\ln{\left((K - 1)^{-1} + \text{e}^{-a_1}\right)} - a_1 \\
&= -\ln{(K-1)} - 2\ln{\left((K - 1)^{-1} + \text{e}^{-a_1}\right)} - a_1 \\
\ln{\frac{\partial z_2}{\partial a_2}} &= \ln{(1 - z_1)}-\ln{(K-2)} - 2\ln{\left((K - 2)^{-1} + \text{e}^{-a_2}\right)} - a_2 \\
&\dots \\
\ln{\frac{\partial z_{k}}{\partial a_{k}}} &= \ln{\left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)} -\ln{(K-k)} - 2\ln{\left((K - k)^{-1} + \text{e}^{-a_k}\right)} - a_k, \\
\end{align}
$$

which we can also see allows us to stay on the log scale by writing the calculation as

$$
\begin{align}
\ln{\frac{\partial z_{k}}{\partial a_{k}}} &= \ln{\left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)}-\ln{(K-k)} - 2\text{logSumExp}\left(\ln{\left((K-k)^{-1}\right)}, -a_k\right) - a_k \\
&= \ln{\left(1 - \sum_{k' = 1}^{k-1}z_{k'}\right)} - \ln{(K-k)} - 2\text{logSumExp}\left(-\ln{\left(K-k\right)}, -a_k\right) - a_k
\end{align}
$$

Now, putting it all together, we can model the log-probability of the Dirichlet distribution without leaving the log scale as follows:

$$
\begin{align}
\ln{\left(P_{\mathbf{a}}(\mathbf{a})\right)} &= \ln{\left(P_{\mathbf{x}}\left(f^{-1}(g^{-1}(h^{-1}(\mathbf{a})))\right)\right)} + \ln{\left(\left|\text{det }J_{f^{-1}}(\mathbf{a})\right|\right)} \\
&= -\ln{\left(\Beta(\boldsymbol{\alpha})\right)} + \sum_{k=1}^K(\alpha_k -1)y_k + \sum_{k=1}^{K-1}\ln{\frac{\partial z_k}{\partial a_k}}
\end{align}
$$

# Reparametrizations
Some distributions in DMS Stan are reparametrizations of others. These do not involve transforming the space over which probability density is defined, so we do not need Jacobian corrections.

## Multinomial, Log-Theta Parametrization

The PDF for the multinomial distribution is given as

$$
P(\mathbf{n} | \mathbf{\theta}) = \frac{\left(\sum_{k=1}^K n_k\right)!}{\prod_{k=1}^Kn_k!}\prod_{k=1}^K\theta_k^{n_k}
$$

In many instances, we wish to stay on the log scale when modeling using the multinomial distribution. That is, we want to parametrize in terms of $\mathbf{\Theta} = \ln{\mathbf{\theta}}$. Making the appropriate substitutions gives us

$$
\begin{align}
    P(\mathbf{n} | \mathbf{\Theta}) &= \frac{\left(\sum_{k=1}^K n_k\right)!}{\prod_{k=1}^Kn_k!}\prod_{k=1}^K\text{e}^{\Theta_k^{n_k}} \\

    &= \frac{\left(\sum_{k=1}^K n_k\right)!}{\prod_{k=1}^Kn_k!}\prod_{k=1}^K\text{e}^{n_k\Theta_k}
\end{align}
$$

The log probability thus becomes


$$
\begin{align}
    \ln{\left(P(\mathbf{n} | \mathbf{\Theta})\right)} &= \ln{\left(\left(\sum_{k=1}^K n_k\right)!\right)} - \ln{\prod_{k=1}^Kn_k!} + \ln{\prod_{k=1}^K\text{e}^{n_k\Theta_k}} \\

    &= \sum_{i=1}^{\sum_{k=1}^K n_k}\ln{i} - \sum_{k=1}^K\ln{n_k!} + \sum_{k=1}^K\ln{\left(\text{e}^{n_k\Theta_k}\right)} \\

    &= \sum_{i=1}^{\sum_{k=1}^K n_k}\ln{i} - \sum_{k=1}^K\sum_{j=1}^{n_k}\ln{j} + \sum_{k=1}^Kn_k\Theta_k \\
\end{align}
$$

Notably, when the observations, $\mathbf{n}$ are constant, as is often the case for multinomial distributions which are used to model observed count data, we can precalculate the sums of logs and store them as "transformed data" attributes for use during log-probability calculations.