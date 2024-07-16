Growth Models
=============
This file describes the different models of organismal growth that are included in DMS-Stan.

## Competitive Lotka-Volterra
The base growth model that we start from is the [Lotka-Volterra equation for competitive growth](https://journals.biologists.com/jeb/article/9/4/389/22702/Experimental-Studies-on-the-Struggle-for):

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \frac{\sum_{j = 1}^{N}\alpha_{ij}x_j}{K_i}\right),
\end{align}
$$

which is a differential equation that describes the rate of change in a population of species $i$ with size $x_i$ given their base rate of change ($r_i$), their carrying capacity ($K_i$), the current population sizes of all competing species ($x_j$) including competition amongst the population itself, and a coefficient that describes the level of impact species $j$ has on species $i$ ($\alpha_{ij}$).

Notably, as applied in DMS Stan where we parametrize our models such that we are working with relative proportions of variants and not absolute abundances, the carrying capacity for all variants must be the same (i.e., when they are the only variant in the population). Thus, all $K_i$ must be equal and so we will ignore it as a scaling factor (an alternative way to think about it is that it is implicitly encoded in all $\alpha_{ij}$). The form of the Lotka-Volterra equation used in DMS Stan is thus

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \sum_{j = 1}^{N}\alpha_{ij}x_j\right).
\end{align}
$$

This generalized equation can be used to describe the relative abundances of different competing populations. The remainder of this document is dedicated to showing how different assumptions can result in various alternate parametrizations.

## Adjusted Exponential
Let's assume that each population of variants imparts equivalent stress on the others. In other words, we will have $\alpha_{ij} = \alpha$ for all $i$ and all $j$. In this case, Eq. 2 simplifies to

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \alpha\sum_{j = 1}^Nx_j\right).
\end{align}
$$

Again, because DMS Stan works with relative proportions of variants, not absolute counts. The total sum of the proportions of all variants ($\sum_{j = 1}^Nx_j$) must thus be constant. This constant can be captured in the $\alpha$ parameter itself, allowing us to simplify to

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i (1 - \alpha),
\end{align}
$$

which can in turn be reparametrized as

$$
\begin{align}
\frac{dx_i}{dt} = \beta r_ix_i,
\end{align}
$$

This is just the exponential growth curve modified to include some global adjustment to growth rates $\beta$. In this model, each variant has a basal growth rate ($r_i$) that is influenced by the relative abundance of all other variants (implictly encoded by $x_i$) with strength $\beta$.

In actuality, we do not need the term $\beta$, as it is just a scaling factor. Indeed, if we were to include it, we would have unidentifiable models, where any values of $r$ would be allowed so long as their relative proportions stayed constant. Thus, in DMS Stan, we set $\beta = 1$ and simply model the exponential growth curve (with the assumption that the scale of $r$ accounts for global interactions among variants):

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i,
\end{align}
$$

The integral of this differential equation is trivial to solve:

$$
\begin{align}
\frac{1}{x_i}dx_i &= r_i dt \\
\int \frac{1}{x_i}dx_i &= \int r_i dt \\
\ln{x_i} + C_{1_i} &= r_it + C_{2_i}
\end{align}
$$

where $C_{n_i}$ is some unknown constant. We can group these constants together to get our final equation of

$$
\begin{align}
\ln{x_i} &= r_it + C_i
\end{align}
$$

When performing our regression in Stan, we will keep this in logarithmic form for the sake of numerical stability. However, by expanding it we can see that this is just the equation for unbounded exponential growth:

$$
\begin{align}
x_i &= \textrm{e} ^ {r_it + C_i} \\
&= \textrm{e} ^ {r_it}\textrm{e} ^ {C_i} \\
&= A_i\textrm{e} ^ {r_it}, \\
\end{align}
$$

where we are defining $A_i = \textrm{e}^{C_i}$ as a scaling constant.

## Modified Sigmoid Growth
It might not necessarily be reasonable to assume that all variants have equivalent impact on one another. Indeed, we may want to assume that each population of variants imparts a different-strength effect on the others. In this case, we can write Eq. 2 as

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \sum_{j = 1}^{N}\alpha_{j}x_j\right),
\end{align}
$$

where each variant has a distinct parameter for $\alpha$ that describes the strength of the impact of its population on both its own and other populations.

This integral is more tricky to solve, but it can be done. First, we need to isolate $x_i$ from the summation:

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - a_ix_i - \sum_{j \ne i}\alpha_{j}x_j\right).
\end{align}
$$

The summation is no longer dependent on $x_i$, so we will treat it as a constant by substituting $\omega_i = 1 - \sum_{j \ne i}\alpha_{j}x_j$:

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i (\omega_i - a_ix_i).
\end{align}
$$

Now let's rearrange to get the above into an integratable form:

$$
\begin{align}
r_idt &= \frac{1}{x_i (\omega_i - \alpha_i x_i )}dx_i \\
r_idt&=\left(\frac{1}{x_i (\omega_i - \alpha_i x_i)}\right)\left(\frac{-\alpha_i^{-1}}{-\alpha_i^{-1}}\right)dx_i \\
r_idt&=-\frac{\alpha_i^{-1}}{x_i (x_i - \alpha_i^{-1}\omega_i)}dx_i \\
\end{align}
$$

For simplicity going forward, we reparametrize such that $\delta_i =\alpha_i^{-1}$ to get


$$
\begin{align}
-r_idt&=\frac{\delta_i}{x_i (x_i - \delta_i\omega_i)}dx_i \\
\end{align}
$$

The above can be integrated by partial fractions if we do some rearranging:

$$
\begin{align}
-r_idt&=\left(\frac{\delta_i}{x_i (x_i - \delta_i\omega_i)}\right)\left(\frac{\omega_i}{\omega_i}\right)dx_i \\
&=\frac{1}{\omega_i}\left(\frac{\delta_i\omega_i}{x_i (x_i - \delta_i\omega_i)}\right)dx_i \\
&=\frac{1}{\omega_i}\left(\frac{\delta_i\omega_i + x_i - x_i}{x_i (x_i - \delta_i\omega_i)}\right)dx_i \\
&=\frac{1}{\omega_i}\left(\frac{\delta_i\omega_i - x_i}{x_i (x_i - \delta_i\omega_i)} + \frac{x_i}{x_i (x_i - \delta_i\omega_i)}\right)dx_i \\
&=\frac{1}{\omega_i}\left(\frac{x_i}{x_i (x_i - \delta_i\omega_i)} - \frac{x_i - \delta_i\omega_i}{x_i (x_i - \delta_i\omega_i)}\right)dx_i \\
-r_i\omega_i dt &= \left(\frac{1}{x_i - \delta_i\omega_i} - \frac{1}{x_i}\right)dx_i
\end{align}
$$

Now we can integrate:

$$
\begin{align}
-\int r_i\omega_i dt &= \int\frac{1}{x_i - \delta_i\omega_i} dx_i - \int\frac{1}{x_i}dx_i\\
-r_i\omega_i t + C_{1_i} &= \ln{\left(x_i - \delta_i\omega_i\right)} + C_{2_i} - \ln{(x_i)} + C_{3_i}
\end{align}
$$

As with earlier, the $C_{n_i}$ terms are unknown constants that can be grouped into a single $C_i$. Doing this, then rearranging to solve for $x_i$ gives us:

$$
\begin{align}
\ln{\left(\frac{x_i}{x_i - \delta_i\omega_i}\right)} &= r_i \omega_i t + C_i  \\
\frac{x_i}{x_i - \delta_i\omega_i} &= \textrm{e}^{r_i \omega_i t + C_i}  \\
x_i &= x_i\textrm{e}^{r_i \omega_i t + C_i} - \delta_i\omega_i\textrm{e}^{r_i \omega_i t + C_i} \\
x_i\left(1 - \textrm{e}^{r_i \omega_i t + C_i}\right) &= - \delta_i\omega_i\textrm{e}^{r_i \omega_i t + C_i} \\
x_i &= - \frac{\delta_i\omega_i\textrm{e}^{r_i \omega_i t + C_i}}{1 - \textrm{e}^{r_i \omega_i t + C_i}} \\
x_i &= - \frac{\delta_i\omega_i\textrm{e}^{r_i \omega_i t + C_i}}{1 - \textrm{e}^{r_i \omega_i t + C_i}} \\
\end{align}
$$

We can further simplify to the below:

$$
\begin{align}
x_i &= - \frac{\delta_i\omega_i}{\textrm{e}^{-r_i \omega_i t + C_i} - 1} \\
x_i &= \frac{\delta_i\omega_i}{1 - \textrm{e}^{-r_i \omega_i t + C_i}} \\
\end{align}
$$

## Sigmoid Growth

Notably, if we assume that there is no interaction between different variants, then $a_j = 0$ for all $j \ne i$ and the $w_i$ term in Eq. 36 goes to $1$. In this case, the model reduces to the standard logistic growth curve:

$$
\begin{align}
x_i &= \frac{\delta_i}{1 - \textrm{e}^{-r_i t + C_i}}
\end{align}
$$
