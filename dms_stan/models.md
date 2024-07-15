Growth Models
=============
This file describes the different models that are included in DMS-Stan.

## Competitive Lotka-Volterra
The base growth model that we start from is the [Lotka-Volterra equation for competitive growth](https://journals.biologists.com/jeb/article/9/4/389/22702/Experimental-Studies-on-the-Struggle-for):

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \frac{\sum_{j = 1}^{N}\alpha_{ij}x_j}{K_i}\right),
\end{align}
$$

which is a differential equation that describes the rate of change in a population of species $i$ with size $x_i$ given their base rate of change ($r_i$), their carrying capacity ($K_i$), the current population sizes of all competing species ($x_j$) including competition amongst the population itself, and a coefficient that describes the level of impact species $j$ has on species $i$ ($\alpha_{ij}$).

Notably, as applied in DMS Stan where we parametrize our models such that we are working with relative proportions of variants and not absolute abundances, the carrying capacity for all variants will be 1, as this is when they dominate the population. Thus, our form of the Lotka-Volterra equation is

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \sum_{j = 1}^{N}\alpha_{ij}x_j\right),
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

Again, because DMS Stan works with relative proportions of variants, not absolute counts. The total sum of the proportions of all variants ($\sum_{j = 1}^Nx_j$) must thus be equal to 1. This gives us

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i (1 - \alpha),
\end{align}
$$

which we will parametrize as

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
\ln{x_i} + C_1 &= r_it + C_2
\end{align}
$$

where $C_n$ is some unknown constant. We can group these constants together to get our final equation of

$$
\begin{align}
\ln{x_i} &= r_it + C
\end{align}
$$

When performing our regression in Stan, we will keep this in logarithmic form for the sake of numerical stability. However, expanding it we can see that this is just the equation for unbounded exponential growth:

$$
\begin{align}
x_i &= \textrm{e} ^ {r_it + C} \\
&= \textrm{e} ^ {r_it}\textrm{e} ^ {C} \\
&= A\textrm{e} ^ {r_it}, \\
\end{align}
$$

where we are defining $A = \textrm{e}^C$ as a scaling constant.

## Modified Sigmoid Growth
It might not necessarily be reasonable to assume that all variants have equivalent impact on one another. Indeed, we may want to assume that each population of variants imparts a different-strength effect on the others. In this case, we can write Eq. 2 as

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \sum_{j = 1}^{N}\alpha_{j}x_j\right),
\end{align}
$$

where each variant has a distinct parameter for $\alpha$ that describes the strength of the impact of its population on both its own and other populations. Because $0 \le x_j \le 1$, the model is identifiable, as rescaling all $\alpha_j$ will result in a different summation.

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
-r_i\omega_i t + C_1 &= \ln{\left(x_i - \delta_i\omega_i\right)} + C_2 - \ln{(x_i)} + C_3
\end{align}
$$

As with earlier, the $C_{k}$ terms are unknown constants that can be grouped into a single $C$. Doing this, then rearranging to solve for $x_i$ gives us:

$$
\begin{align}
\ln{\left(\frac{x_i}{x_i - \delta_i\omega_i}\right)} &= r_i \omega_i t + C  \\
\frac{x_i}{x_i - \delta_i\omega_i} &= \textrm{e}^{r_i \omega_i t + C}  \\
x_i &= x_i\textrm{e}^{r_i \omega_i t + C} - \delta_i\omega_i\textrm{e}^{r_i \omega_i t + C} \\
x_i\left(1 - \textrm{e}^{r_i \omega_i t + C}\right) &= - \delta_i\omega_i\textrm{e}^{r_i \omega_i t + C} \\
x_i &= - \frac{\delta_i\omega_i\textrm{e}^{r_i \omega_i t + C}}{1 - \textrm{e}^{r_i \omega_i t + C}} \\
x_i &= - \frac{\delta_i\omega_i\textrm{e}^{r_i \omega_i t + C}}{1 - \textrm{e}^{r_i \omega_i t + C}} \\
\end{align}
$$

We can further simplify to the below:

$$
\begin{align}
x_i &= - \frac{\delta_i\omega_i}{\textrm{e}^{-r_i \omega_i t + C} - 1} \\
x_i &= \frac{\delta_i\omega_i}{1 - \textrm{e}^{-r_i \omega_i t + C}} \\
\end{align}
$$

## Simplified Sigmoid Growth

Notably, if we assume that there is no interaction between different variants, then $a_j = 0$ for all $j \ne i$ and the $w_i$ term in Eq. 36 goes to $1$. In this case, the model reduces to the standard logistic growth curve:

$$
\begin{align}
x_i &= \frac{\delta_i}{1 - \textrm{e}^{-r_i t + C}}
\end{align}
$$

# OLD

It is likely unreasonable to assume that variants do not interact with one another. We can add some interaction between variants by making the slightly more realistic assumption that variants from the same population interact with one another but not with variants from other populations. To represent this mathematically, we set $\alpha_{j = i} = 1$ and $\alpha_{j \ne i} = 0$. Note that we could set $\alpha_{j = i}$ to any value that we want as the $K_i$ term in the denominator of the term $\frac{\sum_{j = 1}^{N}\alpha_{ij}x_{j}}{K_i}$ acts as a scaling factor. To keep our models identifiable as well as to simplify our analytical solutions, setting $\alpha_{j = i} = 1$ is sensible.

With the above-described substitutions in place, we get

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \frac{\sum_{j = i}x_j}{K_i}\right),
\end{align}
$$

and because there is only one instance of $j=i$ for a given variant $i$, we can simplify further to

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \frac{x_i}{K_i}\right).
\end{align}
$$

Now solving for $x_i$:

$$
\begin{align}
\frac{1}{x_i(1 - x_i / K_i)}dx_i &= r_idt \\
\frac{K_i}{x_i(K_i - x_i)}dx_i &= r_idt \\
\frac{K_i + x_i - x_i}{x_i(K_i - x_i)}dx_i &= r_idt \\
\left(\frac{K_i - x_i}{x_i(K_i - x_i)} + \frac{x_i}{x_i(K_i - x_i)}\right)dx_i &= r_idt \\
\left(\frac{1}{x_i} + \frac{1}{K_i - x_i}\right)dx_i &= r_idt \\
\int \frac{1}{x_i} dx_i + \int\frac{1}{K_i - x_i}dx_i &= \int r_idt \\
\ln x_i + C_1 - \ln{\left(K_i - x_i\right)} + C_2 &= r_i t + C_3\\
\ln x_i - \ln{\left(K_i - x_i\right)} &= r_i t + C\\
\ln{\left(\frac{x_i}{K_i - x_i}\right)} &= r_i t + C\\
\frac{x_i}{K_i - x_i} &= \textrm{e}^{r_i t + C}\\
x_i &= K_i\textrm{e}^{r_i t + C} - x_i\textrm{e}^{r_i t + C}\\
x_i(1 + \textrm{e}^{r_i t + C}) &= K_i\textrm{e}^{r_i t + C}\\
x_i &= K_i\frac{\textrm{e}^{r_i t + C}}{1 + \textrm{e}^{r_i t + C}}\\
x_i &= \frac{K_i}{1 + \textrm{e}^{-r_i t - C}}\\
x_i &= \frac{K_i}{1 + A\textrm{e}^{-r_i t}}\\
\end{align}
$$

This is, of course, the equation for logistic growth. We can put it in log form as below:

$$
\begin{align}
\ln{x_i} &= \ln{K_i} - \ln{\left(1 + A\textrm{e}^{-r_i t}\right)}\\
\end{align}
$$



# Simplified Lotka-Volterra



# OLD






Before that, however, we note that, in DMS Stan, we are not looking at the number of members in a population but rather the relative ratios of population members. That is, the maximum value (or carrying capacity) for $x_i$ is $1$. Thus, we can set $K_i = 1$ to simplify Eq. 1 to

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \sum_{j = 1}^{N}\alpha_{ij}x_j\right).
\end{align}
$$

We will use this form of the equation as our basis for the following sections.



To ensure identifiability of these models, we must set one of the values of $\alpha_{ij}$ to be fixed; otherwise, we can have arbitrary scales of $\alpha_{ij}$ adjusted by $K_i$. It will be most convenient for us to simply set $\alpha_{i=j} = 1$, in which case we can rearrange the equation to

$$
\begin{align}
\frac{dx_i}{dt} = r_ix_i \left(1 - \frac{x_i + \sum_{j \ne i}^{N}\alpha_{ij}x_j}{K_i}\right).
\end{align}
$$

Because we want to fit a regression model in DMS-Stan, the differential form of this equation is not the most useful to us, so we solve Eq. 2 to get $x_i$ as a function of $t$. To begin, we will substitute $A = \frac{\sum_{j \ne i}^{N}\alpha_{ij}x_j}{K_i}$ as these terms are constants:

$$
\begin{align}
\frac{dx_i}{dt} &= r_ix_i \left(1 - \frac{x_i}{K_i} - A\right)\\
\end{align}
$$

Now let's rearrange to get the above into an integratable form:

$$
\begin{align}
dx_i &= r_ix_i \left(1 - \frac{x_i}{K_i} - A\right)dt\\
r_idt &= \frac{1}{x_i \left(1 - \frac{x_i}{K_i} - A\right)}dx_i \\
r_idt&=\left(\frac{1}{x_i \left(1 - \frac{x_i}{K_i} - A\right)}\right)\left(\frac{-K_i}{-K_i}\right)dx_i \\
r_idt&=-\frac{K_i}{x_i \left(-K_i + x_i + AK_i\right)}dx_i \\
-r_idt&=\frac{K_i}{x_i \left(x_i + K_i(A - 1)\right)}dx_i
\end{align}
$$

The above can be integrated by partial fractions if we make some simple modifications:

$$
\begin{align}
-r_idt&=\left(\frac{K_i}{x_i \left(x_i + K_i(A - 1)\right)}\right)\left(\frac{A-1}{A-1}\right)dx_i \\
&=\frac{K_i(A - 1) + (x_i + K_i(A-1)) - (x_i + K_i(A-1))}{x_i(A-1)(x_i + K_i(A - 1))} dx_i\\
&=\frac{1}{A - 1}\left(\frac{K_i(A - 1) - (x_i + K_i(A-1))}{x_i(x_i + K_i(A - 1))} + \frac{x_i + K_i(A-1)}{x_i(x_i + K_i(A - 1))}\right) dx_i \\
&=\frac{1}{A - 1}\left(\frac{K_i(A - 1) - K_i(A-1) - x_i}{x_i(x_i + K_i(A - 1))} + \frac{1}{x_i}\right) dx_i \\
&=\frac{1}{A - 1}\left(\frac{- x_i}{x_i(x_i + K_i(A - 1))} + \frac{1}{x_i}\right) dx_i \\
&=\frac{1}{A - 1}\left(\frac{1}{x_i} - \frac{1}{x_i + K_i(A - 1)}\right) dx_i \\
-r_i(A-1)dt &= \left(\frac{1}{x_i} - \frac{1}{x_i + K_i(A - 1)}\right) dx_i \\
r_i(A-1)dt &= \left(\frac{1}{x_i + K_i(A - 1)} - \frac{1}{x_i}\right) dx_i
\end{align}
$$

Now we can integrate!

$$
\begin{align}
\int{r_i(A - 1)dt} &= \int{\frac{1}{x_i + K_i(A - 1)}dx_i} - \int{\frac{1}{x_i} dx_i}\\
r_i(A - 1)t - C_1 &= \ln{\left(x_i + K_i(A - 1)\right)} + C_2 - \ln{(x_i)} - C_3
\end{align}
$$

In the above, $C_{k}$ are unknown constants. We will combine these constants into a single $C$ and then rearrange to solve in terms of $x_i$:

$$
\begin{align}
r_i(A - 1)t + C &= \ln{\left(\frac{x_i + K_i(A - 1)}{x_i}\right)} \\
\textrm{e}^{r_i(A - 1)t + C} &= \frac{x_i + K_i(A - 1)}{x_i} \\
x_i\textrm{e}^{r_i(A - 1)t + C} &= x_i + K_i(A - 1) \\
x_i\textrm{e}^{r_i(A - 1)t + C} - x_i &= K_i(A - 1) \\
x_i\left(\textrm{e}^{r_i(A - 1)t + C} - 1)\right) &= K_i(A - 1) \\
x_i &= \frac{K_i(A - 1)}{\textrm{e}^{r_i(A - 1)t + C} - 1}
\end{align}
$$

Substituting our value for $A$ back in yields

$$
\begin{align}
x_i &= \frac{K_i(\frac{\sum_{j \ne i}^{N}\alpha_{ij}x_j}{K_i} - 1)}{\textrm{e}^{r_i(\frac{\sum_{j \ne i}^{N}\alpha_{ij}x_j}{K_i} - 1)t + C} - 1} \\
x_i &= \frac{\sum_{j \ne i}^{N}\alpha_{ij}x_j - K_i}{\textrm{e}^{r_it(\frac{\sum_{j \ne i}^{N}\alpha_{ij}x_j}{K_i} - 1) + C} - 1}
\end{align}
$$

## Competitive Lotka-Volterra in DMS Stan

Eq. 26 is still rather complicated, but we can make some simplifying assumptions based on the use-cases of DMS Stan.

First, in DMS Stan, we are looking at competition between populations of the same species with changes to just one gene. Unlike the ecological settings that the Lotka-Volterra competition equations are typically used to describe where there are a variety of different trophic relationships between species (e.g., predatory, commensal, parasitic, etc. relationships), the types of interactions between these variant populations can be expected to be homogeneous. Thus, rather than assigning an $\alpha_{ij}$ to describe all possible interactions, we might instead assign a single $\alpha_j$ to each variant that describes how that variant impacts all others. In this case, Eq. 26 becomes

$$
\begin{align}
x_i &= \frac{\sum_{j \ne i}^{N}\alpha_{j}x_j - K_i}{\textrm{e}^{r_it(\frac{\sum_{j \ne i}^{N}\alpha_{j}x_j}{K_i} - 1) + C} - 1}.
\end{align}
$$

We can also take advantage of the fact that, in DMS-Stan, we are not looking at population sizes explicitly, but rather their relative proportions. As a result, we know that the carrying capacity of all variants must be "1", which is when they are the only variant present in the population. This means that we can set $K_i = 1$ and further simplify our expression:

$$
\begin{align}
x_i &= \frac{\sum_{j \ne i}^{N}\alpha_{j}x_j - 1}{\textrm{e}^{r_it(\sum_{j \ne i}^{N}\alpha_{j}x_j - 1) + C} - 1}.
\end{align}
$$

If we want to take our assumption of homogeneity made for Eq. 27 even further, we might say that we are going to assume that all other variants $j \ne i$ influence variant $i$ to an equal degree. That is, we can write

$$
\begin{align}
x_i &= \frac{\alpha\sum_{j \ne i}^{N}x_j - 1}{\textrm{e}^{r_it(\alpha\sum_{j \ne i}^{N}x_j - 1) + C} - 1};
\end{align}
$$

and, because we know that $\sum_{i=1}^{N}x_i = 1$, it follows that $x_i + \sum_{j \ne i}^{N}x_j = 1$ and $\sum_{j \ne i}^{N}x_j = 1 - x_i$. Making this substitution gives us

$$
\begin{align}
x_i &= \frac{\alpha(1 - x_i) - 1}{\textrm{e}^{r_it(\alpha(1 - x_i) - 1) + C} - 1},
\end{align}
$$

which can be rearranged as follows to get the solution in terms of $x_i$:

