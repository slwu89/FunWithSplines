# FunWithSplines
something about splines

The `jops-book` folder is all from the "Scripts.zip" archive which may be downloaded from [https://psplines.bitbucket.io/](https://psplines.bitbucket.io/).

For Bayesian P-Splines, this is the key paper and describes the RW(1) and RW(2) priors well https://www.tandfonline.com/doi/abs/10.1198/1061860043010 (it's available free on JSTOR).

## Faraway et al. (Bayesian Regression with INLA)

Let the smoothing spline have degree $2m-1$ such that if $m=1$ we get linear splines, and if $m-2$ we get cubic splines. The spline with a penalized least squares criterion is:

$$
\sum_{i=1}^{n} [y_{i} - f(x_{i})]^{2} + \lambda \int [f^{(m)}(x)]^{2} dx
$$

where $f^{(m)}(x)$ is the $m^{th}$ derivative of $f(x)$ and $\lambda$ is the smoothing parameter.

### RW Priors for Equally-Spaced Locations

We have $n$ observations. Now assume that our $x_{i}$ observations are ordered (smallest to largest) and that the distance between each pair is the same, $d_i = x_i - x_{i-1}$ and $d_i = d$. In this case the penalty can be approximated as:

$$
\int \left(f^{(m)}(x)\right)^{2} dx \approx d^{-(2m-1)} \sum_{i=m+1}^{n}  \left[ \nabla^{m}f(x_{i}) \right]^{2}
$$

$\nabla^{m}$ is the $m^{th}$ order backwards difference operator:

  * $\nabla^{1}f(x) = f(x_i) - f(x_{i-1})$
  * $\nabla^{2}f(x) = f(x_i) - 2f(x_{i-1}) + f(x_{i-2})$

Now we assume each difference independently follows a Gaussian distribution:

$$
\nabla^{m}f(x_{i}) \sim N(0, \sigma^{2}_{f}) \quad i = m+1, \ldots, n
$$

In this case we get a Bayesian version of the penalized smoothing spline, with $\lambda=\sigma^{2}_{\epsilon} / \sigma^{2}_{f}$, ($\sigma^{2}_{\epsilon}$ is the error variance for the i.i.d. observations).

### RW Priors for Non-Equally Spaced Locations

Basic idea is that the smoothing spline estimator is the solution to an SDE (Weiner process). Let the observations be ordered again and now let $d_i = x_{i+1} - x_i$. Then the RW(1) prior tells us that:

$$
f(x_{i+1}) - f(x) \sim N(0, d_{i} \sigma^{2}_{f}) \quad i=1, \ldots, n-1
$$

Note that if $d_{i} = 1$ for all $i$ we get the same as the equally spaced case.

For the RW2 prior there is a bit of complexity  

blah blah

### P-Splines

Let $f(x)$ be the unknown function we want to approximate. We use a B-spline of degree $d$ and equally-spaced knots $x_{min} < t_1 < \ldots < t_r < x_{max}$. We can write it as:

$$
f(x) = \sum_{j=1}^{p} \beta_j B_j(x)
$$

Where $B_j$ is the B-spline basis function and $p=d+r+1$ is the "degrees of freedom". What we want to do is penalize $\beta_j$ so that the selection of knots is far less important than smoothing parameter $\lambda$. Use the difference operator defined previously $\nabla^{m}\beta_j$, but on adjacent $\beta_j$ for $j=m+1,\ldots,p$.

Then we get a P-spline estimator that minimizes:

$$
\sum_{i=1}^{n} \left[ y_i - \sum_{j=1}^{p} \beta_{j} B_{j}(x_{i}) \right]^{2} + \lambda \sum_{j=m+1}^{p} \left( \nabla^{m} \beta_{j} \right)^{2}
$$

The difference between P-splines and smoothing splines is this: for splines the observed unique $x$ values are the knots and $\lambda$ alone is used to control the smoothing. For P-splines the knots of the B-splines used for the basis are much smaller than $n$.

The problem has the following Bayesian representation:

$$
y | \beta, \sigma_{\epsilon}^{2} \sim N(B\beta, \sigma_{\epsilon}^{2}I), \quad \beta | \sigma_{\beta}^{2}  \sim N \left( 0, \sigma^{2}_{\beta} Q_{m} \right)
$$

Where $Q_{m}$ is "the one from the RW models". But it's not clear if it is the one for equally spaced observations ($Q=D^{T}D$) or the highly complex one for non-equally spaced observations. As before, we have $\lambda = \sigma^{2}_{\epsilon} / \sigma^{2}_{\beta}$.

## Synthesis

From 2.3 in JOPS book and Lang and Brezger (2003) we get that the first and second order differences are as below. Let $\beta$ refer to the vector of coefficients. Note that $\nabla^{d}\beta_i$ does not exist for $i < 1+d$.

  * $\nabla \beta_i = \beta_i - \beta_{i-1}$ and we have $\nabla \beta_i \sim N(0,\sigma_{f}^{2})$ therefore $\beta_i \sim N(\beta_{i-1},\sigma_{f}^{2}), \quad i>1$, and we let $\beta_1$ follow some diffuse prior.
  * $\nabla^{2}\beta_i = \nabla(\beta_i - \beta_{i-1}) = \nabla \beta_i - \nabla \beta_{i-1} = (\beta_i - \beta_{i-1}) - (\beta_{i-1} - \beta_{i-2}) = \beta_i - 2\beta_{i-1} + \beta_{i-2}$. We have $\nabla^{2}\beta_i \sim N(0,\sigma_{f}^{2})$ therefore $\beta_i \sim N(2\beta_{i-1}-\beta_{i-2},\sigma_{f}^{2}), \quad i>2$, and we let $\beta_1$ and $\beta_2$ follow some diffuse prior.

## (G)LMMs with residual autocorrelation

This wild goose chase arose when I read section 2.5.2 of Wood's [GAM book](https://www.maths.ed.ac.uk/~swood34/igam/index.html), where the following LMM is fit:

$$
\text{height}_{ji} = \beta_{0} + \beta_{1}\text{age}_{ji} + \beta_{2}\text{age}_{ji}^{2} + \beta_{3}\text{age}_{ji}^{3} + b_{0} + b_{1}\text{age}_{ji} + b_{2}\text{age}_{ji}^{2} + b_{3}\text{age}_{ji}^{3} + \epsilon_{ji}
$$

For $i^{th}$ measurement on the $j^{th}$ tree. The complexity is that the $\epsilon_{ji}$ terms have an AR(1) correlation structure within each tree $j$ to account for repeated observations. It turns out that the [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) package cannot fit models with this type of residual correlation (confirmed by [asking on Discourse](https://discourse.julialang.org/t/ar-1-correlation-structure-on-errors-with-mixedmodels-jl/128066/5)).

The Wood book uses the `nlme` package in R to do the fit, the documentation for that package is [https://rdrr.io/cran/nlme/](https://rdrr.io/cran/nlme/) and has some helpful notes on the `corAR1` function used to set up the correlation structure. The `nlme` code is somewhat complex though, and it wouldn't be straightforward to just read it to figure out how to replicate (although that may be necessary).

I came across some other things that may be useful to figure out how to untie (cut) the gordian knot. Ben Bolker has an article [fitting mixed models with (temporal) correlations in R](https://bbolker.github.io/mixedmodels-misc/notes/corr_braindump.html) here which provides some examples. I believe he has an example there of using `lme4` to fit in a very manual way a model with AR(1) correlation (something the package by default cannot do). This is worth looking into in *a lot* of detail, especially because there is a pure R implementation of the guts of `lme4` here [https://github.com/lme4/lme4pureR/tree/master](https://github.com/lme4/lme4pureR/tree/master). So we may even seek to see if we can do it with the R version, and then from that translate it to Julia. It would be a good idea to read about the [`modular`](https://rdrr.io/github/lme4/lme4/man/modular.html) method from `lme4` which is used in that tutorial.

There are some other things to read, such as Ben Bolker's site [https://bbolker.github.io/mixedmodels-misc/](https://bbolker.github.io/mixedmodels-misc/) and start with the GLMM FAQ [https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html).

May be worth it to keep [Pinheiro and Bates (2000)](https://link.springer.com/book/10.1007/b98882) close at hand.
