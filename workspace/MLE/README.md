# Core Statistics Examples

## Urchins

### EM optimization

As we recall the basic problem with fitting models with random effects is that to find the MLE $\hat{\theta}$ we need to evaluate the intractable integral:

$$
L(\theta) = f_{\theta}(y) = \int f_{\theta}(y,b) db
$$

The EM algorithm avoids messing with that integral and replaces it with another integral that is sometimes more analytically tractable but much more straightforward to approximate. It starts with a parameter guess $\theta{'}$ and the decomposition of the joint likelihood:

$$
\log f_{\theta}(y,b) = \log f_{\theta}(b|y) + \log f_{\theta}(y)
$$

We then take the expectation of this entire expression wrt $f_{\theta^{'}}(b|y)$ (the E step):

$$
E_{b|y,\theta^{'}} \log f_{\theta}(y,b) = E_{b|y,\theta^{'}} \log f_{\theta}(b|y) + \log f_{\theta}(y)
$$

and rewrite as:

$$
Q_{\theta^{'}}(\theta) = P_{\theta^{'}}(\theta) + l(\theta)
$$

Note that $E_{b|y,\theta^{'}} \log f_{\theta}(b|y) = \int \log f_{\theta}(b|y) f_{\theta^{'}}(b|y) db$.

The algorithm is to find (the M step) $\theta^{*} = \text{argmax}_{\theta}Q_{\theta^{'}}(\theta)$ and set $\theta^{'} \leftarrow \theta^{*}$ until convergence.

Above is the basic EM algorithm but Wood uses a higher order Laplace approximation for the E step, which seriously helps in evaluating $E_{b|y,\theta^{'}} \log f_{\theta}(b|y)$.