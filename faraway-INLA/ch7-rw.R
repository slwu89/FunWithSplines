library(INLA)
library(brinla)
library(ggplot2)

n <- 100
x <- seq(0, 1, length=n)
f.true <- (sin(2*pi*x^3))^3
y <- f.true + rnorm(n, sd=0.2)

data.inla <- list(y = y, x = x)
formula1 <- y ~ -1 + f(x, model = "rw1", constr = FALSE)
result1 <- inla(formula1, data = data.inla)
formula2 <- y ~ -1 + f(x, model = "rw2", constr = FALSE)
result2 <- inla(formula2, data = data.inla)

plot_it <- function(results, title) {
    fhat <- results$summary.random$x$mean
    f.lb <- results$summary.random$x$`0.025quant`
    f.ub <- results$summary.random$x$`0.975quant`

    data.plot <- data.frame(y = y, x = x, f.true = f.true, fhat = fhat, f.lb = f.lb, f.ub = f.ub)
    ggplot(data.plot, aes(x = x, y = y)) + 
        geom_line(aes(y = fhat)) + 
        geom_line(aes(y = f.true), linetype = 2) + 
        geom_ribbon(aes(ymin = f.lb, ymax = f.ub), alpha=0.2) + 
        geom_point(aes(y=y)) +
        theme_bw(base_size = 20) +
        ggtitle(title)
}

plot_it(result1, "RW(1) smoothing prior")
plot_it(result2, "RW(2) smoothing prior")

# what about the P-spline section, how is it any different?
