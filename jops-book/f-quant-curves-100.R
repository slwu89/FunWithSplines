# Quantile curves for body weight of 100 boys (Dutch boys data)
# Paul Eilers and Brian Marx 2019
# A graph in "Practical Smoothing. The Joys of P-splines"

library(colorspace)
library(AGD)
library(ggplot2)
library(JOPS)

# Get the data
data(boys7482)
Dat = subset(boys7482, age > 5, select = c(age, wgt))
Dat = na.omit(Dat)
m0 = nrow(Dat)
m = 100
set.seed(2013)
sel = sample(1:m0, m)
x = (Dat$age[sel])
y = Dat$wgt[sel]

# P-spline paremeters
xl = min(x)
xr = max(x)
nsegx = 20
bdeg = 3
pp = seq(0.1, 0.9, by = 0.1)
pp = c(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
np= length(pp)

# Compute bases
B = bbase(x, xl, xr, nsegx, bdeg)
nbx = ncol(B)
xg = seq(xl, xr, length = 100)
Bg = bbase(xg, xl, xr, nsegx, bdeg)

# Compute penalty matrices
D = diff(diag(nbx), diff = 2)
lambdax = 1
P = lambdax * t(D) %*% D

Zg = NULL
for (p in pp) {
  b = 0.001
  dzmax = 1e-3 * max(y)
  z = 0
  for (it in 1:500) {
    r = y - z
    w = ifelse(r > 0, p, 1- p)  / sqrt(b ^ 2 + r ^ 2)
    W = diag(c(w))
    Q = t(B) %*% W %*% B
    a = solve(Q + P, t(B) %*% (w * y))
    znew = B %*% a
    dz = sum(abs(z - znew))
    # cat(it, dz, '\n')
    if (dz < dzmax) break
    z = znew
  }
  Zg = cbind(Zg, Bg %*% a)
  cat(p, '\n')
}
np = length(pp)
cols = diverge_hcl(np, h = c(240, -30), c= 100, l = 60, power = 1)

# Dataframes for ggplot
DF1 = data.frame(x = x, y = y)
np = length(pp)
ng = length(xg)
DF2 = data.frame(x = rep(xg, np), y = as.vector(Zg), p = as.factor(rep(pp, each = ng)))

# Make and save plots
plt1 =  ggplot(DF1,  aes(x = x, y = y)) +
  geom_point(color = I('grey70'), size = 1.5) +
  xlab('Age (yr)') + ylab('Weight (kg)') +
  ggtitle('Quantile curves for weights of 100 Dutch boys') +
  geom_line(data = DF2, aes(x = x, y = y, group = p, color = p), size = 1) +
  JOPS_theme()  +
  labs(color = bquote(tau)) +
  scale_color_manual(values=rainbow_hcl(np, start = 10, end = 350))

plot(plt1)


