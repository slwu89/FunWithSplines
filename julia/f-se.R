
library(ggplot2)
library(colorspace)
library(MASS)
library(JOPS)

# Get the data
data(mcycle)
x = mcycle$times
y = mcycle$accel
Data = data.frame(x, y)

# Fitting, the tuning parameter chosen based on min CV
# fit = psNormal(x, y, nseg = 20, bdeg = 3, pord = 2, lambda = 0.8)

nseg = 20
bdeg = 3
pord = 2
lambda = 0.8

xl = min(x)
xr = max(x)
wts = NULL
xgrid = 100


# figure out bbase
dx <- (xr - xl) / nseg
knots <- seq(xl - bdeg * dx, xr + bdeg * dx, by = dx)

((xr + bdeg * dx) - (xl - bdeg * dx)) / dx

# psNormal <- function(x, y, xl = min(x), xr = max(x), nseg = 10, bdeg = 3,
#                      pord = 2, lambda = 1, wts = NULL, xgrid = 100) {
  m <- length(x)
  B <- bbase(x, xl = xl, xr = xr, nseg = nseg, bdeg = bdeg)

  # Construct penalty stuff
  n <- dim(B)[2]
  P <- sqrt(lambda) * diff(diag(n), diff = pord)
  nix <- rep(0, n - pord)

  # Fit
  if (missing(wts)) {
    wts <- rep(1, m)
  }
  f <- lsfit(rbind(B, P), c(y, nix), intercept = FALSE, wt = c(wts, (nix + 1)))
  qr <- f$qr
  h <- hat(qr)[1:m]
  beta <- f$coef
  mu <- B %*% beta

  # Cross-validation and dispersion
  r <- (y - mu) / (1 - h)
  cv <- sqrt(mean(r^2))
  ed <- sum(h)
  sigma <- sqrt(sum((y - mu)^2) / (m - ed))

  # Compute curve on grid
  if (length(xgrid) == 1) {
    xgrid <- seq(xl, xr, length = xgrid)
  }
  Bu <- bbase(xgrid, xl = xl, xr = xr, nseg = nseg, bdeg = bdeg)
  zu <- Bu %*% beta
  ygrid <- zu

  # SE bands on a grid using QR
  R <- qr.R(qr)
  L <- forwardsolve(t(R), t(Bu))
  v2 <- sigma^2 * colSums(L * L)
  se_eta <- sqrt(v2)

  # Return list
  pp <- list(
    x = x, y = y, B = B, P = P, muhat = mu, nseg = nseg, xl = xl,
    xr = xr, bdeg = bdeg, pord = pord, lambda = lambda,
    cv = cv, effdim = ed, ed_resid = m - ed, wts = wts,
    pcoeff = beta, family = "gaussian", link = "identity",
    sigma = sigma, xgrid = xgrid, ygrid = ygrid, se_eta = se_eta
  )
  class(pp) <- "pspfit"
  return(pp)
# }