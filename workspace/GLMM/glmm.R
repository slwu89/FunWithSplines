library(lme4)

dd <- expand.grid(id=factor(1:20),rep=factor(1:20))
set.seed(101)
dd <- transform(dd,x=rnorm(nrow(dd)),y=rnorm(nrow(dd)))
dd$z <- simulate(~1+ (x+y|id),
                 family=gaussian,
                 newdata=dd,
                 newparams=list(beta=1,
                                theta=rep(1,6),
                                sigma=1))[[1]]


lmod <- lFormula(z ~ 1 + (x+y|id), data=dd)
devfun <- do.call(mkLmerDevfun, lmod)

## direct computation of Cholesky root for an AR(1) matrix
ar1_chol <- function(rho,p) {
    R <- matrix(0,p,p)
    R[1,] <- rho^(0:(p-1))        ## formula for 1st row
    cc <- sqrt(1 - rho^2);        ## scaling factor: c^2 + rho^2 = 1
    R2 <-  cc * R[1,]             ## formula for 2nd row */
    for (j in 2:p) {              ## shift elements in 2nd row for remaining rows
        R[j, j:p] <- R2[1:(p-j+1)] 
    }
    return(R)
}

## keep messing up devfun??
## devfun <- do.call(mkLmerDevfun, lmod)
wrap_AR1 <- function(theta) {
  cc <- ar1_chol(theta[2],p=3)
  tvec <- t(cc)[lower.tri(cc,diag=TRUE)]*theta[1]
  d <- devfun(tvec)
  ## cat(theta,d,"\n")
  return(d)
}
opt_AR1 <- nloptwrap(par=c(1,0),wrap_AR1,lower=c(0,-0.49),upper=c(Inf,1))
VarCorr(m_AR1 <- mkMerMod(environment(devfun), opt_AR1, lmod$reTrms, fr = lmod$fr))
