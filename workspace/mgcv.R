library(gamair)

## Q.13 trees
## a)
EV.func <- function(b,g,h)
{ mu <- b[1]*g^b[2]*h^b[3]
  J <- cbind(g^b[2]*h^b[3],mu*log(g),mu*log(h))
  list(mu=mu,J=J)
}
## b)
attach(trees)
b <- c(.002,2,1);b.old <- 100*b+100
while (sum(abs(b-b.old))>1e-7*sum(abs(b.old))) {
   EV <- EV.func(b,Girth,Height)
   z <- (Volume-EV$mu) + EV$J%*%b
   b.old <- b
   b <- coef(lm(z~EV$J-1))
}
b
## c)
sig2 <- sum((Volume - EV$mu)^2)/(nrow(trees)-3)
Vb <- solve(t(EV$J)%*%EV$J)*sig2
se <- diag(Vb)^.5;se

library(nlme)
data(Rail)
write.csv(Rail, "./data/Rail.csv", row.names=FALSE)

# ch2
library(nlme) ## for Rail data
options(contrasts=c("contr.treatment","contr.treatment"))
Z <- model.matrix(~Rail$Rail-1) ## r.e. model matrix
X <- matrix(1,18,1)             ## fixed model matrix
## fit the model...
y = Rail$travel
theta = c(0.1,-2)

llm <- function(theta,X,Z,y) {
  ## untransform parameters...
  sigma.b <- exp(theta[1])
  sigma <- exp(theta[2])
  ## extract dimensions...
  n <- length(y); pr <- ncol(Z); pf <- ncol(X)
  ## obtain \hat \beta, \hat b...
  X1 <- cbind(X,Z)
  ipsi <- c(rep(0,pf),rep(1/sigma.b^2,pr))
  b1 <- solve(crossprod(X1)/sigma^2+diag(ipsi),
              t(X1)%*%y/sigma^2)
  ## compute log|Z'Z/sigma^2 + I/sigma.b^2|...
  ldet <- sum(log(diag(chol(crossprod(Z)/sigma^2 + 
              diag(ipsi[-(1:pf)])))))
  ## compute log profile likelihood...
  l <- (-sum((y-X1%*%b1)^2)/sigma^2 - sum(b1^2*ipsi) - 
  n*log(sigma^2) - pr*log(sigma.b^2) - 2*ldet - n*log(2*pi))/2
  attr(l,"b") <- as.numeric(b1) ## return \hat beta and \hat b
  -l 
}

rail.mod <- optim(c(0,0),llm,hessian=TRUE,
                           X=X,Z=Z,y=Rail$travel)
exp(rail.mod$par) ## variance components
solve(rail.mod$hessian) ## approx cov matrix for theta 
attr(llm(rail.mod$par,X,Z,Rail$travel),"b")

# ch2.5 linear mixed models in R
library(nlme)
data(Rail)
lme(travel ~ 1, Rail, list(Rail = ~ 1))

data(Loblolly)
Loblolly$age <- Loblolly$age - mean(Loblolly$age)
lmc <- lmeControl(niterEM=500,msMaxIter=100)
m0 <- lme(
  height ~ age + I(age^2) + I(age^3),Loblolly,
  random = list(Seed = ~ age + I(age^2) + I(age^3)), 
  correlation = corAR1(form = ~ age|Seed),control=lmc
)
