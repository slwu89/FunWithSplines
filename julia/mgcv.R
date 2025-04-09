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
