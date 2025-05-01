# mle urchins
urchins <- read.table("https://webhomes.maths.ed.ac.uk/~swood34/data/urchin-vol.txt")

v0e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(exp(w)*exp(exp(g)*a)))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v0 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

v1e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(
          exp(p)/exp(g) + exp(p)*(a - ((p-g-w)/exp(g)))
     ))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v1 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))


pdR <- function(H,k.mult=20,tol=.Machine$double.eps^.8) {
     k <- 1
     tol <- tol * norm(H)
     n <- ncol(H)
     while (
          inherits(try(R <- chol(H + (k-1)*tol*diag(n)), silent=TRUE),"try-error")
     ) {
          k <- k * k.mult
     }
     R 
}

# eval f_{\theta}(y,b) and the gradient and hessian
lfyb <- function(b,y,a,th) {
     ## evaluate joint p.d.f. of y and b + grad. and Hessian.
     n <- length(y)
     g <- b[1:n]; p <- b[1:n+n]
     am <- (p-g-th[1])/exp(g)
     ind <- a < am
     f0 <- v0(a[ind],y[ind],g[ind],p[ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     f1 <- v1(a[!ind],y[!ind],g[!ind],p[!ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     lf <- sum(f0) + sum(f1)
     g <- matrix(0,n,2) ## extract gradient to g... g[ind,] <- attr(f0,"gradient") ## dlfyb/db g[!ind,] <- attr(f1,"gradient") ## dlfyb/db
     h <- array(0,c(n,2,2)) ## extract Hessian to H... h[ind,,] <- attr(f0,"hessian")
     h[!ind,,] <- attr(f1,"hessian") 
     H <- matrix(0,2*n,2*n)
     for (i in 1:2) for (j in 1:2) {
          indi <- 1:n + (i-1)*n; indj <- 1:n + (j-1)*n
          diag(H[indi,indj]) <- h[,i,j]
     }
     list(lf=lf,g=as.numeric(g),H=H)
}

# s \log f_{\theta}(y,b) + \log f_{\theta^{'}}(y,b)
lfybs <- function(s,b,vol,age,th,thp) {
     ## evaluate s log f(y,b;th) + log f(y,b;thp)
     lf <- lfyb(b,vol,age,thp)
     if (s!=0) {
          lfs <- lfyb(b,vol,age,th)
          lf$lf <- lf$lf + s * lfs$lf
          lf$g <- lf$g + s * lfs$g
          lf$H <- lf$H + s * lfs$H
     }
     lf 
}

# now need function to maximize this wrt b
# returns \log f_{\theta}(y, \hat{b}) (just the joint log likelihood at \hat{b}) if s=0
# and \log |H_{s}|/s otherwise to compute 5.14 which is the approximation of Q_{\theta^{'}}(\theta)