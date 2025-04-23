urchins <- read.csv2("./workspace/MLE/urchin.CSV", sep=" ", header=TRUE)

v0e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(exp(w)*exp(exp(g)*a)))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v0 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

v1e 

v1