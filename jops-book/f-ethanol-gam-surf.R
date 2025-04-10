# Image of GAM surface (Ethanol data)
# A graph in the book 'Practical Smoothing. The Joys of P-splines'
# Paul Eilers and Brian Marx, 2019

library(SemiPar)
library(reshape2)
library(data.table)
library(ggplot2)
library(JOPS)
library(metR)

# Get the data
data(ethanol)

# Basis matrix for compression ratio (C)
clo = 7.5
chi = 18
ngrid = 50
B1 = bbase(ethanol$C, clo, chi)

# Basis matrix for equivalenc ratio (E)
elo = 0.5
ehi = 1.25
B2 = bbase(ethanol$E, elo, ehi)

# Penalty matrix
n = ncol(B1)
D = diff(diag(n), diff = 2)
r1 = 1:n
r2 = r1 + n
lambdas = c(1, 0.1)
P = kronecker(diag(lambdas), t(D) %*% D)
P = P + diag(2 * n) * 1e-06

# Fit the model
B = cbind(B1, B2)
y = ethanol$NOx
a = solve(t(B) %*% B + P, t(B) %*% y)
mu = B %*% a
cat('SD of residuals:', sd(y - mu), '\n')

# Add fitted components to data frame (for ggplot)
Fmod = ethanol
Fmod$f1 = B1 %*% a[r1]
Fmod$f2 = B2 %*% a[r2]

# Model fit on fine grid
cgrid = seq(clo, chi, length = ngrid)
Bg1 = bbase(cgrid, clo, chi)
egrid = seq(elo, ehi, length = ngrid)
Bg2 = bbase(egrid, elo, ehi)
zc = Bg1 %*% a[r1]
ze = Bg2 %*% a[r2]
Fit = outer(c(zc), c(ze), "+")

# Fill data frames for ggplot
Fc = data.frame(cgrid, zc)
Fe = data.frame(egrid, ze)

# Data frame for plotting data points with signs of residuals
x = ethanol$C
y = ethanol$E
z = ethanol$NOx
cols = c("blue", "yellow")[(z > mu) + 1]
pchs = c("+", "-")[(z > mu) + 1]
Data = data.frame(x = x, y = y, cols = cols, pchs = pchs, NOx = z)

# Turn matrix into a "long" data frame
Mu = Fit
rownames(Mu) = cgrid
colnames(Mu) = egrid
dens <- melt(Mu)
names(dens) = c('x', 'y', 'NOx')

# Plot fit with contours
sl = T
ccol = 'blue'
plt = ggplot(dens,  aes(x, y, fill = NOx)) +
       geom_raster(show.legend = sl) +
       scale_fill_gradientn(colours = terrain.colors(100))+
       geom_contour(data = dens, aes(z = NOx), color = ccol, show.legend = T) +
       geom_text_contour(aes(z = NOx), color = ccol, size = 3)  +
       ylab('Equivalence ratio') + xlab('Compression ratio') +
       ggtitle("GAM for NOx emission, ethanol data") +
       geom_point(data = Data, aes(x =x, y = y),  shape = pchs, size = 5) +
       JOPS_theme() +
       theme(panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             plot.title = element_text(size = 16),
             axis.title.x = element_text(size = 16),
             axis.title.y = element_text(size = 16))


print(plt)
