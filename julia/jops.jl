using LinearAlgebra
using BSplines

"""
Give the `k` order difference operator for an `n` dimensional matrix
"""
function diffmat(n, k)
    k < n || error("k must be strictly less than n")
    r = 1 * I(n)
    for _ in 1:k
        r = r[2:end, :] - r[1:end-1, :]
    end
    return r
end

# the reason that `BSplineBasis` has extra knots at the end is explained here: https://mc-stan.org/learn-stan/case-studies/splines_in_stan.html
# basically it is because if you have splines of order k (polynomial degree + 1) then you need to repeat the endpoints k-1 times
# otherwise things will not be well defined at the endpoints
"""
Compute a B-spline basis matrix

    * `x`: vector of points we evaluate the basis functions at
    * `xl`: lower support, usu min(x)
    * `xr`: upper support, usu max(x)
    * `nseg`: number of segments on support
    * `bdeg`: degree of splines (1 for linear, 3 for cubic are typical choices)

Returns a basis matrix `B` with number of rows equal to length of `x` and number of columns
equal `nseg` + `bdeg`.
"""
function bbase(x, xl, xr, nseg, bdeg)
    # compute breakpoints
    dx = (xr - xl)/nseg
    knots = collect(range(start = xl - bdeg * dx, step = dx, stop = xr + bdeg * dx))

    # BSplineBasis duplicates low/upper breakpoints such that they appear k times
    k = bdeg + 1
    bbasis = BSplines.BSplineBasis(k, knots)

    # the basis matrix evaluated at points x
    B = basismatrix(bbasis, x)
    B = B[:, bdeg+1:end-bdeg]
    return B
end