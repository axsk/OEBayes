## Transformation of densities
# f: X -> Y

struct Transformation
    f
    finv
end

apply(t::Transformation, x) = t.f(x)
invert(t::Transformation, x) = t.finv(x)

function pullbackdensity(t::Transformation, ys, wy)
    f(x)    = apply(t, x)
    finv(x) = invert(t, x) 
    df(x) = ForwardDiff.derivative(f, x)

    xs = finv.(ys)
    wx = wy .* df.(xs)

    xs, wx
end

function pushforwarddensity(t::Transformation, xs, wx)
    f(x)    = apply(t, x)
    finv(y) = invert(t, y) 
    dfinv(y) = ForwardDiff.derivative(finv, y)

    ys = f.(xs) # new coordinates
    wy = wx .* dfinv.(ys)

    ys, wy
end

function weighttodensity(xs, w)
    a, b = extrema(xs)
    @assert isapprox(xs, linspace(a,b,length(xs)))
    @assert isapprox(sum(w), 1)
    w * (length(xs) / (b-a))
end

