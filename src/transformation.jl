## Transformation of densities
# f: X -> Y

using ForwardDiff

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
    @assert isapprox(xs, LinRange(a,b,length(xs)))
    @assert isapprox(sum(w), 1)
    w * (length(xs) / (b-a))
end

function transformmodel(m::FEModel, transf)
    transformedlims = [apply(transf,x) for x in extrema(m.xs)]
    xst = LinRange(transformedlims..., length(m.xs))
    ft(x) = m.f(invert(transf, x))

    FEModel(f=ft, xs=xst, σ=m.σ)
end
