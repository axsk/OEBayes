using ForwardDiff
include(Pkg.dir("GynC") * "/src/eb/optim.jl")

 logL(w, L) = sum(log.(L * w)) 
dlogL(w, L) = sum(L ./ (L*w), 1) |> vec

 dkl(w, j) = sum(w[i] * log(w[i]/j[i]) for i in 1:length(w))
ddkl(w, j) = [log(w[i]/j[i]) + 1 for i in 1:length(w)]

function objective(m, reg, data)
    L = likelihoodmat(m, data)  
    nL = size(L, 1) # normalization to bound gradients

    #if reg < 0
	j = jeffreysprior(m)
	obj(w)  = reg * dkl(w,j) - logL(w,L) / nL
	dobj(w) = reg * ddkl(w,j) - dlogL(w,L) / nL
    #else
	#obj(w)  = - logL(w,L) / nL
	#dobj(w) = - dlogL(w,L) / nL
    #end

    obj, dobj
end

function ebprior(m, data, γ, c=OPTCONFIG)
    obj, dobj = objective(m, γ, data)
    opt = simplex_minimize(obj, dobj, ones(length(m.xs)), config=c)
end


## Plot
using Plots
using StatPlots
pyplot()


### Models
using Parameters
using Distributions

abstract type Model end

l2err(m, w) = norm(w - wprior(m))


## f + Error Model
@with_kw struct FEModel <: Model
    f = x->x.^2
    n = 30
    xs = linspace(1,2,n)
    σ = 0.1
    γ = 1
    prior = TruncatedNormal(1.5, 0.1, 1, 2)
end

wprior(m::FEModel) = normalize(pdf.(m.prior, m.xs), 1)
jeffreysprior(m::FEModel) = normalize([ForwardDiff.derivative(m.f, x) for x in m.xs], 1)
likelihoodmat(m::FEModel, data) = [pdf(Normal(y, m.σ), d) for d in data, y in m.f.(m.xs)]

function generatedata(m::FEModel, n; smooth=true)
    if smooth
	dataxs = rand(m.prior, n)
    else
	dataxs = m.xs[rand(Categorical(wtrue(m)), n)]
    end
    datays = m.f.(dataxs) + rand(Normal(0, m.σ), n)
end

Plots.plot(m::FEModel, w) = plot(m.xs, w)

## Normal(mu, sig) Model
@with_kw struct MuSigModel <: Model
    nx = 20
    ny = 20
    bndmu = (-1,1)
    bndsig = (0.01, 0.5)
    xs = [(x,y) for x in linspace(bndmu...,nx), y in linspace(bndsig...,ny)] |> vec
    γ  = 1
    priormu  = TruncatedNormal(0, 0.3, bndmu...)
    priorsig = TruncatedNormal(0.1, 0.03, bndsig...)
    nmeas = 2
end

wprior(m::MuSigModel) = map(x -> pdf(m.priormu, x[1]) * pdf(m.priorsig, x[2]), m.xs) |> x->x/sum(x) #:: Vector
jeffreysprior(m::MuSigModel) = [x[2]^-2 for x in m.xs] |> x -> (x / sum(x)) :: Vector
likelihoodmat(m::MuSigModel, data) = [prod(pdf.(Normal(x[1],x[2]), d)) for d in data, x in m.xs] :: Matrix
generatedata(m::MuSigModel, ndata) = [rand(Normal(rand(m.priormu), rand(m.priorsig)), m.nmeas) for i = 1:ndata] :: Vector

Plots.plot(m::MuSigModel, w) = surface((x->x[1]).(m.xs), (x->x[2]).(m.xs), w, xlabel="μ", ylabel="σ")


## Poisson

@with_kw struct PoissonModel <: Model
    nx = 20
    xs = linspace(0.1, 5, nx)
    prior = Gamma(5,1)
end

wprior(m::PoissonModel) = map(x->pdf(m.prior, x), m.xs) |> x->x/sum(x)

jeffreysprior(m::PoissonModel) = [sqrt(1/x) for x in m.xs] |> x->x/sum(x)

likelihoodmat(m::PoissonModel, data) = [pdf(Poisson(x), d) for d in data, x in m.xs]

generatedata(m::PoissonModel, ndata) = [rand(Poisson(rand(m.prior))) for i = 1:ndata]

Plots.plot(m::PoissonModel, w) = plot(m.xs, w)


## Transformed Model
struct TransformedModel <: Model
    model
    transformation
end

struct Transformation
    f
    finv
end

apply(t::Transformation, x) = t.f(x)
invert(t::Transformation, x) = t.finv(x)


