### Models
using Parameters
using Distributions
using ForwardDiff

abstract type Model end

" assign uniform density as fallback for all models"
sampledensity(m::Model) = ones(length(m.xs))

l2err(m, w) = norm(w - wprior(m))
wprior(m::Model, prior) = normalize(pdf.(prior, m.xs), 1)

## f + Error Model
struct FEModel <: Model
    f
    xs
    σ
end

FEModel(; f=x->x.^2, n=30, xs=linspace(1,2,n), σ=0.1) = FEModel(f,xs,σ)

jeffreysprior(m::FEModel) = normalize([ForwardDiff.derivative(m.f, x) for x in m.xs], 1)
likelihoodmat(m::FEModel, data) = [pdf(Normal(y, m.σ), d) for d in data, y in m.f.(m.xs)]

function generatedata(m::FEModel, prior, n; smooth=true)
    if smooth
	dataxs = rand(prior, n)
    else
	dataxs = m.xs[rand(Categorical(wtrue(m)), n)]
    end
    datays = m.f.(dataxs) + rand(Normal(0, m.σ), n)
end


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

#Plots.plot(m::MuSigModel, w) = surface((x->x[1]).(m.xs), (x->x[2]).(m.xs), w, xlabel="μ", ylabel="σ")


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

#Plots.plot(m::PoissonModel, w) = plot(m.xs, w)
