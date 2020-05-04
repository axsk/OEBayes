abstract type Regularizer end

 logL(w, L) = sum(log.(L * w)) 
dlogL(w, L) = sum(L ./ (L*w), dims=1) |> vec

function ebprior(m::Model, data, reg::Regularizer, c=OptConfig())
    L = likelihoodmat(m, data)
    ebprior(L, reg, c)
end

function ebprior(L::Matrix, reg::Regularizer, c = OptConfig())
    nd, nx = size(L)
    obj(w)  = -  logL(w, L) / nd +  f(reg, w)
    dobj(w) = - dlogL(w, L) / nd + df(reg, w)
    opt = simplex_minimize(obj, dobj, ones(nx), config=c)
end

haszerorow(L) = any(all(L[i,:].<=0) for i in 1:size(L,1))

## Regularizers

struct ReferenceRegularizer <: Regularizer
    j::Vector
    γ::Number
    sampledensity::Vector
end

ReferenceRegularizer(m::Model, γ) = ReferenceRegularizer(jeffreysprior(m), γ, sampledensity(m))

f(r::ReferenceRegularizer, w) = r.γ * dkl(w, r.j, r.sampledensity)
df(r::ReferenceRegularizer, w) = r.γ * ddkl(w, r.j, r.sampledensity)

dkl(w, j, ref) = sum(w[i] * log(w[i]*ref[i]/j[i]) for i in 1:length(w))
ddkl(w, j, ref) = [log(w[i]*ref[i]/j[i]) + 1 for i in 1:length(w)]


struct DirichletRegularizer <: Regularizer
    α::Number
end
# Note that were missing the regularization Parameter, but its just an additive constant in logspace, not influencing the optimization

f(r::DirichletRegularizer, w) = -(r.α - 1) * sum(log(wk) for wk in w) # - logpdf of Dir(α)
df(r::DirichletRegularizer, w) = -(r.α - 1) * [1/wk for wk in w]


struct ThikonovRegularizer <: Regularizer
    γ::Number
end

f(r::ThikonovRegularizer, w) = r.γ * sum(abs2, w)
df(r::ThikonovRegularizer, w) = r.γ * 2 * w
