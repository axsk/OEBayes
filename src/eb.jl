abstract type Regularizer end

 logL(w, L) = sum(log.(L * w)) 
dlogL(w, L) = sum(L ./ (L*w), 1) |> vec

function ebprior(m::Model, data, reg::Regularizer, c=OptConfig())
    L = likelihoodmat(m, data)
    @assert !haszerorow(L)
    nL = size(L, 1)

    obj(w)  = -  logL(w, L) / nL +  f(reg, w)
    dobj(w) = - dlogL(w, L) / nL + df(reg, w)

    opt = simplex_minimize(obj, dobj, ones(length(m.xs)), config=c)
end

haszerorow(L) = any(all(L[i,:].<=0) for i in 1:size(L,1))

## Regularizers

type ReferenceRegularizer <: Regularizer
    j::Vector
    γ::Number
end

ReferenceRegularizer(m::Model, γ) = ReferenceRegularizer(jeffreysprior(m), γ)

f(r::ReferenceRegularizer, w) = r.γ * dkl(w, r.j)
df(r::ReferenceRegularizer, w) = r.γ * ddkl(w, r.j)

dkl(w, j) = sum(w[i] * log(w[i]/j[i]) for i in 1:length(w))
ddkl(w, j) = [log(w[i]/j[i]) + 1 for i in 1:length(w)]


type DirichletRegularizer <: Regularizer
    α::Number
end
# Note that were missing the regularization Parameter, but its just an additive constant in logspace, not influencing the optimization

f(r::DirichletRegularizer, w) = -(r.α - 1) * sum(log(wk) for wk in w) # - logpdf of Dir(α)
df(r::DirichletRegularizer, w) = -(r.α - 1) * [1/wk for wk in w]


type ThikonovRegularizer <: Regularizer
    γ::Number
end

f(r::ThikonovRegularizer, w) = r.γ * sum(abs2, w)
df(r::ThikonovRegularizer, w) = r.γ * 2 * w
