using ForwardDiff
include(Pkg.dir("GynC") * "/src/eb/optim.jl")

 logL(w, L) = sum(log.(L * w)) 
dlogL(w, L) = sum(L ./ (L*w), 1) |> vec

 dkl(w, j) = sum(w[i] * log(w[i]/j[i]) for i in 1:length(w))
ddkl(w, j) = [log(w[i]/j[i]) + 1 for i in 1:length(w)]

function objective(m, reg, data)
    L = likelihoodmat(m, data)  
    nL = size(L, 1) # normalization to bound gradients

    j = jeffreysprior(m)
    obj(w)  = reg * dkl(w,j) - logL(w,L) / nL
    dobj(w) = reg * ddkl(w,j) - dlogL(w,L) / nL

    obj, dobj
end

function ebprior(m, data, γ, c=OPTCONFIG)
    obj, dobj = objective(m, γ, data)
    opt = simplex_minimize(obj, dobj, ones(length(m.xs)), config=c)
end

function dirichletprior(m, data, alpha = 1, c=OPTCONFIG)
    L  = likelihoodmat(m, data)

    F(w)  = - ( logL(w, L) + (alpha-1) * sum(log(wk) for wk in w))
    dF(w) = - (dlogL(w, L) + (alpha-1) * [1/wk for wk in w])

    wo = simplex_minimize(F,dF, ones(length(m.xs)), config=c)
end
