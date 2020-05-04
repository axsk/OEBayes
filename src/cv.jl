import MLBase

function cvscore(m, d, estimator::Function, k=length(d))
    L = likelihoodmat(m, d)
    score(w, i) = logL(w, L[i,:])   # logL(priorest_[-i] | data_[i])
    scores = MLBase.cross_validate(i->estimator(d[i]), score, length(d), MLBase.Kfold(length(d), k))
    mean(scores)
end

" Given model m, data and a list of gammas, return the (cv-)best regularizer"
function cvreference(m, data, gammas; k=length(d), c = OptConfig())
    j = jeffreysprior(m)
    regs = [ReferenceRegularizer(j, gamma) for gamma in gammas]
    cvs = [cvscore(m, data, d->ebprior(m, d, r, c), k) for r in regs]
    i = indmax(cvs)
    in(gammas[i], extrema(gammas)) && warn("extremal choice of gamma, consider extending range") 
    regs[i], cvs
end


# TODO: cleanup
# above and below code do the same
# bottom one is cleaner but works only for the ReferenceRegularizer.

import Optim 

function cvreference(m, d, min, max, ; c = OptConfig(), kwargs...)

    L = likelihoodmat(m, d)

    function gammascore(gamma)  
        score = 0
        for i=1:length(d)
            inds = collect(1:length(d))
            deleteat!(inds, i)
            w = ebprior(L[inds, :], ReferenceRegularizer(m, gamma), c)
            score += logL(w, L[[i],:])
        end
        score / length(d)
    end

    opt = Optim.optimize(x->-gammascore(x[1]) - 1.40, min, max, Optim.GoldenSection(); kwargs...)
    opt.minimizer, opt
end





