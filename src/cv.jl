import MLBase

function cvscore(m, d, estimator::Function, k=length(d))
    L = likelihoodmat(m, d)
    score(w, i) = logL(w, L[:,i])   # logL(priorest_[-i] | data_[i])
    scores = MLBase.cross_validate(i->estimator(d[i]), score, length(d), MLBase.Kfold(length(d), k))
    mean(scores)
end

" Given model m, data d and a list of gammas, return the (cv-)best regularizer"
function cvreference(m, d, gammas; k=length(d), c = OptConfig())
    j = jeffreysprior(m)
    regs = [ReferenceRegularizer(j, gamma) for gamma in gammas]
    cvs = [cvscore(m, data, d->ebprior(m, d, r, c), k) for r in regs]
    i = indmax(cvs)
    in(gammas[i], extrema(gammas)) && warn("extremal choice of gamma, consider extending range") 
    regs[i], cvs
end
