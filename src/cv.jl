import MLBase

function cvscore(m, d, estimator, k=length(d))
    scores = MLBase.cross_validate(
        i->estimator(d[i]),
        (w,i)->datalikelihood(m, w, d[i]),
        length(d),
        MLBase.Kfold(length(d), k))
    mean(scores)
end

function datalikelihood(m, w, d)
    L = likelihoodmat(m, d)
    logL(w,L)
end

# TODO: think about easier interface
# i.e. give data, model, gammas

function cvreference(m, d, gammas; k=length(d), c = OptConfig())
    regs = [ReferenceRegularizer(m, gamma) for gamma in gammas]
    cvs = [cvscore(m, data, d->ebprior(m, d, r, c), k) for r in regs]
    i = indmax(cvs)
    in(gammas[i], extrema(gammas)) && warn("extremal choice of gamma, consider extending range") 
    regs[i]
end
