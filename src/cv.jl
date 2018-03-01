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
