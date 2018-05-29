using NLopt
using Parameters

@with_kw type OptConfig
    XTOLREL = 0
    XTOLABS = 0
    FTOLREL = 0
    CTOLABS = 0
    DEBUG   = false
    MAXEVAL = 0
    METHOD = :softmax
    OPTIMIZER = :LD_MMA
end


# derivative based optimization of a function over the unit simplex, using unconstrained optimization via the softmax function
function simplex_minimize(f, df, x0; config=OptConfig())
    @unpack_OptConfig config

    opt = Opt(OPTIMIZER, length(x0))

    global xhist, shist, ghist
    xhist = []
    shist = []
    ghist = []

    x0   = log.(x0)

    lasts = fill(NaN, length(x0))

    function objective(x,g)
        s = softmax(x)
        fs = f(s)

        if lasts != s && norm(lasts - s, Inf) < XTOLABS
            force_stop!(opt)
        end

        lasts = s

        if length(g) > 0
            g[:] = softmaxjac(s) * df(s)
        end

        if DEBUG
            push!(xhist, copy(x))
            push!(shist, s)
            push!(ghist, copy(g))
        end

        if any(isnan.(s)) || any(isnan.(g)) 
            warn("NaN encountered")
            force_stop!(opt)
        end

        return fs
    end


    min_objective!(opt, objective)

    xtol_rel!(opt, XTOLREL)
    #xtol_abs!(opt, XTOLABS)
    ftol_rel!(opt, FTOLREL)
    maxeval!(opt, MAXEVAL)
   
    minf, minx, ret = optimize(opt, x0)

    softmax(minx)
end

function softmax(x)
    x = x - maximum(x) # avoid overflow of exp
    r = exp.(x)
    r ./ sum(r)
end

function softmaxjac(s)
    # s is already the softmax here
    diagm(s) - s * s'
end