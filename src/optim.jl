using NLopt
@with_kw struct OptConfig
    XTOLREL = 0
    XTOLABS = 0
    FTOLREL = 0
    CTOLABS = 0
    DEBUG   = false
    MAXEVAL = 10000
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

    last = fill(NaN, length(x0))

    function objective(x,g)
        s = softmax(x)
        fs = f(s)

        delta = last - s

        # check for tolerances on the transformed simplex space
        if (last != s) && # if lasts=s continue, since some optimizers re-evaluate points
            (norm(delta, Inf) < XTOLABS) ||
            (norm(delta ./ last, Inf) < XTOLREL)
            force_stop!(opt)
        end

        last = s

        if length(g) > 0
            objectivejac!(g, s, df(s)) |> testnan
        end

        if DEBUG
            push!(xhist, copy(x))
            push!(shist, s)
            push!(ghist, copy(g))
        end

        if any(isnan.(s)) || any(isnan.(g)) 
            @warn("NaN encountered in optimization [$(sum(s)), $(sum(g)), $(sum(df(s)))]")
            force_stop!(opt)
        end

        return fs
    end


    min_objective!(opt, objective)

    # xtol_rel!(opt, XTOLREL) taken care of by own check on transformed space
    # xtol_abs!(opt, XTOLABS) 
    ftol_rel!(opt, FTOLREL)
    maxeval!(opt, MAXEVAL)

    minf, minx, ret = optimize(opt, x0)

    (opt.numevals == MAXEVAL) && @warn("Simplex optimization did not converge")

    softmax(minx)
end

function softmax(x)
    x = x .- maximum(x) # avoid overflow of exp
    r = exp.(x)
    r ./ sum(r)
end


# optimized (inplace, matrix-free) version of
# dobj/dw = df/ds * ds/dw 
# where obj(w) = f(s(w))
# and ds/dw = diagm(s) - s * s'
function objectivejac!(g::Vector, s::Vector, df::Vector)
     d = dot(s,df)
     for i=1:length(g)
        g[i] = s[i] * (df[i]-d)
     end
     g
end