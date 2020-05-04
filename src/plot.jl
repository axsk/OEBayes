using PyPlot, LaTeXStrings
using Seaborn


function compareplot(m, t, wtrue, w, wt, name, x1, y1, x2, y2; figsize=(5,3), savename = "", savemat = false)

    mt   = transformmodel(m, t)

    figure(figsize=figsize)

    #subplot(1,2,1)
    plot(m.xs, wtrue, label="\$\\pi^*\$", alpha=.3, linewidth=2.2)

    plot(m.xs, weighttodensity(m.xs, w), label="\$\\pi_{\\rm $name}\$")

    xsp, wtp = pullbackdensity(t, mt.xs, weighttodensity(mt.xs,wt))
    plot(xsp, wtp, label="\$\\varphi_*^{-1} \\pi^\\varphi_{\\rm $name}\$", linestyle="--")

    Seaborn.seaborn.despine()
    legend(ncol=1)#, numpoints=10)
    
    xticks(x1)
    xlim(x1)
    ylim(y1)
    yticks(y1)

    p1 = gcf()
    savename != "" && savefig(savename * ".pdf", bbox_inches="tight")

 
    figure(figsize=figsize)

    t1 = pushforwarddensity(t, m.xs, wtrue)
    t2 = pushforwarddensity(t, m.xs, weighttodensity(m.xs, w))

    #subplot(1,2,2)
    plot(t1..., label="\$\\varphi_* \\pi^*\$", alpha=.3, linewidth=2.2, linestyle="--")
    plot(t2..., label="\$\\varphi_* \\pi_{\\rm $name}\$", linestyle="--", zorder=5)
    plot(mt.xs, weighttodensity(mt.xs, wt), label="\$\\pi^\\varphi_{\\rm $name}\$")

    Seaborn.seaborn.despine()
    legend(ncol=1)#, numpoints=3)

    xticks(x2)
    xlim(x2)
    ylim(y2)
    yticks(y2)

    p2 = gcf()
    savename != "" && savefig(savename * "t.pdf", bbox_inches="tight")

    p1,p2
end

#=using MAT

function savemat(m,t, wtrue, w, wt, filename)
    mt   = transformmodel(m, t)
    xsp, wtp = pullbackdensity(t, mt.xs, weighttodensity(mt.xs,wt))

    xst, piTf = pushforwarddensity(t, m.xs, wtrue)
    xst, pif = pushforwarddensity(t, m.xs, weighttodensity(m.xs, w))

    matwrite(filename, Dict(
        "xs" => m.xs,
        "pitrue" => wtrue,
        "pi" => weighttodensity(m.xs, w),
        "pitp" => weighttodensity(m.xs, w),
        "xst" => xst,
        "pitrue" => piTf,
        "pif" => pif,
        "pit" => weighttodensity(mt.xs, wt)
    ))

end
=#
