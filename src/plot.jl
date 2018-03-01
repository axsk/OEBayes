using PyPlot, LaTeXStrings
using Seaborn


function compareplot(m, t, wtrue, w, wt, name)

    mt   = transformmodel(m, t)

    plot(m.xs, wtrue, label="\$\\pi_{\\rm true}\$", alpha=.3, linewidth=2.2)

    plot(m.xs, weighttodensity(m.xs, w), label="\$\\pi_{\\rm $name}\$")

    xsp, wtp = pullbackdensity(t, mt.xs, weighttodensity(mt.xs,wt))
    plot(xsp, wtp, label="\$\\varphi_*^{-1} \\pi^\\varphi_{\\rm $name}\$", linestyle="--")

    Seaborn.seaborn["despine"]()
    legend()
    
    #xticks([0,4])
    #xlim([0,4])
    #yticks([0,0.5])
    #scatter(d, zeros(d))

    p1 = gcf()

    figure()

    t1 = pushforwarddensity(t, m.xs, wtrue)
    t2 = pushforwarddensity(t, m.xs, weighttodensity(m.xs, w))

    plot(t1..., label="\$\\varphi_* \\pi_{\\rm true}\$", alpha=.3, linewidth=2.2, linestyle="--")
    plot(t2..., label="\$\\varphi_* \\pi_{\\rm $name}\$", linestyle="--", zorder=5)
    plot(mt.xs, weighttodensity(mt.xs, wt), label="\$\\pi^\\varphi_{\\rm $name}\$")

    Seaborn.seaborn["despine"]()
    legend()

    #xticks([0,])
    #xlim([0,4])
    #yticks([0,0.1])
    #xmax = t.f(maximum(m.xs))
    #xlim([0, xmax])
    #xticks([0,50])
    #ylim([0, 0.2])
    #yticks([0,0.2])

    p2 = gcf()

    p1,p2
end

