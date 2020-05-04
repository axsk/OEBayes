module ObjectiveEmpiricalBayes

include("models.jl")
include("optim.jl")
include("eb.jl")
include("transformation.jl")
include("cv.jl")
include("plot.jl")

function testnan(x)
    any(isnan.(x)) && throw(DomainError(x, "A computation returned NaN"))
    x
end

end
