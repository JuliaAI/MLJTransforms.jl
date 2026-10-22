joinpath(@__DIR__, "..", "..", "generate.jl") |> include
generate(@__DIR__, execute=true)
