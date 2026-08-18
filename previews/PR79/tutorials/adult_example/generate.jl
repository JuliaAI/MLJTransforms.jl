# Use the per-tutorial environment defined by `Project.toml` in this folder
joinpath(@__DIR__, "..", "..", "generate.jl") |> include
generate(@__DIR__, execute = true)
