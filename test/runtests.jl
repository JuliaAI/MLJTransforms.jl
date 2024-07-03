using MLJTransforms
using Test
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJModelInterface
using MLJBase
using StatsBase
using Random
const MMI = MLJModelInterface

@testset "Target Encoding" begin
    include("generic.jl")
    include("encoders/target_encoding.jl")
    include("encoders/ordinal_encoding.jl")
    include("encoders/frequency_encoder.jl")
    include("transformers/cardinality_reducer.jl")
end

