using MLJTransforms
using Test
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJModelInterface
using MLJBase
const MMI = MLJModelInterface

@testset "Target Encoding" begin
   include("target_encoding.jl")
   include("ordinal_encoding.jl")
   include("frequency_encoder.jl")
end

