using MLJTransforms
using Test
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJModelInterface: MLJModelInterface
using MLJBase
const MMI = MLJModelInterface

@testset "Target Encoding" begin
   include("target_encoding.jl")
end

