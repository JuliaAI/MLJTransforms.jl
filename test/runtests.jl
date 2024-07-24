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

# Other transformers
using Tables, CategoricalArrays
using ScientificTypes: scitype
using Statistics
using StableRNGs
stable_rng = StableRNGs.StableRNG(123)
using Dates: DateTime, Date, Time, Day, Hour
_get(x) = CategoricalArrays.DataAPI.unwrap(x)


include("utils.jl")
include("generic.jl")
include("encoders/target_encoding.jl")
include("encoders/ordinal_encoding.jl")
include("encoders/frequency_encoder.jl")
include("transformers/cardinality_reducer.jl")
include("encoders/missingness_encoding.jl")
