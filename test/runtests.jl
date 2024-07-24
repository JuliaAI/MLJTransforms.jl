using MLJTransforms
using Test
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJModelInterface
using MLJBase
using StatsBase
using LinearAlgebra
using StatsModels
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

# Other transformers
include("transformers/other_transformers/fill_imputation.jl")
include("transformers/other_transformers/one_hot_encoding.jl")
include("transformers/other_transformers/continuous_transform_time.jl")
include("transformers/other_transformers/interaction_transform.jl")
include("transformers/other_transformers/continuous_encoding.jl")
include("transformers/other_transformers/uni_box_cox.jl")
include("transformers/other_transformers/standardization.jl")
include("transformers/other_transformers/uni_discretization.jl")
