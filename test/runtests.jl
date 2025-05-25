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
using LinearAlgebra
using StatsModels

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
# include("encoders/target_encoding.jl")
# include("encoders/ordinal_encoding.jl")
# include("encoders/frequency_encoder.jl")
# include("transformers/cardinality_reducer.jl")
# include("encoders/missingness_encoding.jl")
include("encoders/contrast_encoder.jl")

# Other transformers
include("transformers/other_transformers/fill_imputer.jl")
include("transformers/other_transformers/one_hot_encoder.jl")
include("transformers/other_transformers/univariate_time_type_to_continuous.jl")
include("transformers/other_transformers/interaction_transformer.jl")
include("transformers/other_transformers/continuous_encoder.jl")
include("transformers/other_transformers/univariate_boxcox_transformer.jl")
include("transformers/other_transformers/standardizer.jl")
include("transformers/other_transformers/univariate_discretizer.jl")