module MLJTransforms
using Tables
using ScientificTypes
using ScientificTypes: scitype
using CategoricalArrays
using MLJModelInterface
using TableOperations
using StatsBase
using LinearAlgebra

# Other transformers
using Combinatorics
import Distributions
using Parameters
using Dates
using OrderedCollections

const MMI = MLJModelInterface

# Functions of generic use across transformers
include("generic.jl")
include("utils.jl")

# Target encoding
include("encoders/target_encoding/errors.jl")
include("encoders/target_encoding/target_encoding.jl")
include("encoders/target_encoding/interface_mlj.jl")
export target_encoder_fit, target_encoder_transform, TargetEncoder

# Ordinal encoding
include("encoders/ordinal_encoding/ordinal_encoding.jl")
include("encoders/ordinal_encoding/interface_mlj.jl")
export ordinal_encoder_fit, ordinal_encoder_transform, OrdinalEncoder

# Frequency encoding
include("encoders/frequency_encoding/frequency_encoding.jl")
include("encoders/frequency_encoding/interface_mlj.jl")
export frequency_encoder_fit, frequency_encoder_transform, FrequencyEncoder

# Cardinality reduction
include("transformers/cardinality_reducer/cardinality_reducer.jl")
include("transformers/cardinality_reducer/interface_mlj.jl")
export cardinality_reducer_fit, cardinality_reducer_transform, CardinalityReducer
# MLJModels transformers
include("transformers/other_transformers/continuous_encoding.jl")
include("transformers/other_transformers/interaction_transform.jl")
include("transformers/other_transformers/continuous_transform_time.jl")
include("transformers/other_transformers/fill_imputation.jl")
include("transformers/other_transformers/one_hot_encoding.jl")
include("transformers/other_transformers/standardization.jl")
include("transformers/other_transformers/uni_box_cox.jl")
include("transformers/other_transformers/uni_discretization.jl")
include("transformers/other_transformers/metadata_registry.jl")

export UnivariateDiscretizer,
    UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, FillImputer, UnivariateFillImputer,
    UnivariateTimeTypeToContinuous, InteractionTransformer
end
