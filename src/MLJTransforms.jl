module MLJTransforms
using Tables
using ScientificTypes
using ScientificTypes: scitype
using CategoricalArrays
using MLJModelInterface
using TableOperations
using StatsBase
using LinearAlgebra
using OrderedCollections: OrderedDict
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
export  TargetEncoder

# Ordinal encoding
include("encoders/ordinal_encoding/ordinal_encoding.jl")
include("encoders/ordinal_encoding/interface_mlj.jl")
export  OrdinalEncoder

# Frequency encoding
include("encoders/frequency_encoding/frequency_encoding.jl")
include("encoders/frequency_encoding/interface_mlj.jl")
export frequency_encoder_fit, frequency_encoder_transform, FrequencyEncoder
export  FrequencyEncoder

# Cardinality reduction
include("transformers/cardinality_reducer/cardinality_reducer.jl")
include("transformers/cardinality_reducer/interface_mlj.jl")
export cardinality_reducer_fit, cardinality_reducer_transform, CardinalityReducer
export  CardinalityReducer
include("encoders/missingness_encoding/missingness_encoding.jl")
include("encoders/missingness_encoding/interface_mlj.jl")
export  MissingnessEncoder

# Contrast encoder
include("encoders/contrast_encoder/contrast_encoder.jl")
include("encoders/contrast_encoder/interface_mlj.jl")
export ContrastEncoder

# MLJModels transformers
include("transformers/other_transformers/continuous_encoder.jl")
include("transformers/other_transformers/interaction_transformer.jl")
include("transformers/other_transformers/univariate_time_type_to_continuous.jl")
include("transformers/other_transformers/fill_imputer.jl")
include("transformers/other_transformers/one_hot_encoder.jl")
include("transformers/other_transformers/standardizer.jl")
include("transformers/other_transformers/univariate_boxcox_transformer.jl")
include("transformers/other_transformers/univariate_discretizer.jl")
include("transformers/other_transformers/metadata_shared.jl")

export UnivariateDiscretizer,
    UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, FillImputer, UnivariateFillImputer,
    UnivariateTimeTypeToContinuous, InteractionTransformer
end
