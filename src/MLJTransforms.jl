module MLJTransforms
using Tables
using ScientificTypes
using ScientificTypes: scitype
using CategoricalArrays
using MLJModelInterface
using TableOperations
using StatsBase
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
export  TargetEncoder

# Ordinal encoding
include("encoders/ordinal_encoding/ordinal_encoding.jl")
include("encoders/ordinal_encoding/interface_mlj.jl")
export ordinal_encoder_fit, ordinal_encoder_transform, OrdinalEncoder
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

end