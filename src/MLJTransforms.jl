module MLJTransforms
using Tables
using ScientificTypes
using CategoricalArrays
using MLJModelInterface
using TableOperations
using StatsBase

const MMI = MLJModelInterface

# Functions of generic use across transformers
include("generic.jl")

# Target encoding
include("target_encoding/errors.jl")
include("target_encoding/target_encoding.jl")
include("target_encoding/interface_mlj.jl")
export target_encoder_fit, target_encoder_transform, TargetEncoder

# Ordinal encoding
include("ordinal_encoding/ordinal_encoding.jl")
include("ordinal_encoding/interface_mlj.jl")
export ordinal_encoder_fit, ordinal_encoder_transform, OrdinalEncoder

# Frequency encoding
include("frequency_encoding/frequency_encoding.jl")
include("frequency_encoding/interface_mlj.jl")
export frequency_encoder_fit, frequency_encoder_transform, FrequencyEncoder


end