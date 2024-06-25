module MLJTransforms
using Tables
using ScientificTypes
using CategoricalArrays
using MLJModelInterface: MLJModelInterface
using TableOperations
using MLJBase
const MMI = MLJModelInterface

# Target encoding
include("target_encoding/errors.jl")
include("target_encoding/target_encoding.jl")
export target_encoding_fit, transformit
include("target_encoding/interface_mlj.jl")
export TargetEncoder


end