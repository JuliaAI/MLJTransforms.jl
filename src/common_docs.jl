const X_doc = """
- X: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
    `Multiclass` or `OrderedFactor`
"""
const X_doc_mlj = """
- `X` is any table of input features (eg, a `DataFrame`). Features to be transformed must
   have element scitype `Multiclass` or `OrderedFactor`. Use `schema(X)` to 
   check scitypes. 
"""
const features_doc = """
- features=[]: A list of names of categorical features given as symbols to exclude or include from encoding,
  according to the value of `ignore`, or a single symbol (which is treated as a vector with one symbol),
  or a callable that returns true for features to be included/excluded
"""
const ignore_doc = """
- ignore=true: Whether to exclude or include the features given in `features`
"""
const ordered_factor_doc = """
- ordered_factor=false: Whether to encode `OrderedFactor` or ignore them
"""
const encoded_features_doc = """
- encoded_features: The subset of the categorical features of `X` that were encoded
"""
const cache_doc = """
- `cache`: The output of `contrast_encoder_fit`
"""

