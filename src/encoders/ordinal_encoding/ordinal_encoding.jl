
"""
**Private method.**

Fit an encoder to encode the levels of categorical variables in a given table as integers (ordered arbitrarily).

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `dtype`: The numerical concrete type of the encoded features. Default is `Float32`.

# Returns (in a dict)

  - `index_given_feat_level`: Maps each level for each column in a subset of the categorical features of X into an integer.
  - `encoded_features`: The subset of the categorical features of X that were encoded
"""
function ordinal_encoder_fit(
    X,
    features = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    output_type::Type = Float32,
)
    # 1. Define feature mapper
    function feature_mapper(col, name)
        feat_levels = levels(col)
        index_given_feat_val =
            Dict{eltype(feat_levels), output_type}(
                value => index for (index, value) in enumerate(feat_levels)
            )
        return index_given_feat_val
    end

    # 2. Pass it to generic_fit
    index_given_feat_level, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,)
    cache = (
      index_given_feat_level = index_given_feat_level,
      encoded_features = encoded_features,
    )
    return cache
end


"""
**Private method.**

Encode the levels of a categorical variable in a given table as integers.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `cache`: The output of `ordinal_encoder_fit`

# Returns

  - `X_tr`: The table with selected features after the selected features are encoded by ordinal encoding.
"""
function ordinal_encoder_transform(X, cache::NamedTuple)
    index_given_feat_level = cache.index_given_feat_level
    return generic_transform(X, index_given_feat_level)
end
