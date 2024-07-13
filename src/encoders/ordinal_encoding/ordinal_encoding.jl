
"""
Fit an encoder to encode the levels of categorical variables in a given table as integers (ordered arbitrarily).

# Arguments

  - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the columns given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them

# Returns (in a dict)

  - `index_given_feat_level`: Maps each level for each column in a subset of the categorical columns of X into an integer.
  - `encoded_features`: The subset of the categorical columns of X that were encoded
"""
function ordinal_encoder_fit(
    X,
    features::AbstractVector{Symbol} = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
)
    # 1. Define column mapper
    function feature_mapper(col, ind)
        feat_levels = levels(col)
        index_given_feat_val =
            Dict{Any, Integer}(value => index for (index, value) in enumerate(feat_levels))
        return index_given_feat_val
    end

    # 2. Pass it to generic_fit
    index_given_feat_level, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )
    cache = Dict(
        :index_given_feat_level => index_given_feat_level,
        :encoded_features => encoded_features,
    )
    return cache
end


"""
Encode the levels of a categorical variable in a given table as integers.

# Arguments

  - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `cache`: The output of `ordinal_encoder_fit`

# Returns

  - `X_tr`: The table with selected columns after the selected columns are encoded by ordinal encoding.
"""
function ordinal_encoder_transform(X, cache::Dict)
    index_given_feat_level = cache[:index_given_feat_level]
    return generic_transform(X, index_given_feat_level)
end
