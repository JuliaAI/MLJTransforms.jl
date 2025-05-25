
"""
**Private method.**

Fit an encoder that encodes the categorical values in the specified
categorical features with their (normalized or raw) frequencies of occurrence in the dataset.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `normalize=false`: Whether to use normalized frequencies that sum to 1 over category values or to use raw counts.

# Returns (in a dict)

  - `statistic_given_feat_val`: The frequency of each level of each selected categorical feature
  - `encoded_features`: The subset of the categorical features of X that were encoded
"""
function frequency_encoder_fit(
    X,
    features = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    normalize::Bool = false,
)
    # 1. Define feature mapper
    function feature_mapper(col, name)
        frequency_map = (!normalize) ? countmap(col) : proportionmap(col)
        statistic_given_feat_val = Dict{Any, Real}(level=>frequency_map[level] for level in levels(col))
        return statistic_given_feat_val
    end

    # 2. Pass it to generic_fit
    statistic_given_feat_val, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )
    cache = Dict(
        :statistic_given_feat_val => statistic_given_feat_val,
        :encoded_features => encoded_features,
    )
    return cache
end

"""
**Private method.**

Encode the levels of a categorical variable in a given table with their (normalized or raw) frequencies of occurrence in the dataset.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `cache`: The output of `frequency_encoder_fit`

# Returns

  - `X_tr`: The table with selected features after the selected features are encoded by frequency encoding.
"""
function frequency_encoder_transform(X, cache::Dict)
    statistic_given_feat_val = cache[:statistic_given_feat_val]
    return generic_transform(X, statistic_given_feat_val)
end
