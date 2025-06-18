
"""
**Private method.**

Fit an encoder that encodes the categorical values in the specified
categorical features with their (normalized or raw) frequencies of occurrence in the dataset.

# Arguments

    $X_doc
    $features_doc
    $ignore_doc
    $ordered_factor_doc
  - `normalize=false`: Whether to use normalized frequencies that sum to 1 over category values or to use raw counts.

# Returns as a named-tuple

  - `statistic_given_feat_val`: The frequency of each level of each selected categorical feature
  $encoded_features_doc
"""
function frequency_encoder_fit(
    X,
    features = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    normalize::Bool = false,
    output_type::Type = Float32,
)
    # 1. Define feature mapper
    function feature_mapper(col, name)
        frequency_map = (!normalize) ? countmap(col) : proportionmap(col)
        feat_levels = levels(col)
        statistic_given_feat_val = Dict{eltype(feat_levels), output_type}(
            level => frequency_map[level] for level in feat_levels
        )
        return statistic_given_feat_val
    end

    # 2. Pass it to generic_fit
    statistic_given_feat_val, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper)

    cache = (
        statistic_given_feat_val = statistic_given_feat_val,
        encoded_features = encoded_features,
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
function frequency_encoder_transform(X, cache::NamedTuple)
    statistic_given_feat_val = cache.statistic_given_feat_val
    return generic_transform(X, statistic_given_feat_val)
end
