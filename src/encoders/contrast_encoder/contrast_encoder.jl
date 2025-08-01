include("errors.jl")

"""
** Private Method **

This and the following four methods implement the contrast matrix for dummy coding, sum coding, 
    backaward/forward difference coding and helmert coding.
Where `k` is the number of levels in the feature and the returned contrast matrix has dimensions (k,k-1).
"""
### 1. Dummy Coding
function get_dummy_contrast(k)
    return Matrix(1.0I, k, k - 1)
end


### 2. Sum Coding
function get_sum_contrast(k)
    C = Matrix(1.0I, k, k - 1)
    C[end, :] .= -1.0
    return C
end

### 3. Backward Difference Coding
function create_backward_vector(index::Int, length::Int)
    # [i/k i/k i/k .. i/k i/k]
    vec = ones(length) .* index / length

    # [ -(k-i)/k -(k-i)/k -(k-i)/k .. i/k i/k]
    vec[1:index] .= index / length - 1
    return vec
end
function get_backward_diff_contrast(k)
    return hcat([create_backward_vector(i, k) for i in 1:k-1]...)
end

### 4. Forward Difference Coding
function get_forward_diff_contrast(k)
    return -get_backward_diff_contrast(k)
end

### 5. Helmert Coding
function create_helmert_vector(index::Int, length::Int)
    # [-1 -1 -1 .. -1 -1]
    vec = -ones(length)
    # [ -1 -1 -1 i .. 0 0]
    vec[index+1] = index
    # [ 0 0 i .. -1 -1]
    if index + 2 <= length
        vec[index+2:end] .= 0.0
    end
    return vec
end
function get_helmert_contrast(k)
    return hcat([create_helmert_vector(i, k) for i in 1:k-1]...)
end

"""
** Private Method **

Fit a contrast encoing scheme on given data in `X`.

# Arguments

    $X_doc
    $features_doc
  - `mode=:dummy`: The type of encoding to use. Can be one of `:contrast`, `:dummy`, `:sum`, `:backward_diff`, `:forward_diff`, `:helmert` or `:hypothesis`.
    If `ignore=false` (features to be encoded are listed explictly in `features`), then this can be a vector of the same length as `features` to specify a different
    contrast encoding scheme for each feature
  - `buildmatrix=nothing`: A function or other callable with signature `buildmatrix(colname, k)`,
    where `colname` is the name of the feature levels and `k` is it's length, and which returns contrast or
    hypothesis matrix with row/column ordering consistent with the ordering of `levels(col)`. Only relevant if `mode` is `:contrast` or `:hypothesis`.
    $ignore_doc
    $ordered_factor_doc

# Returns as a named-tuple

  - `vec_given_feat_level`: Maps each level for each column in the selected categorical features to a vector
  $encoded_features_doc
"""
function contrast_encoder_fit(
    X,
    features = Symbol[];
    mode::Union{Symbol, AbstractVector{Symbol}} = :dummy,
    buildmatrix = nothing,
    ignore::Bool = true,
    ordered_factor::Bool = false,
)
    # mode should be a vector only if features is a vector of the same length
    mode_is_vector = false
    if mode isa Vector{Symbol}
        mode_is_vector = true
        ignore && throw(ArgumentError(IGNORE_MUST_FALSE_VEC_MODE))
        length(features) == length(mode) ||
            throw(ArgumentError(LENGTH_MISMATCH_VEC_MODE(length(mode), length(features))))
    end

    # buildmatrix should be specified if mode is :contrast or :hypothesis
    if mode in (:contrast, :hypothesis)
        buildmatrix === nothing && throw(ArgumentError(BUILDFUNC_MUST_BE_SPECIFIED))
    end


    # ensure mode is one of :contrast, :dummy, :sum, :backward_diff, :forward_diff, :helmert, :polynomial, :hypothesis
    function feature_mapper(col, name)
        feat_levels = levels(col)
        k = length(feat_levels)
        feat_mode = (mode_is_vector) ? mode[findfirst(isequal(name), features)] : mode
        if feat_mode == :contrast
            contrastmatrix = buildmatrix(name, k)
            size(contrastmatrix) == (k, k - 1) ||
                throw(ArgumentError(MATRIX_SIZE_ERROR(k, size(contrastmatrix), name)))
        elseif feat_mode == :hypothesis
            hypothesismatrix = buildmatrix(name, k)
            size(hypothesismatrix) == (k - 1, k) ||
                throw(ArgumentError(MATRIX_SIZE_ERROR_HYP(k, size(hypothesismatrix), name)))
            contrastmatrix = pinv(hypothesismatrix)
        elseif feat_mode == :dummy
            contrastmatrix = get_dummy_contrast(k)
        elseif feat_mode == :sum
            contrastmatrix = get_sum_contrast(k)
        elseif feat_mode == :backward_diff
            contrastmatrix = get_backward_diff_contrast(k)
        elseif feat_mode == :forward_diff
            contrastmatrix = get_forward_diff_contrast(k)
        elseif feat_mode == :helmert
            contrastmatrix = get_helmert_contrast(k)
        else
            throw(ArgumentError("Mode $feat_mode is not supported."))
        end

        vector_given_value_given_feature = OrderedDict(
            level => contrastmatrix[l, :] for (l, level) in enumerate(feat_levels)
        )
        return vector_given_value_given_feature
    end

    # 2. Pass it to generic_fit
    vector_given_value_given_feature, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )
    cache = (
        vector_given_value_given_feature = vector_given_value_given_feature,
        encoded_features = encoded_features,
    )

    return cache
end

"""
** Private Method **

Use a fitted contrast encoder to encode the levels of selected categorical variables with contrast encoding.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `cache`: The output of `contrast_encoder_fit`

# Returns

  - `X_tr`: The table with selected features after the selected features are encoded by contrast encoding.
"""
function contrast_encoder_transform(X, cache::NamedTuple)
    vector_given_value_given_feature = cache.vector_given_value_given_feature
    return generic_transform(
        X,
        vector_given_value_given_feature,
        single_feat = false;
        use_levelnames = true,
    )
end
