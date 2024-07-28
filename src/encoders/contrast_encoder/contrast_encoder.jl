include("errors.jl")

"""
** Private Method **
This and the following four methods implement the contrast matrix for dummy coding, sum coding, 
    backaward/forward difference coding and helmert coding.
Where `k` is the number of levels in the feature and the returned contrast matrix has dimensions (k,k-1).
"""
### 1. Dummy Coding
function get_dummy_contrast(k)
    return Matrix(1.0I, k, k-1)
end


### 2. Sum Coding
function get_sum_contrast(k)
    C = Matrix(1.0I, k, k-1)
    C[end, :] .= -1.0
    return C
end

### 3. Backward Difference Coding
function create_backward_vector(index::Int, length::Int)
    # [i/k i/k i/k .. i/k i/k]
    vec = ones(length) .* index / length

    # [ -(k-i)/k -(k-i)/k -(k-i)/k .. i/k i/k]
    vec[1:index] .= index/length - 1
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

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `mode=:dummy`: The type of encoding to use. Can be one of `:contrast`, `:dummy`, `:sum`, `:backward_diff`, `:forward_diff`, `:helmert` or `:hypothesis`.
  If `ignore=false` (features to be encoded are listed explictly in `features`), then this can be a vector of the same length as `features` to specify a different
  contrast encoding scheme for each feature
  - `buildmatrix=nothing`: A function that should take column name as a symbol and the number of levels as input and return a contrast or hypothesis matrix. 
  Only relevant if `mode` is `:contrast` or `:hypothesis`.
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them

# Returns (in a dict)

  - `vec_given_feat_level`: Maps each level for each column in the selected categorical features to a vector
  - `encoded_features`: The subset of the categorical features of X that were encoded
"""
function contrast_encoder_fit(
    X,
    features::AbstractVector{Symbol} = Symbol[];
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
        length(features) == length(mode) || throw(ArgumentError(LENGTH_MISMATCH_VEC_MODE(length(mode), length(features))))
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
            size(contrastmatrix) == (k, k-1) || throw(ArgumentError(MATRIX_SIZE_ERROR(k, size(contrastmatrix), name)))
        elseif feat_mode == :hypothesis
            hypothesismatrix = buildmatrix(name, k) 
            size(hypothesismatrix) == (k-1, k) || throw(ArgumentError(MATRIX_SIZE_ERROR_HYP(k, size(hypothesismatrix), name)))
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

        vec_given_feat_val = Dict(level=>contrastmatrix[l, :] for (l, level) in enumerate(feat_levels))
        return vec_given_feat_val
    end

    # 2. Pass it to generic_fit
    vec_given_feat_val, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )

    cache = Dict(
        :vec_given_feat_val  => vec_given_feat_val,
        :encoded_features => encoded_features,
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
function contrast_encoder_transform(X, cache::Dict)
    vec_given_feat_val = cache[:vec_given_feat_val]
    return generic_transform(X, vec_given_feat_val, single_feat = false)
end