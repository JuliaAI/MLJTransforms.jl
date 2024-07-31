include("errors.jl")

"""
**Private method.**

Fit a transformer that maps any missing value into a new level (e.g., "Missing"). By this, missingness will be treated as a new
level by any subsequent model. This assumes that the categorical features have raw
types that are in `Union{Char, AbstractString, Number}`.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `label_for_missing::Dict{<:Type, <:Any}()= Dict( AbstractString => "missing", Char => 'm', )`: A
    dictionary where the possible values for keys are the types in `Union{Char, AbstractString, Number}` and where each value
    signifies the new level to map into given a column raw super type. By default, if the raw type of the column subtypes `AbstractString`
    then missing values will be replaced with `"missing"` and if the raw type subtypes `Char` then the new value is `'m'`
    and if the raw type subtypes `Number` then the new value is the lowest value in the column - 1.

# Returns (in a dict)

  - `new_cat_given_col_val`: A dictionary that for each column, maps `missing` into some value according to `label_for_missing`
  - `encoded_features`: The subset of the categorical features of X that were encoded
"""
function missingness_encoder_fit(
    X,
    features::AbstractVector{Symbol} = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    label_for_missing::Dict{<:Type, <:Any} = Dict(    
        AbstractString => "missing",
        Char => 'm',
    ),
)
    supportedtypes = Union{Char, AbstractString, Number}

    # 1. Define feature mapper
    function feature_mapper(col, name)
        col_type = nonmissingtype(eltype(col)).parameters[1]
        feat_levels = levels(col; skipmissing=true)

        # Ensure column type is valid (can't test because never occurs)
        # Converting array elements to strings before wrapping in a `CategoricalArray`, as...
        if !(col_type <: supportedtypes)
            throw(ArgumentError(UNSUPPORTED_COL_TYPE_ME(col_type)))
        end

        # Ensure label_for_missing keys are valid types
        for possible_col_type in keys(label_for_missing)
            if !(possible_col_type in union_types(supportedtypes))
                throw(ArgumentError(VALID_TYPES_NEW_VAL_ME(possible_col_type)))
            end
        end

        # Check no collision between keys(label_for_missing) and feat_levels
        for value in values(label_for_missing)
            if !ismissing(value) 
                if value in feat_levels
                    throw(ArgumentError(COLLISION_NEW_VAL_ME(value)))
                end
            end
        end

        # Get ancestor type of column
        elgrandtype = nothing
        for allowed_type in union_types(supportedtypes)
            if col_type <: allowed_type
                elgrandtype = allowed_type
                break
            end
        end
        
        # Nonmissing levels remain as is
        new_cat_given_col_val = Dict{Missing, col_type}()

        # Missing levels are mapped
        if elgrandtype in keys(label_for_missing)
            new_cat_given_col_val[missing] = label_for_missing[elgrandtype]
        elseif elgrandtype == Number
            new_cat_given_col_val[missing] = minimum(feat_levels) - 1
        else
            throw(ArgumentError(UNSPECIFIED_COL_TYPE_ME(col_type, label_for_missing)))
        end

        return new_cat_given_col_val::Dict{Missing, col_type}
    end

    # 2. Pass it to generic_fit
    new_cat_given_col_val, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor, feature_mapper = feature_mapper,
    )
    cache = Dict(
        :new_cat_given_col_val => new_cat_given_col_val,
        :encoded_features => encoded_features,
    )
    return cache
end

"""
**Private method.**

Apply a fitted missingness encoder to a table given the output of `missingness_encoder_fit`

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `cache`: The output of `missingness_encoder_fit`

# Returns

  - `X_tr`: The table with selected features after the selected features are transformed by missingness encoder
"""
function missingness_encoder_transform(X, cache::Dict)
    new_cat_given_col_val = cache[:new_cat_given_col_val]
    return generic_transform(X, new_cat_given_col_val; ignore_unknown = true)
end

