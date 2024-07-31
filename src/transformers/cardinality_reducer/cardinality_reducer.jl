
include("errors.jl")

"""
**Private method.**

Fit a transformer that maps any level of a categorical feature that occurs with
frequency < `min_frequency` into a new level (e.g., "Other"). This is useful when some categorical features have
high cardinality and many levels are infrequent. This assumes that the categorical features have raw
types that are in `Char`, `AbstractString`, and `Number`.

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the features given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `min_frequency::Real=3`: Any level of a categorical feature that occurs with frequency < `min_frequency` will be mapped to a new level. Could be
    an integer or a float which decides whether raw counts or normalized frequencies are used.
  - `label_for_infrequent::Dict{<:Type, <:Any}()= Dict( AbstractString => "Other", Char => 'O', )`: A
    dictionary where the possible values for keys are the types in `Char`, `AbstractString`, and `Number` and each value signifies
    the new level to map into given a column raw super type. By default, if the raw type of the column subtypes `AbstractString`
    then the new value is `"Other"` and if the raw type subtypes `Char` then the new value is `'O'`
    and if the raw type subtypes `Number` then the new value is the lowest value in the column - 1.

# Returns (in a dict)

  - `new_cat_given_col_val`: A dictionary that maps each level in a
    categorical feature to a new level (either itself or the new level specified in `label_for_infrequent`)
  - `encoded_features`: The subset of the categorical features of X that were encoded
"""
function cardinality_reducer_fit(
    X,
    features::AbstractVector{Symbol} = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    min_frequency::Real = 3,                        
    label_for_infrequent::Dict{<:Type, <:Any} = Dict(    
        AbstractString => "Other",
        Char => 'O',
    ),
)   
    supportedtypes = Union{Char, AbstractString, Number}

    # 1. Define feature mapper
    function feature_mapper(col, name)
        val_to_freq = (min_frequency isa AbstractFloat) ? proportionmap(col) : countmap(col)
        col_type = eltype(col).parameters[1]
        feat_levels = levels(col)

        # Ensure column type is valid (can't test because never occurs)
        # Converting array elements to strings before wrapping in a `CategoricalArray`, as...
        if !(col_type <: supportedtypes)
            throw(ArgumentError(UNSUPPORTED_COL_TYPE(col_type)))
        end

        # Ensure label_for_infrequent keys are valid types
        for possible_col_type in keys(label_for_infrequent)
            if !(possible_col_type in union_types(supportedtypes))
                throw(ArgumentError(VALID_TYPES_NEW_VAL(possible_col_type)))
            end
        end

        # Check no collision between keys(label_for_infrequent) and keys(val_to_freq)
        for value in values(label_for_infrequent)
            if value in keys(val_to_freq)
                throw(ArgumentError(COLLISION_NEW_VAL(value)))
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

        new_cat_given_col_val = Dict{col_type, col_type}()
        for level in feat_levels
            if level in keys(val_to_freq)
                if val_to_freq[level] < min_frequency
                    if elgrandtype in keys(label_for_infrequent)
                        new_cat_given_col_val[level] = label_for_infrequent[elgrandtype]
                    elseif elgrandtype == Number
                        new_cat_given_col_val[level] = minimum(feat_levels) - 1
                    else
                        throw(ArgumentError(UNSPECIFIED_COL_TYPE(col_type, label_for_infrequent)))
                    end
                end
            end
        end
        return new_cat_given_col_val::Dict{col_type, col_type}
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

Apply a fitted cardinality reducer to a table given the output of `cardinality_reducer_fit`

# Arguments

  - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `cache`: The output of `cardinality_reducer_fit`

# Returns

  - `X_tr`: The table with selected features after the selected features are transformed by cardinality reducer
"""
function cardinality_reducer_transform(X, cache::Dict)
    new_cat_given_col_val = cache[:new_cat_given_col_val]
    return generic_transform(X, new_cat_given_col_val; ignore_unknown = true)
end
