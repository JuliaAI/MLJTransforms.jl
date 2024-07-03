
include("errors.jl")
# To go from e.g., Union{Integer, String} to (Integer, String)
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

"""
Fit a transformer that maps any level of a categorical column that occurs with
frequency < `min_freq` into a new level (e.g., "Other"). This is useful when some categorical columns have
high cardinality and many levels are infrequent. This assumes that the categorical columns have raw
types that are in `ScientificTypes.SupportedTypes` (e.g., Number, AbstractString, Char).

# Arguments

  - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `features=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
  - `ignore=true`: Whether to exclude or includes the columns given in `features`
  - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
  - `min_freq::Real=3`: Any level of a categorical column that occurs with frequency < `min_freq` will be mapped to a new level. Could be
    an integer or a float which decides whether raw counts or normalized frequencies are used.
  - `infreq_val=Dict{<:Type, <:Any}()= Dict( AbstractString => "Other", Char => 'O', )`: A
    dictionary where the possible values for keys are the types in `ScientificTypes.SupportedTypes` and the values
    are the new level to map into for each raw super type. By default, if the raw type of the column subtypes `AbstractString`
    then the new value is `"Other"` and if the raw type subtypes `Char` then the new value is `'O'`
    and if the raw type subtypes `Number` then the new value is the lowest value in the column - 1.

# Returns (in a dict)

  - `new_cat_given_col_val`: A dictionary that maps each level in a
    categorical column to a new level (either itself or the new level specified in `infreq_val`)
  - `encoded_features`: The subset of the categorical columns of X that were encoded
"""
function cardinality_reducer_fit(
    X,
    features::AbstractVector{Symbol} = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
    min_freq::Real = 3,
    infreq_val::Dict{<:Type, <:Any} = Dict(
        AbstractString => "Other",
        Char => 'O',
    ),
)

    # 1. Define column mapper
    function feature_mapper(col)
        val_to_freq = (min_freq isa AbstractFloat) ? proportionmap(col) : countmap(col)
        col_type = eltype(col).parameters[1]

        # Ensure column type is valid (can't test because never occurs)
        # Converting array elements to strings before wrapping in a `CategoricalArray`, as `Object{Int64}` unsupported by CategoricalArrays. 
        if !(col_type <: ScientificTypes.SupportedTypes)
            throw(ArgumentError(UNSUPPORTED_COL_TYPE(col_type)))
        end

        # Ensure infreq_val keys are valid types
        for possible_col_type in keys(infreq_val)
            if !(possible_col_type in union_types(ScientificTypes.SupportedTypes))
                throw(ArgumentError(VALID_TYPES_NEW_VAL(possible_col_type)))
            end
        end

        # Check no collision between keys(infreq_val) and keys(val_to_freq)
        for value in values(infreq_val)
            if value in keys(val_to_freq)
                throw(ArgumentError(COLLISION_NEW_VAL(value)))
            end
        end

        # Get ancestor type of column
        elgrandtype = nothing
        for allowed_type in union_types(ScientificTypes.SupportedTypes)
            if col_type <: allowed_type
                elgrandtype = allowed_type
                break
            end
        end

        new_cat_given_col_val = Dict{eltype(col), eltype(col)}()
        for (level, freq) in val_to_freq
            if freq >= min_freq
                new_cat_given_col_val[level] = level
            else
                if elgrandtype in keys(infreq_val)
                    new_cat_given_col_val[level] = categorical([infreq_val[elgrandtype]])[1]
                elseif elgrandtype == Number
                    new_cat_given_col_val[level] =
                        categorical([convert(Integer, minimum(col)) - 1])[1]
                else
                    throw(ArgumentError(UNSPECIFIED_COL_TYPE(col_type, infreq_val)))
                end
            end
        end
        return new_cat_given_col_val
    end

    # 2. Pass it to generic_fit
    new_cat_given_col_val, encoded_features = generic_fit(
        X, features; ignore = ignore, feature_mapper = feature_mapper,
    )
    cache = Dict(
        :new_cat_given_col_val => new_cat_given_col_val,
        :encoded_features => encoded_features,
    )
    return cache
end


"""
Apply a fitted `cardinality_reducer_fit` to a table.

# Arguments

  - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/)
    `Multiclass` or `OrderedFactor`
  - `cache`: The output of `cardinality_reducer_fit`

# Returns

  - `X_tr`: The table with selected columns after the selected columns are transformed by cardinality reducer
"""
function cardinality_reducer_transform(X, cache::Dict)
    new_cat_given_col_val = cache[:new_cat_given_col_val]
    return generic_transform(X, new_cat_given_col_val)
end
