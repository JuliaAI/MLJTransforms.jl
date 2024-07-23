### CardinalityReducer with MLJ Interface

# 1. Interface Struct
mutable struct CardinalityReducer{
    AS <: AbstractVector{Symbol},
    R <: Real,
    T <: Type,
    A <: Any,
} <: Unsupervised
    features::AS
    ignore::Bool
    ordered_factor::Bool
    min_frequency::R
    label_for_infrequent::Dict{T, A}
end;

# 2. Constructor
function CardinalityReducer(;
    features = Symbol[],
    ignore = true,
    ordered_factor = false,
    min_frequency = 3,
    label_for_infrequent = Dict(
        AbstractString => "Other",
        Char => 'O',
    ),
)
    return CardinalityReducer(features, ignore, ordered_factor, min_frequency, label_for_infrequent)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::CardinalityReducer, fitresult) = (
    new_cat_given_col_val = fitresult,
)

# 5. Fit method
function MMI.fit(transformer::CardinalityReducer, verbosity::Int, X)
    generic_cache = cardinality_reducer_fit(
        X,
        transformer.features;
        ignore = transformer.ignore,
        ordered_factor = transformer.ordered_factor,
        min_frequency = transformer.min_frequency,
        label_for_infrequent = transformer.label_for_infrequent,
    )
    fitresult = generic_cache[:new_cat_given_col_val]

    report = (encoded_features = generic_cache[:encoded_features],)        # report only has list of encoded columns
    cache = nothing
    return fitresult, cache, report
end;


# 6. Transform method
function MMI.transform(transformer::CardinalityReducer, fitresult, Xnew)
    generic_cache = Dict(
        :new_cat_given_col_val =>
            fitresult,
    )
    Xnew_transf = cardinality_reducer_transform(Xnew, generic_cache)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    CardinalityReducer,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    CardinalityReducer,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.CardinalityReducer",
)




"""
$(MMI.doc_header(CardinalityReducer))

`CardinalityReducer` maps any level of a categorical column that occurs with
frequency < `min_frequency` into a new level (e.g., "Other"). This is useful when some categorical columns have
high cardinality and many levels are infrequent. This assumes that the categorical columns have raw
types that are in `ScientificTypes.SupportedTypes` (e.g., Number, AbstractString, Char).


# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`). Features to be transformed must
   have element scitype `Multiclass` or `OrderedFactor`. Use `schema(X)` to 
   check scitypes. 

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `features=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
- `ignore=true`: Whether to exclude or includes the columns given in `features`
- `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
- `min_frequency::Real=3`: Any level of a categorical column that occurs with frequency < `min_frequency` will be mapped to a new level. Could be
an integer or a float which decides whether raw counts or normalized frequencies are used.
- `label_for_infrequent=Dict{<:Type, <:Any}()= Dict( AbstractString => "Other", Char => 'O', )`: A
dictionary where the possible values for keys are the types in `ScientificTypes.SupportedTypes` and each value signifies
the new level to map into given a column raw super type. By default, if the raw type of the column subtypes `AbstractString`
then the new value is `"Other"` and if the raw type subtypes `Char` then the new value is `'O'`
and if the raw type subtypes `Number` then the new value is the lowest value in the column - 1.

# Operations

- `transform(mach, Xnew)`: Apply cardinality reduction to selected `Multiclass` or `OrderedFactor` features of `Xnew` specified by hyper-parameters, and 
   return the new table.   Features that are neither `Multiclass` nor `OrderedFactor`
   are always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `new_cat_given_col_val`: A dictionary that maps each level in a
    categorical column to a new level (either itself or the new level specified in `label_for_infrequent`)

# Report

The fields of `report(mach)` are:

- `encoded_features`: The subset of the categorical columns of X that were encoded

# Examples

```julia
using StatsBase
using MLJ

# Define categorical columns
A = [ ["a" for i in 1:100]..., "b", "b", "b", "c", "d"]
B = [ [0 for i in 1:100]..., 1, 2, 3, 4, 4]

# Combine into a named tuple
X = (A = A, B = B)

# Coerce A, C, D to multiclass and B to continuous and E to ordinal
X = coerce(X,
:A => Multiclass,
:B => Multiclass
)

encoder = CardinalityReducer(ordered_factor = false, min_frequency=3)
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)

julia> proportionmap(Xnew.A)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Float64} with 3 entries:
  "Other" => 0.0190476
  "b"     => 0.0285714
  "a"     => 0.952381

julia> proportionmap(Xnew.B)
Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Float64} with 2 entries:
  0  => 0.952381
  -1 => 0.047619
```

See also
[`FrequencyEncoder`](@ref)
"""
CardinalityReducer