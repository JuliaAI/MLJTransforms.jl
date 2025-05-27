### MissingnessEncoder with MLJ Interface

# 1. Interface Struct
mutable struct MissingnessEncoder{
    T <: Type,
    A <: Any,
} <: Unsupervised
    features::A
    ignore::Bool
    ordered_factor::Bool
    label_for_missing::Dict{T, A}
end;

# 2. Constructor
function MissingnessEncoder(;
    features = Symbol[],
    ignore = true,
    ordered_factor = false,
    label_for_missing = Dict(
        AbstractString => "missing",
        Char => 'm',
    ),
)
    return MissingnessEncoder(features, ignore, ordered_factor, label_for_missing)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::MissingnessEncoder, fitresult) = (
    label_for_missing_given_feature = fitresult,
)

# 5. Fit method
function MMI.fit(transformer::MissingnessEncoder, verbosity::Int, X)
    generic_cache = missingness_encoder_fit(
        X,
        transformer.features;
        ignore = transformer.ignore,
        ordered_factor = transformer.ordered_factor,
        label_for_missing = transformer.label_for_missing,
    )
    fitresult = generic_cache[:label_for_missing_given_feature]

    report = (encoded_features = generic_cache[:encoded_features],)        # report only has list of encoded features
    cache = nothing
    return fitresult, cache, report
end;


# 6. Transform method
function MMI.transform(transformer::MissingnessEncoder, fitresult, Xnew)
    generic_cache = Dict(
        :label_for_missing_given_feature =>
            fitresult,
    )
    Xnew_transf = missingness_encoder_transform(Xnew, generic_cache)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    MissingnessEncoder,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    MissingnessEncoder,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.MissingnessEncoder",
)



"""
$(MMI.doc_header(MissingnessEncoder))

`MissingnessEncoder` maps any missing level of a categorical feature into a new level (e.g., "Missing"). 
By this, missingness will be treated as a new
level by any subsequent model. This assumes that the categorical features have raw
types that are in `Char`, `AbstractString`, and `Number`.

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`). Features to be transformed must
   have element scitype `Multiclass` or `OrderedFactor`. Use `schema(X)` to 
   check scitypes. 

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
- `ignore=true`: Whether to exclude or includes the features given in `features`
- `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
- `label_for_missing::Dict{<:Type, <:Any}()= Dict( AbstractString => "missing", Char => 'm', )`: A
dictionary where the possible values for keys are the types in `Char`, `AbstractString`, and `Number` and where each value
signifies the new level to map into given a column raw super type. By default, if the raw type of the column subtypes `AbstractString`
then missing values will be replaced with `"missing"` and if the raw type subtypes `Char` then the new value is `'m'`
and if the raw type subtypes `Number` then the new value is the lowest value in the column - 1.

# Operations

- `transform(mach, Xnew)`: Apply cardinality reduction to selected `Multiclass` or `OrderedFactor` features of `Xnew` specified by hyper-parameters, and 
   return the new table.   Features that are neither `Multiclass` nor `OrderedFactor`
   are always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `label_for_missing_given_feature`: A dictionary that for each column, maps `missing` into some value according to `label_for_missing`

# Report

The fields of `report(mach)` are:

- `encoded_features`: The subset of the categorical features of X that were encoded

# Examples

```julia
import StatsBase.proportionmap
using MLJ

# Define a table with missing values
Xm = (
    A = categorical(["Ben", "John", missing, missing, "Mary", "John", missing]),
    B = [1.85, 1.67, missing, missing, 1.5, 1.67, missing],
    C= categorical([7, 5, missing, missing, 10, 0, missing]),
    D = [23, 23, 44, 66, 14, 23, 11],
    E = categorical([missing, 'g', 'r', missing, 'r', 'g', 'p'])
)

encoder = MissingnessEncoder()
mach = fit!(machine(encoder, Xm))
Xnew = transform(mach, Xm)

julia> Xnew
(A = ["Ben", "John", "missing", "missing", "Mary", "John", "missing"],
 B = Union{Missing, Float64}[1.85, 1.67, missing, missing, 1.5, 1.67, missing],
 C = [7, 5, -1, -1, 10, 0, -1],
 D = [23, 23, 44, 66, 14, 23, 11],
 E = ['m', 'g', 'r', 'm', 'r', 'g', 'p'],)

```

See also
[`CardinalityReducer`](@ref)
"""
MissingnessEncoder