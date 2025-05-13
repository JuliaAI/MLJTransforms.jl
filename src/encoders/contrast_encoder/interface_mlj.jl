### ContrastEncoding with MLJ Interface

# 1. Interface Struct
mutable struct ContrastEncoder{AS <: AbstractVector{Symbol}} <: Unsupervised
    features::AS
    ignore::Bool
    mode::Union{Symbol, AS}
    buildmatrix::Any
    ordered_factor::Bool
end;

# 2. Constructor
function ContrastEncoder(;
    features = Symbol[],
    ignore = true,
    mode = :dummy,
    buildmatrix = nothing,
    ordered_factor = false,
)
    return ContrastEncoder(features, ignore, mode, buildmatrix, ordered_factor)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::ContrastEncoder, fitresult) = (
    vector_given_value_given_feature = fitresult,
)

# 5. Fit method
function MMI.fit(transformer::ContrastEncoder, verbosity::Int, X)
    generic_cache = contrast_encoder_fit(
        X,
        transformer.features;
        ignore = transformer.ignore,
        mode = transformer.mode,
        buildmatrix = transformer.buildmatrix,
        ordered_factor = transformer.ordered_factor,
    )
    fitresult = generic_cache[:vector_given_value_given_feature]

    report = (encoded_features = generic_cache[:encoded_features],)        # report only has list of encoded features
    cache = nothing
    return fitresult, cache, report
end;


# 6. Transform method
function MMI.transform(transformer::ContrastEncoder, fitresult, Xnew)
    generic_cache = Dict(
        :vector_given_value_given_feature =>
            fitresult,
    )
    Xnew_transf = contrast_encoder_transform(Xnew, generic_cache)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    ContrastEncoder,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    ContrastEncoder,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.ContrastEncoder",
)


"""
$(MMI.doc_header(ContrastEncoder))

`ContrastEncoder` implements the following contrast encoding methods for 
categorical features: dummy, sum, backward/forward difference, and Helmert coding. 
More generally, users can specify a custom contrast or hypothesis matrix, and each feature 
can be encoded using a different method.

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
- `mode=:dummy`: The type of encoding to use. Can be one of `:contrast`, `:dummy`, `:sum`, `:backward_diff`, `:forward_diff`, `:helmert` or `:hypothesis`.
If `ignore=false` (features to be encoded are listed explictly in `features`), then this can be a vector of the same length as `features` to specify a different
contrast encoding scheme for each feature
- `buildmatrix=nothing`: A function or other callable with signature `buildmatrix(colname, k)`, 
where `colname` is the name of the feature levels and `k` is it's length, and which returns contrast or 
hypothesis matrix with row/column ordering consistent with the ordering of `levels(col)`. Only relevant if `mode` is `:contrast` or `:hypothesis`.
- `ignore=true`: Whether to exclude or includes the features given in `features`
- `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them

# Operations

- `transform(mach, Xnew)`: Apply contrast encoding to selected `Multiclass` or `OrderedFactor features of `Xnew` specified by hyper-parameters, and 
   return the new table. Features that are neither `Multiclass` nor `OrderedFactor`
   are always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `vector_given_value_given_feature`: A dictionary that maps each level for each column in a subset of the categorical features of X into its frequency.

# Report

The fields of `report(mach)` are:

- `encoded_features`: The subset of the categorical features of X that were encoded

# Examples

```julia
using MLJ

# Define categorical dataset
X = (
    name   = categorical(["Ben", "John", "Mary", "John"]),
    height = [1.85, 1.67, 1.5, 1.67],
    favnum = categorical([7, 5, 10, 1]),
    age    = [23, 23, 14, 23],
)

# Check scitype coercions:
schema(X)

encoder =  ContrastEncoder(
    features = [:name, :favnum],
    ignore = false, 
    mode = [:dummy, :helmert],
)
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)

julia > Xnew
    (name_John = [1.0, 0.0, 0.0, 0.0],
    name_Mary = [0.0, 1.0, 0.0, 1.0],
    height = [1.85, 1.67, 1.5, 1.67],
    favnum_5 = [0.0, 1.0, 0.0, -1.0],
    favnum_7 = [2.0, -1.0, 0.0, -1.0],
    favnum_10 = [-1.0, -1.0, 3.0, -1.0],
    age = [23, 23, 14, 23],)
```

See also
[`OneHotEncoder`](@ref)
"""
ContrastEncoder