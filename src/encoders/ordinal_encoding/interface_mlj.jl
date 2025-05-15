### OrdinalEncoding with MLJ Interface

# 1. Interface Struct
mutable struct OrdinalEncoder{AS <: AbstractVector{Symbol}} <: Unsupervised
    features::AS
    ignore::Bool
    ordered_factor::Bool
    output_type::Type
end;

# 2. Constructor
function OrdinalEncoder(;
    features = Symbol[],
    ignore = true,
    ordered_factor = false,
    output_type = Float32,
)
    return OrdinalEncoder(features, ignore, ordered_factor, output_type)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::OrdinalEncoder, fitresult) = (
    index_given_feat_level = fitresult,
)

# 5. Fit method
function MMI.fit(transformer::OrdinalEncoder, verbosity::Int, X)
    generic_cache = ordinal_encoder_fit(
        X,
        transformer.features;
        ignore = transformer.ignore,
        ordered_factor = transformer.ordered_factor,
        output_type = transformer.output_type,
    )
    fitresult =
        generic_cache[:index_given_feat_level]
    report = (encoded_features = generic_cache[:encoded_features],)        # report only has list of encoded features
    cache = nothing
    return fitresult, cache, report
end;


# 6. Transform method
function MMI.transform(transformer::OrdinalEncoder, fitresult, Xnew)
    generic_cache = Dict(
        :index_given_feat_level => fitresult,
    )
    Xnew_transf = ordinal_encoder_transform(Xnew, generic_cache)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    OrdinalEncoder,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    OrdinalEncoder,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.OrdinalEncoder",
)




"""
$(MMI.doc_header(OrdinalEncoder))

`OrdinalEncoder` implements ordinal encoding which replaces the categorical values in the specified
    categorical features with integers (ordered arbitrarily). This will create an implicit ordering between
    categories which may not be a proper modelling assumption.

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
- `output_type`: The numerical concrete type of the encoded features. Default is `Float32`.

# Operations

- `transform(mach, Xnew)`: Apply ordinal encoding to selected `Multiclass` or `OrderedFactor features of `Xnew` specified by hyper-parameters, and 
   return the new table.   Features that are neither `Multiclass` nor `OrderedFactor`
   are always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `index_given_feat_level`: A dictionary that maps each level for each column in a subset of the categorical features of X into an integer. 

# Report

The fields of `report(mach)` are:

- `encoded_features`: The subset of the categorical features of X that were encoded

# Examples

```julia
using MLJ

# Define categorical features
A = ["g", "b", "g", "r", "r",]  
B = [1.0, 2.0, 3.0, 4.0, 5.0,]
C = ["f", "f", "f", "m", "f",]  
D = [true, false, true, false, true,]
E = [1, 2, 3, 4, 5,]

# Combine into a named tuple
X = (A = A, B = B, C = C, D = D, E = E)

# Coerce A, C, D to multiclass and B to continuous and E to ordinal
X = coerce(X,
:A => Multiclass,
:B => Continuous,
:C => Multiclass,
:D => Multiclass,
:E => OrderedFactor,
)

# Check scitype coercion:
schema(X)

encoder = OrdinalEncoder(ordered_factor = false)
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)

julia > Xnew
    (A = [2, 1, 2, 3, 3],
    B = [1.0, 2.0, 3.0, 4.0, 5.0],
    C = [1, 1, 1, 2, 1],
    D = [2, 1, 2, 1, 2],
    E = CategoricalArrays.CategoricalValue{Int64, UInt32}[1, 2, 3, 4, 5],)
```

See also
[`TargetEncoder`](@ref)
"""
OrdinalEncoder