### FrequencyEncoding with MLJ Interface

# 1. Interface Struct
mutable struct FrequencyEncoder{A <: Any} <: Unsupervised
    features::A
    ignore::Bool
    ordered_factor::Bool
    normalize::Bool
    output_type::Type
end;

# 2. Constructor
function FrequencyEncoder(;
    features = Symbol[],
    ignore = true,
    ordered_factor = false,
    normalize = false,
    output_type = Float32,
)
    return FrequencyEncoder(features, ignore, ordered_factor, normalize, output_type)
end;


# 4. Fitted parameters (for user access)
MMI.fitted_params(::FrequencyEncoder, fitresult) = (
    statistic_given_feat_val = fitresult,
)

# 5. Fit method
function MMI.fit(transformer::FrequencyEncoder, verbosity::Int, X)
    generic_cache = frequency_encoder_fit(
        X,
        transformer.features;
        ignore = transformer.ignore,
        ordered_factor = transformer.ordered_factor,
        normalize = transformer.normalize,
        output_type = transformer.output_type,
    )
    fitresult = generic_cache.statistic_given_feat_val

    report = (encoded_features = generic_cache.encoded_features,)        # report only has list of encoded features
    cache = nothing
    return fitresult, cache, report
end;


# 6. Transform method
function MMI.transform(transformer::FrequencyEncoder, fitresult, Xnew)
    generic_cache = (
        statistic_given_feat_val = fitresult,
    )
    Xnew_transf = frequency_encoder_transform(Xnew, generic_cache)
    return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
    FrequencyEncoder,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    FrequencyEncoder,
    input_scitype = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.FrequencyEncoder",
)




"""
$(MMI.doc_header(FrequencyEncoder))

`FrequencyEncoder` implements frequency encoding which replaces the categorical values in the specified
    categorical features with their (normalized or raw) frequencies of occurrence in the dataset. 

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

    mach = machine(model, X)

Here:

$X_doc_mlj

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

$features_doc
$ignore_doc
$ordered_factor_doc
- ` normalize=false`: Whether to use normalized frequencies that sum to 1 over category values or to use raw counts.
- `output_type=Float32`: The type of the output values. The default is `Float32`, but you can set it to `Float64` or any other type that can hold the frequency values.

# Operations

- `transform(mach, Xnew)`: Apply frequency encoding to selected `Multiclass` or `OrderedFactor features of `Xnew` specified by hyper-parameters, and 
   return the new table.   Features that are neither `Multiclass` nor `OrderedFactor`
   are always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `statistic_given_feat_val`: A dictionary that maps each level for each column in a subset of the categorical features of X into its frequency.

# Report

The fields of `report(mach)` are:

$encoded_features_doc

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

# Check scitype coercions:
schema(X)

encoder = FrequencyEncoder(ordered_factor = false, normalize=true)
mach = fit!(machine(encoder, X))
Xnew = transform(mach, X)

julia > Xnew
    (A = [2, 1, 2, 2, 2],
    B = [1.0, 2.0, 3.0, 4.0, 5.0],
    C = [4, 4, 4, 1, 4],
    D = [3, 2, 3, 2, 3],
    E = CategoricalArrays.CategoricalValue{Int64, UInt32}[1, 2, 3, 4, 5],)
```

See also
[`TargetEncoder`](@ref)
"""
FrequencyEncoder