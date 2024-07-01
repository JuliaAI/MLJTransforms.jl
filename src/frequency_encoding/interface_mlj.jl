### TargetEncoding with MLJ Interface

# 1. Interface Struct
mutable struct FrequencyEncoder{AS <: AbstractVector{Symbol}} <: Unsupervised
	cols::AS
	exclude_cols::Bool
	encode_ordinal::Bool
	normalize::Bool
end;

# 2. Constructor
function FrequencyEncoder(;
	cols = Symbol[],
	exclude_cols = true,
	encode_ordinal = false,
	normalize = false,
)
	return FrequencyEncoder(cols, exclude_cols, encode_ordinal, normalize)
end;



# 4. Fit result structure (what will be sent to transform)
struct FrequencyEncoderResult <: MMI.MLJType
	# target statistic for each level of each categorical column
	statistic_given_col_val::Dict{Symbol, Dict{Any, Any}}
end

# 5. Fitted parameters (for user access)
MMI.fitted_params(::FrequencyEncoder, fitresult) = (
	statistic_given_col_val = fitresult.statistic_given_col_val
)

# 6. Fit method
function MMI.fit(transformer::FrequencyEncoder, verbosity::Int, X)
	fit_res = frequency_encoder_fit(
		X,
		transformer.cols;
		exclude_cols = transformer.exclude_cols,
		encode_ordinal = transformer.encode_ordinal,
		normalize = transformer.normalize,
	)
	fitresult = FrequencyEncoderResult(
		fit_res[:statistic_given_col_val],
	)
	report = Dict(:encoded_cols => fit_res[:encoded_cols])        # report only has list of encoded columns
	cache = nothing
	return fitresult, cache, report
end;


# 7. Transform method
function MMI.transform(transformer::FrequencyEncoder, fitresult, Xnew)
	fit_res = Dict(
		:statistic_given_col_val =>
			fitresult.statistic_given_col_val,
	)
	Xnew_transf = frequency_encoder_transform(Xnew, fit_res)
	return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
	FrequencyEncoder,
	name = "MLJTransforms",
	package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
	package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
	is_pure_julia = true,
)

MMI.metadata_model(
	FrequencyEncoder,
	input_scitype = Table(Union{Infinite, Finite}),
	output_scitype = Table(Union{Infinite, Finite}),
	load_path = "MLJTransforms.FrequencyEncoder",
)




"""
$(MMI.doc_header(FrequencyEncoder))

`FrequencyEncoder` implements frequency encoding which replaces the categorical values in the specified
	categorical columns with their (normalized or raw) frequencies of occurrence in the dataset. 

In MLJ (or MLJModels) do `model = FrequencyEncoder()` which is equivalent to `model = FrequencyEncoder(cols = Symbol[],
	exclude_cols = true,
	encode_ordinal = false, 
	normalize false
	)` to construct a model instance.

# Training data

In MLJ (or MLJBase) bind an instance unsupervised `model` to data with

	mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`). Categorical columns in this table must have
	scientific types `Multiclass` or `OrderedFactor` for their elements.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `cols=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
- `exclude_cols=true`: Whether to exclude or includes the columns given in `cols`
- `encode_ordinal=false`: Whether to encode `OrderedFactor` or ignore them
- `normalize=false`: Whether to use normalized frequencies that sum to 1 over category values or to use raw counts.

# Operations

- `transform(mach, Xnew)`: Apply target encoding to the`Multiclass` or `OrderedFactor` selected columns of `Xnew` and return the new table. 
	Columns that are not `Multiclass` or `OrderedFactor` will be always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `statistic_given_col_val`: A dictionary that maps each level for each column in a subset of the categorical columns of X into its frequency.

# Examples

```julia
using ScientificTypes
using MLJTransforms
using MLJ

# Define categorical columns
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

encoder = FrequencyEncoder(encode_ordinal = false, normalize=true)
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