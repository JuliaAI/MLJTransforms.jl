### TargetEncoding with MLJ Interface

# 1. Interface Struct
mutable struct TargetEncoder{R1 <: Real, R2 <: Real, AS <: AbstractVector{Symbol}} <:
			   Unsupervised
	cols::AS
	exclude_cols::Bool
	encode_ordinal::Bool
	lambda::R1
	m::R2
end;

# 2. Constructor
function TargetEncoder(;
	cols = Symbol[],
	exclude_cols = true,
	encode_ordinal = false,
	lambda = 1.0,
	m = 0,
)
	t = TargetEncoder(cols, exclude_cols, encode_ordinal, lambda, m)
	MMI.clean!(t)
	return t
end;


# 3. Hyperparameter checks
function MMI.clean!(t::TargetEncoder)
	message = ""
	if t.m < 0
		throw(
			ArgumentError(NON_NEGATIVE_m(t.m)),
		)
	end
	if t.lambda < 0 || t.lambda > 1
		throw(
			ArgumentError(INVALID_lambda(t.lambda)),
		)
	end
	return message
end


# 4. Fit result structure (what will be sent to transform)
struct TargetEncoderResult{
	I <: Integer,
	S <: AbstractString,
	# U <: Union{AbstractFloat, AbstractVector{<:AbstractFloat}},  # Unable to keep this after making fit generic (U->Any)
} <: MMI.MLJType
	# target statistic for each level of each categorical column
	y_stat_given_col_level::Dict{Symbol, Dict{Any, Any}}
	task::S            # "Regression", "Classification" 
	num_classes::I     # num_classes in case of classification
end




# 5. Fitted parameters (for user access)
MMI.fitted_params(::TargetEncoder, fitresult) = (
	y_statistic_given_col_level = fitresult.y_stat_given_col_level,
	task = fitresult.task,
)

# 6. Fit method
function MMI.fit(transformer::TargetEncoder, verbosity::Int, X, y)
	fit_res = target_encoder_fit(
		X, y,
		transformer.cols;
		exclude_cols = transformer.exclude_cols,
		encode_ordinal = transformer.encode_ordinal,
		lambda = transformer.lambda,
		m = transformer.m,
	)
	fitresult = TargetEncoderResult(
		fit_res[:y_stat_given_col_level],
		fit_res[:task],
		fit_res[:num_classes],
	)
	report = Dict(:encoded_cols => fit_res[:encoded_cols])        # report only has list of encoded columns
	cache = nothing
	return fitresult, cache, report
end;


# 7. Transform method
function MMI.transform(transformer::TargetEncoder, fitresult, Xnew)
	fit_res = Dict(
		:y_stat_given_col_level =>
			fitresult.y_stat_given_col_level,
		:num_classes => fitresult.num_classes,
		:task => fitresult.task,
	)
	Xnew_transf = target_encoder_transform(Xnew, fit_res)
	return Xnew_transf
end

# 8. Extra metadata
MMI.metadata_pkg(
	TargetEncoder,
	name = "MLJTransforms",
	package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
	package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
	is_pure_julia = true,
)

MMI.metadata_model(
	TargetEncoder,
	input_scitype =
	Tuple{
		Table(Union{Infinite, Finite}),
		AbstractVector,
	},
	output_scitype = Table(Union{Infinite, Finite}),
	load_path = "MLJTransforms.TargetEncoder",
)

function MMI.fit_data_scitype(t::TargetEncoder)
	return Tuple{
		Table(Union{Infinite, Finite}),
		AbstractVector,
	}
end


"""
$(MMI.doc_header(TargetEncoder))

`TargetEncoder` implements target encoding as defined in [1] to encode categorical variables 
	into continuous ones using statistics from the target variable.

In MLJ (or MLJModels) do `model = TargetEncoder()` which is equivalent to `model = TargetEncoder(cols = Symbol[],
	exclude_cols = true,
	encode_ordinal = false,
	lambda = 1.0,
	m = 0,)` to construct a model instance.

# Training data

In MLJ (or MLJBase) bind an instance `model` to data with

	mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`). Categorical columns in this table must have
	scientific types `Multiclass` or `OrderedFactor` for their elements.

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous` or `Count` for regression problems and 
  `Multiclass` or `OrderedFactor` for classification problems; check the scitype with `schema(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `cols=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
- `exclude_cols=true`: Whether to exclude or includes the columns given in `cols`
- `encode_ordinal=false`: Whether to encode `OrderedFactor` or ignore them
- `λ`: Shrinkage hyperparameter used to mix between posterior and prior statistics as described in [1]
- `m`: An integer hyperparameter to compute shrinkage as described in [1]. If `m="auto"` then m will be computed using
 empirical Bayes estimation as described in [1]

# Operations

- `transform(mach, Xnew)`: Apply target encoding to the`Multiclass` or `OrderedFactor` selected columns of `Xnew` and return the new table. 
	Columns that are not `Multiclass` or `OrderedFactor` will be always left unchanged.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `task`: Whether the task is `Classification` or `Regression`
- `y_statistic_given_col_level`: A dictionary with the necessary statistics to encode each categorical column. It maps each 
	level in each categorical column to a statistic computed over the target.

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

# Define the target variable 
y = ["c1", "c2", "c3", "c1", "c2",]

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
y = coerce(y, Multiclass)

encoder = TargetEncoder(encode_ordinal = false, lambda = 1.0, m = 0,)
mach = fit!(machine(encoder, X, y))
Xnew = transform(mach, X)

julia > schema(Xnew)
┌───────┬──────────────────┬─────────────────────────────────┐
│ names │ scitypes         │ types                           │
├───────┼──────────────────┼─────────────────────────────────┤
│ A_1   │ Continuous       │ Float64                         │
│ A_2   │ Continuous       │ Float64                         │
│ A_3   │ Continuous       │ Float64                         │
│ B     │ Continuous       │ Float64                         │
│ C_1   │ Continuous       │ Float64                         │
│ C_2   │ Continuous       │ Float64                         │
│ C_3   │ Continuous       │ Float64                         │
│ D_1   │ Continuous       │ Float64                         │
│ D_2   │ Continuous       │ Float64                         │
│ D_3   │ Continuous       │ Float64                         │
│ E     │ OrderedFactor{5} │ CategoricalValue{Int64, UInt32} │
└───────┴──────────────────┴─────────────────────────────────┘
```

# Reference
[1] Micci-Barreca, Daniele. 
	“A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems” 
	SIGKDD Explor. Newsl. 3, 1 (July 2001), 27–32.

See also
[`OneHotEncoder`](@ref)
"""
TargetEncoder