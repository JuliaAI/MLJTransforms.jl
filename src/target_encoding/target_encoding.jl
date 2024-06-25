"""
Given the targets belonging to a specific category (level) for a categorical variable, 
find the frequency of the positive class (binary classification is assumed).

# Arguments
- `targets_for_level`: An abstract vector containing the target Multiclass or OrderedFactor values
- `y_classes`: An abstract vector containing the classes in the target variable

# Returns
- `freq`: A float for the frequency of the positive class given the category
"""
function compute_label_freq_for_level(targets_for_level, y_classes)
	# Assumes binary classification where the first level is the positive class
	positive_class = y_classes[1]
	freq = sum(targets_for_level .== positive_class) / length(targets_for_level)
	return freq
end

"""
Given the targets belonging to a specific category (level) for a categorical variable, 
find the mean of these values (regression is assumed).

# Arguments
- `targets_for_level`: An abstract vector containing the target Continuous or Count values

# Returns
- `avg`: A float for the mean of the targets given the category
"""
function compute_target_mean_for_level(targets_for_level)
	avg = mean(targets_for_level)
	return avg
end

"""
Given the targets belonging to a specific category (level) for a categorical variable, 
find the frequency of each of the classes (multiclass classification assumed classification is assumed).

# Arguments
- `targets_for_level`: An abstract vector containing the target Multiclass or OrderedFactor values
- `y_classes`: An abstract vector containing the classes in the target variable

# Returns
- `freqs`: A vector of floats for the frequency of the positive class
"""
function compute_label_freqs_for_level(targets_for_level, y_classes)
	# e.g., if y_classes = [1, 2, 3, 4]
	# then get the frequency of occurrence of each in targets_for_level
	freqs = [
		sum(targets_for_level .== y_level) / length(targets_for_level) for
		y_level in y_classes
	]
	return freqs
end

"""
Given the hyperparameter m, compute λ as in [Micci-Barreca, 2001] unless λ was overridden.
"""
function compute_shrinkage(targets_for_level; m = 0, λ = 1.0)
	# If λ has changed from the default then don't recompute
	if m != 0 && λ == 1.0
		ni = length(targets_for_level)
		λ = ni / (m + ni)
	end
	return λ
end

"""
Compute m automatically using empirical Bayes estimation as suggested in [Micci-Barreca, 2001]. 
Only possible for regression tasks
"""
function compute_m_auto(task, targets_for_level; y_var)
	# call if m="auto" to compute by empirical Bayes est.
	task !== "Regression" && error("m = `auto` is only supported for regression")
	y_var_level = std(targets_for_level)^2
	m = y_var / y_var_level
	return m
end

"""
Implement mixing between a posterior and a prior statistic  by computing `λ * posterior + (1 - λ) * prior`
"""
function mix_stats(; posterior, prior, λ)
	# mixing prior and posterior with mixing factor λ
	return λ .* posterior .+ (1 - λ) .* prior
	# prior is like the frequency of the positive class over the whole data
	# posterior is the frequency given rows that have the specific catefory
end


"""
	target_encoding_fit(X, y, cols=[]; exclude_cols=true, encode_ordinal=false, λ = 1.0, m=0)

Fit a target encoder on table X with target y by computing the necessary statistics for every categorical column.

# Arguments
- `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
`Multiclass` or `OrderedFactor`
- `y`:  An abstract vector of labels (e.g., strings) that correspond to the observations in X
- `cols=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
- `exclude_cols=true`: Whether to exclude or includes the columns given in `cols`
- `encode_ordinal=false`: Whether to encode `OrderedFactor` or ignore them
- `λ`: Shrinkage hyperparameter used to mix between posterior and prior statistics as described in [1]
- `m`: An integer hyperparameter to compute shrinkage as described in [1]. If `m="auto"` then m will be computed using
 empirical Bayes estimation as described in [1]

# Returns
- `cache`: A dictionary containing a dictionary `y_stat_given_col_level` with the necessary statistics needed to transform
every categorical column as well as other metadata needed for transform.
"""
function target_encoding_fit(
	X,
	y::AbstractVector,
	cols::AbstractVector{Symbol} = Symbol[];
	exclude_cols::Bool = true,
	encode_ordinal::Bool = false,
	lambda::Real= 1.0, 
	m::Real = 0,
)
	# Get X column types and names
	col_names = Tables.schema(X).names
	

	# Modify column_names based on cols 
	col_names = (exclude_cols) ? setdiff(col_names, cols) : intersect(col_names, cols)

	# Figure out task (classification or regression)
	y_scitype = elscitype(y)
	classification_types = (y_scitype <: Multiclass || y_scitype <: OrderedFactor)
	regression_types = (y_scitype <: Continuous || y_scitype <: Count)
	task =
		regression_types ? "Regression" :
		classification_types ? "Classification" : "Unknown"
	task == "Unknown" && error(
		"Your target must be Continuous/Count for regression or Multiclass/OrderedFactor for classification",
	)

	# Setup prior statistics and structures for posterior statistics
	if task == "Regression"
		y_stat_given_col_level = Dict{Symbol, Dict{Any, AbstractFloat}}()
		y_mean = mean(y)                             # for mixing
		m == "auto" && (y_var = std(y)^2)              # for empirical Bayes estimation
	else
		y_classes = levels(y)
		is_multiclass = length(y_classes) > 2
		if !is_multiclass       # binary case
			y_stat_given_col_level = Dict{Symbol, Dict{Any, AbstractFloat}}()
			y_prior = sum(y .== y_classes[1]) / length(y)   # for mixing
		else                    # multiclass case
			y_stat_given_col_level =
				Dict{Symbol, Dict{Any, AbstractVector{AbstractFloat}}}()
			y_priors = [sum(y .== y_level) / length(y) for y_level in y_classes]    # for mixing
		end
	end

	# Loop on each column and gether per level
	encoded_cols = Symbol[]
	for col_name in col_names
		col = MMI.selectcols(X, col_name)
		col_type = elscitype(col)
		col_has_allowed_type =
			col_type <: Multiclass || (encode_ordinal && col_type <: OrderedFactor)
		if col_has_allowed_type
			push!(encoded_cols, col_name)
			for level in levels(col)
				# Initialize dict of levels for the feature
				y_stat_given_col_level[col_name] =
					get(y_stat_given_col_level, col_name, Dict{Any, Float64}())
				# Get the targets of an example that belong to this level
				targets_for_level = y[col.==level]

				# Compute λ for mixing
				m == "auto" && (m = compute_m_auto(task, targets_for_level; y_var = y_var))
				lambda = compute_shrinkage(targets_for_level; m = m, λ = lambda)

				if task == "Classification"
					if !is_multiclass           # binary classification
						y_freq_for_level =
							compute_label_freq_for_level(targets_for_level, y_classes)
						y_stat_given_col_level[col_name][level] =
							mix_stats(posterior = y_freq_for_level, prior = y_prior, λ = lambda)
					else                        # multiclass classification
						y_freqs_for_level =
							compute_label_freqs_for_level(targets_for_level, y_classes)
						y_stat_given_col_level[col_name][level] = mix_stats(
							posterior = y_freqs_for_level,
							prior = y_priors,
							λ = lambda,
						)
					end
				else                            # regression
					y_mean_for_level = compute_target_mean_for_level(targets_for_level)
					y_stat_given_col_level[col_name][level] =
						mix_stats(posterior = y_mean_for_level, prior = y_mean, λ = lambda)
				end
			end
		end
	end
	cache = Dict(
		:task => task,
		:num_classes => (task == "Regression") ? -1 : length(y_classes),
		:y_stat_given_col_level => y_stat_given_col_level,
		:encoded_cols => encoded_cols
	)
	return cache
end


"""
Function to generate new column names: col_name_0, col_name_1,..., col_name_n
"""
function generate_new_column_names(col_name, num_inds)
    return [Symbol("$(col_name)_$i") for i in 1:num_inds]
end


"""
	transformit(X, cache)

Transform given data with fitted target encoder cache.

# Arguments
- `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
`Multiclass` or `OrderedFactor`
- `cache`: A dictionary containing a dictionary `y_stat_given_col_level` with the necessary statistics for 
every categorical column as well as other metadata needed for transform

# Returns
- `X`: A table where the categorical columns as specified during fitting are transformed by target encoding. Other columns will remain
    the same. This will attempt to preserve the type of the table but may not succeed. 
"""

function transformit(X, cache)
    col_names = Tables.schema(X).names
    task = cache[:task]
    y_stat_given_col_level = cache[:y_stat_given_col_level]
    num_classes = cache[:num_classes]

    
    Xr = Tables.rowtable(X)             # for efficient mapping

    # Dynamically construct the function arguments
    function_arguments = Dict()
    new_col_names = []

    for col in col_names
        # Create the transformation function for each column
        if col in keys(y_stat_given_col_level)
            if task == "Regression" || (task == "Classification" && num_classes < 3)
                level2float = y_stat_given_col_level[col]
                # Create a function that returns the target statistics for the given level
                function_arguments[Symbol(col)] = x -> level2float[Tables.getcolumn(x, col)]
                push!(new_col_names, Symbol(col))
            else    # Multiclassification
                level2vector = y_stat_given_col_level[col]
                col_names_with_inds = generate_new_column_names(col, num_classes)
                # Each column will generate k columns where k is the number of classes
                for (i, col_name_with_ind) in enumerate(col_names_with_inds)
                    function_arguments[Symbol(col_name_with_ind)] = x -> level2vector[Tables.getcolumn(x, col)][i]
                    push!(new_col_names, Symbol(col_name_with_ind))
                end
            end
        # Not to be transformed => left as is
        else
            function_arguments[Symbol(col)] = x -> Tables.getcolumn(x, col)
            push!(new_col_names, Symbol(col))
        end
    end

    # Create the transformation function arguments from the dict
    transformation_function = x -> NamedTuple{Tuple(new_col_names)}(map(name -> function_arguments[name](x), new_col_names))

    # Apply the transformation
    transformed_X = Xr |> TableOperations.map(transformation_function) |> Tables.columntable

    # Attempt to preserve table type
    transformed_X = Tables.materializer(X)(transformed_X)
    return transformed_X
end