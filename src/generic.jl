# generic functions go here; such function can be used  throughout multiple methods


"""
**Private method.**

A generic function to fit a class of transformers where its convenient to define a single `feature_mapper` function that
takes the column as a vector and potentially other arguments (as passed in ...args and ...kwargs) and returns
a dictionary that maps each level of the categorical column to a scalar or vector
according to the transformation logic. In other words, the `feature_mapper` simply answers the question "For level n of
the current categorical column c, what should the new value or vector (multiple columns) be as defined by the transformation
logic?"

# Arguments

	- `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
	`Multiclass` or `OrderedFactor`
	- `features=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
	- `ignore=true`: Whether to exclude or includes the columns given in `features`
	- `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
	- `feature_mapper`: Defined above. 

# Returns

	- `mapping_per_feat_level`: Maps each level for each column in a subset of the categorical columns of
	 X into a scalar or a vector. 
	- `encoded_features`: The subset of the categorical columns of X that were encoded
"""
function generic_fit(X,
	features::AbstractVector{Symbol} = Symbol[],
	args...;
	ignore::Bool = true,
	ordered_factor::Bool = false,
	feature_mapper,
	kwargs...,
)
	# 1. Get X column types and names
	feat_names = Tables.schema(X).names

	#2.  Modify column_names based on features 
	feat_names = (ignore) ? setdiff(feat_names, features) : intersect(feat_names, features)

	# 3. Define mapping per column per level dictionary
	mapping_per_feat_level = Dict{Symbol, Dict{Any, Any}}()

	# 4. Use column mapper to compute the mapping of each level in each column
	encoded_features = Symbol[]# to store column that were actually encoded
	for feat_name in feat_names
		col = Tables.getcolumn(X, feat_name)
		feat_type = elscitype(col)
		feat_has_allowed_type =
			feat_type <: Multiclass || (ordered_factor && feat_type <: OrderedFactor)
		if feat_has_allowed_type  # then should be encoded
			push!(encoded_features, feat_name)
			# Compute the dict using the given feature_mapper function
			mapping_per_feat_level[feat_name] = feature_mapper(col, args...; kwargs...)
		end
	end
	return mapping_per_feat_level, encoded_features
end



"""
**Private method.**

Function to generate new column names: feat_name_0, feat_name_1,..., feat_name_n
"""
function generate_new_feat_names(feat_name, num_inds, existing_names)
    conflict = true		# will be kept true as long as there is a conflict
    count = 1			# number of conflicts+1 = number of underscores

	new_column_names = []
    while conflict
        suffix = repeat("_", count)  
        new_column_names = [Symbol("$(feat_name)$(suffix)$i") for i in 1:num_inds]
        conflict = any(name -> name in existing_names, new_column_names)
        count += 1
    end
    return new_column_names
end



"""
**Private method.**

Given a table `X` and a dictionary `mapping_per_feat_level` which maps each level for each column in 
a subset of categorical columns of X into a scalar or a vector (as specified in single_feat)

  - transforms each value (some level) in each column in `X` using the function in `mapping_per_feat_level` 
  into a scalar (single_feat=true)

  - transforms each value (some level) in each column in `X` using the function in `mapping_per_feat_level` 
  into a set of k columns where k is the length of the vector (single_feat=false)
  - In both cases it attempts to preserve the type of the table.
  - In the latter case, it assumes that all levels under the same category are mapped to vectors of the same length. Such
	assumption is necessary because any column in X must correspond to a constant number of columns 
	in the output table (which is equal to k).
  - Columns not in the dictionary are mapped to themselves (i.e., not changed).
"""
function generic_transform(X, mapping_per_feat_level; single_feat = true)
	Xr = Tables.rowtable(X)             # for efficient mapping
	feat_names = Tables.schema(X).names

	# Dynamically construct the function arguments
	function_arguments = Dict()
	new_feat_names = []

	for col in feat_names
		# Create the transformation function for each column
		if col in keys(mapping_per_feat_level)
			if single_feat
				level2scalar = mapping_per_feat_level[col]
				# Create a function that returns the target statistics for the given level
				function_arguments[Symbol(col)] =
					x -> level2scalar[Tables.getcolumn(x, col)]
				push!(new_feat_names, Symbol(col))
			else
				level2vector = mapping_per_feat_level[col]
				feat_names_with_inds = generate_new_feat_names(
					col,
					length(first(mapping_per_feat_level[col])[2]),
					feat_names,
				)
				# Each column will generate k columns where k is the number of classes
				for (i, feat_name_with_ind) in enumerate(feat_names_with_inds)
					function_arguments[Symbol(feat_name_with_ind)] =
						x -> level2vector[Tables.getcolumn(x, col)][i]
					push!(new_feat_names, Symbol(feat_name_with_ind))
				end
			end
			# Not to be transformed => left as is
		else
			function_arguments[Symbol(col)] = x -> Tables.getcolumn(x, col)
			push!(new_feat_names, Symbol(col))
		end
	end

	# Create the transformation function arguments from the dict
	transformation_function =
		x -> NamedTuple{Tuple(new_feat_names)}(
			map(name -> function_arguments[name](x), new_feat_names),
		)

	# Apply the transformation
	transformed_X = Xr |> TableOperations.map(transformation_function) |> Tables.columntable

	# Attempt to preserve table type
	transformed_X = Tables.materializer(X)(transformed_X)
	return transformed_X
end
