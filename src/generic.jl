# generic functions go here; such function can be used  throughout multiple methods

"""
Given a table `X` and a dictionary `mapping_per_col_level` which maps each level for each column in a subset of categorical columns of X into a scalar or a vector (as specified in single_col)
- transforms each value (some level) in each column in `X` using the function in `mapping_per_col_level` into a scalar (single_col=true)
- transforms each value (some level) in each column in `X` using the function in `mapping_per_col_level` into a set of k columns where k is the length of the vector (single_col=false)
* In both cases it attempts to preserve the type of the table. 
* In the latter case, it assumes that all levels under the same category are mapped to vectors of the same length. Such 
    assumption is necessary because any column in X must correspond to a constant number of columns in the output table (which is equal to k).
* Columns not in the dictionary are mapped to themselves (i.e., not changed).
"""
function generic_transform(X, col_names,  mapping_per_col_level, single_col = true)
	Xr = Tables.rowtable(X)             # for efficient mapping

	# Dynamically construct the function arguments
	function_arguments = Dict()
	new_col_names = []

	for col in col_names
		# Create the transformation function for each column
		if col in keys(mapping_per_col_level)
			if single_col
				level2float = mapping_per_col_level[col]
				# Create a function that returns the target statistics for the given level
				function_arguments[Symbol(col)] = x -> level2float[Tables.getcolumn(x, col)]
				push!(new_col_names, Symbol(col))
			else
				level2vector = mapping_per_col_level[col]
				col_names_with_inds = generate_new_column_names(col, length(first(mapping_per_col_level[col])[2]))
				# Each column will generate k columns where k is the number of classes
				for (i, col_name_with_ind) in enumerate(col_names_with_inds)
					function_arguments[Symbol(col_name_with_ind)] =
						x -> level2vector[Tables.getcolumn(x, col)][i]
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
	transformation_function =
		x -> NamedTuple{Tuple(new_col_names)}(
			map(name -> function_arguments[name](x), new_col_names),
		)

	# Apply the transformation
	transformed_X = Xr |> TableOperations.map(transformation_function) |> Tables.columntable

	# Attempt to preserve table type
	transformed_X = Tables.materializer(X)(transformed_X)
	return transformed_X
end
