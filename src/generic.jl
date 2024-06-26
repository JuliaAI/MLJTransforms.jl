# generic functions go here; such function can be used  throughout multiple methods


"""
A generic function to fit a class transformers where its convenient to define a single `column_mapper` function that 
	takes the column as a vector and potentially other arguments (as passed in ...args and ...kwargs) and returns 
	a dictionary that maps each level of the categorical column to a scalar or vector
	according to the transformation logic. In other words, the `column_mapper` simply answers the question "For level n of
	the current categorical column c, what should the new value or vector (multiple columns) be as defined by the transformation
	logic?"

# Arguments
	- `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
	`Multiclass` or `OrderedFactor`
	- `cols=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
	- `exclude_cols=true`: Whether to exclude or includes the columns given in `cols`
	- `encode_ordinal=false`: Whether to encode `OrderedFactor` or ignore them
	- `column_mapper`: Defined above. 
# Returns
	- `mapping_per_col_level`: Maps each level for each column in a subset of the categorical columns of X into a scalar or a vector. 
	- `encoded_cols`: The subset of the categorical columns of X that were encoded
"""
function generic_fit(
	X,
	cols::AbstractVector{Symbol} = Symbol[],
	args...;
	exclude_cols::Bool = true,
	encode_ordinal::Bool = false,
	column_mapper::Function,
	kwargs...,
)
	# 1. Get X column types and names
	col_names = Tables.schema(X).names

	#2.  Modify column_names based on cols 
	col_names = (exclude_cols) ? setdiff(col_names, cols) : intersect(col_names, cols)

	# 3. Define mapping per column per level dictionary
	mapping_per_col_level = Dict{Symbol, Dict{Any, Any}}()

	# 4. Use column mapper to compute the mapping of each level in each column
	encoded_cols = Symbol[]				# to store column that were actually encoded
	for col_name in col_names
		col = MMI.selectcols(X, col_name)
		col_type = elscitype(col)
		col_has_allowed_type =
			col_type <: Multiclass || (encode_ordinal && col_type <: OrderedFactor)
		if col_has_allowed_type		  # then should be encoded
			push!(encoded_cols, col_name)
			# Compute the dict using the given column_mapper function
			mapping_per_col_level[col_name] = column_mapper(col, args...; kwargs...)
		end
	end
	return mapping_per_col_level, encoded_cols
end



"""
Function to generate new column names: col_name_0, col_name_1,..., col_name_n
"""
function generate_new_column_names(col_name, num_inds)
	return [Symbol("$(col_name)_$i") for i in 1:num_inds]
end


"""
Given a table `X` and a dictionary `mapping_per_col_level` which maps each level for each column in a subset of categorical columns of X into a scalar or a vector (as specified in single_col)
- transforms each value (some level) in each column in `X` using the function in `mapping_per_col_level` into a scalar (single_col=true)
- transforms each value (some level) in each column in `X` using the function in `mapping_per_col_level` into a set of k columns where k is the length of the vector (single_col=false)
* In both cases it attempts to preserve the type of the table. 
* In the latter case, it assumes that all levels under the same category are mapped to vectors of the same length. Such 
    assumption is necessary because any column in X must correspond to a constant number of columns in the output table (which is equal to k).
* Columns not in the dictionary are mapped to themselves (i.e., not changed).
"""
function generic_transform(X, mapping_per_col_level; single_col = true)
	Xr = Tables.rowtable(X)             # for efficient mapping
	col_names = Tables.schema(X).names

	# Dynamically construct the function arguments
	function_arguments = Dict()
	new_col_names = []

	for col in col_names
		# Create the transformation function for each column
		if col in keys(mapping_per_col_level)
			if single_col
				level2scalar = mapping_per_col_level[col]
				# Create a function that returns the target statistics for the given level
				function_arguments[Symbol(col)] = x -> level2scalar[Tables.getcolumn(x, col)]
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
