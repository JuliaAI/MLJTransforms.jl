
"""
Fit an encoder to encode the levels of categorical variables in a given table as integers (ordered arbitrarily).

# Arguments
 - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
 - `cols=[]`: A list of names of categorical columns given as symbols to exclude or include from encoding
 - `exclude_cols=true`: Whether to exclude or includes the columns given in `cols`
 - `encode_ordinal=false`: Whether to encode `OrderedFactor` or ignore them

# Returns (in a dict)
 - `index_given_col_level`: Maps each level for each column in a subset of the categorical columns of X into an integer. 
 - `encoded_cols`: The subset of the categorical columns of X that were encoded
"""
function ordinal_encoder_fit(
	X,
	cols::AbstractVector{Symbol} = Symbol[];
	exclude_cols::Bool = true,
	encode_ordinal::Bool = false,
)
	# 1. Define column mapper
	function column_mapper(col)
		col_levels = levels(col)
		index_given_col_val = Dict{Any, Integer}(value => index for (index, value) in enumerate(col_levels))
        return index_given_col_val 
	end
	
	# 2. Pass it to generic_fit
	index_given_col_level, encoded_cols = generic_fit(
		X, cols; exclude_cols = exclude_cols, encode_ordinal = encode_ordinal, column_mapper = column_mapper
	)
	cache = Dict(
		:index_given_col_level => index_given_col_level,
		:encoded_cols => encoded_cols,
	)
	return cache
end


"""
Encode the levels of a categorical variable in a given table as integers.

# Arguments
 - `X`: A table where the elements of the categorical columns have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) `Multiclass` or `OrderedFactor`
 - `cache`: The output of `ordinal_encoder_fit`

 # Returns
 - `X_tr`: The table with selected columns after the selected columns are encoded by ordinal encoding.

"""
function ordinal_encoder_transform(X, cache::Dict)
	index_given_col_level = cache[:index_given_col_level]
	return generic_transform(X,  index_given_col_level)
end