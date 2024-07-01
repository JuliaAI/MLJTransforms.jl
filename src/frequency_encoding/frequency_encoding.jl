function frequency_encoder_fit(
	X,
	cols::AbstractVector{Symbol} = Symbol[];
	exclude_cols::Bool = true,
	encode_ordinal::Bool = false,
	normalize::Bool = false,
)
	# 1. Define column mapper
	function column_mapper(col)
		statistic_given_col_val = (!normalize) ? countmap(col) : proportionmap(col)
		return statistic_given_col_val
	end

	# 2. Pass it to generic_fit
	statistic_given_col_val, encoded_cols = generic_fit(
		X, cols; exclude_cols = exclude_cols, encode_ordinal = encode_ordinal,
		column_mapper = column_mapper,
	)
	cache = Dict(
		:statistic_given_col_val => statistic_given_col_val,
		:encoded_cols => encoded_cols,
	)
	return cache
end

function frequency_encoder_transform(X, cache::Dict)
	statistic_given_col_val = cache[:statistic_given_col_val]
	return generic_transform(X, statistic_given_col_val)
end
