function frequency_encoder_fit(
	X,
	features::AbstractVector{Symbol} = Symbol[];
	ignore::Bool = true,
	ordered_factor::Bool = false,
	normalize::Bool = false,
)
	# 1. Define column mapper
	function feature_mapper(col)
		statistic_given_feat_val = (!normalize) ? countmap(col) : proportionmap(col)
		return statistic_given_feat_val
	end

	# 2. Pass it to generic_fit
	statistic_given_feat_val, encoded_features = generic_fit(
		X, features; ignore = ignore, ordered_factor = ordered_factor,
		feature_mapper = feature_mapper,
	)
	cache = Dict(
		:statistic_given_feat_val => statistic_given_feat_val,
		:encoded_features => encoded_features,
	)
	return cache
end

function frequency_encoder_transform(X, cache::Dict)
	statistic_given_feat_val = cache[:statistic_given_feat_val]
	return generic_transform(X, statistic_given_feat_val)
end
