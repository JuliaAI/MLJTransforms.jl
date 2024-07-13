# Function to create a dummy dataset
function create_dummy_dataset(target_type::Symbol; as_dataframe::Bool = false, return_y::Bool=true)
    # Define categorical columns with shorter names
    A = ["g", "b", "g", "r", "r", "r", "r", "b", "b", "r"]  
    B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    C = ["f", "f", "f", "m", "f", "m", "f", "m", "f", "m"]  
    D = [true, false, true, false, true, false, true, false, true, false]
    E = [1, 2, 3, 4, 5, 6, 6, 3, 2, 1]
    F = ["s", "s", "s", "s", "l", "m", "s", "l", "m", "m"]  

    # Define the target variable based on the target type
    if target_type == :binary
        y = [0, 1, 1, 1, 0, 1, 0, 1, 1, 1]
        y = coerce(y, Multiclass)
    elseif target_type == :binary_str
        y = ["no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes"]
        y = coerce(y, Multiclass)
    elseif target_type == :multiclass
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        y = coerce(y, Multiclass)
    elseif target_type == :multiclass_str
        y = ["c1", "c2", "c3", "c1", "c2", "c3", "c1", "c2", "c3", "c1"]
        y = coerce(y, Multiclass)
    elseif target_type == :regression
        y = [10.5, 15.2, 13.1, 14.7, 11.9, 16.8, 17.0, 19.3, 18.1, 20.0]
        y = coerce(y, Continuous)
    else
        error("Unsupported target type.")
    end

    # Combine into a named tuple
    X = (A = A, B = B, C = C, D = D, E = E, F = F)

    # Coerce A, C, D, F to multiclass and B to continuous and E to ordinal
    X = coerce(X,
    :A => Multiclass,
    :B => Continuous,
    :C => Multiclass,
    :D => Multiclass,
    :E => OrderedFactor,
    :F => Multiclass,
    )

    as_dataframe && (X = DataFrame(X))
    
    return (return_y) ?  (X, y) : X;
end


struct Object{I<:Integer}
	A::I
end
# Create dummy dataset but with high cardinality
function generate_high_cardinality_table(num_rows; obj=false, special_cat='E')
	# Set the random seed for reproducibility
	Random.seed!(stable_rng, 123)

	# Define the categories for the categorical features with their respective probabilities
	low_card_categories = ['A', 'B', 'C', 'D', special_cat]
	low_card_probs = [0.7, 0.2, 0.05, 0.04, 0.01]  # Imbalanced distribution



	high_card_categories1 = [((obj) ? Object(i) : i) for i in 1:100]
	high_card_probs1 = vcat(fill(0.01, 90), fill(0.1, 10))  # Last 10 categories more frequent

	high_card_categories2 = [string("Group", i) for i in 1:200]
	high_card_probs2 = vcat(fill(0.005, 190), fill(0.05, 10))  # Last 10 categories more frequent

	# Function to generate a weighted random sample
	function weighted_sample(categories, probs)
		cumulative_probs = cumsum(probs)
		rand_val = rand()
		for (i, p) in enumerate(cumulative_probs)
			if rand_val <= p
				return categories[i]
			end
		end
	end

	# Generate the categorical features with imbalanced distributions
	low_card_feature = [weighted_sample(low_card_categories, low_card_probs) for _ in 1:num_rows]
	high_card_feature1 = [weighted_sample(high_card_categories1, high_card_probs1) for _ in 1:num_rows]
	high_card_feature2 = [weighted_sample(high_card_categories2, high_card_probs2) for _ in 1:num_rows]

	dataset = DataFrame(
	LowCardFeature = low_card_feature,
	HighCardFeature1 = high_card_feature1,
	HighCardFeature2 = high_card_feature2
	)

	dataset = coerce(dataset,
	:LowCardFeature  => Multiclass,
	:HighCardFeature1 => Multiclass,
	:HighCardFeature2 => Multiclass,
	)

	return dataset

end


function generate_X_with_missingness(;john_name="John")
    Xm = (
        A = categorical(["Ben", john_name, missing, missing, "Mary", "John", missing]),
        B = [1.85, 1.67, missing, missing, 1.5, 1.67, missing],
        C= categorical([7, 5, missing, missing, 10, 5, missing]),
        D = [23, 23, 44, 66, 14, 23, 11],
        E = categorical([missing, 'g', 'r', missing, 'r', 'g', 'p'])
    )

    return Xm
end


# Display the dataset
dataset = generate_high_cardinality_table(1000; obj=false)