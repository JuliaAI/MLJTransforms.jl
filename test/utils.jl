
function create_dummy_dataset(target_type::Symbol; as_dataframe::Bool = false)
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
    elseif target_type == :binary_str
        y = ["no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes"]
    elseif target_type == :multiclass
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    elseif target_type == :multiclass_str
        y = ["c1", "c2", "c3", "c1", "c2", "c3", "c1", "c2", "c3", "c1"]
    elseif target_type == :regression
        y = [10.5, 15.2, 13.1, 14.7, 11.9, 16.8, 17.0, 19.3, 18.1, 20.0]
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
	return X, y
end