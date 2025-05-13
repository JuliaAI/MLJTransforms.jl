# generic functions go here; such function can be used  throughout multiple methods


"""
**Private method.**

A generic function to fit a class of transformers where its convenient to define a single `feature_mapper` function that
takes the column as a vector and potentially other arguments (as passed in ...args and ...kwargs) and returns
a dictionary that maps each level of the categorical feature to a scalar or vector
according to the transformation logic. In other words, the `feature_mapper` simply answers the question "For level n of
the current categorical feature c, what should the new value or vector (multiple features) be as defined by the transformation
logic?"

# Arguments

    - `X`: A table where the elements of the categorical features have [scitypes](https://juliaai.github.io/ScientificTypes.jl/dev/) 
    `Multiclass` or `OrderedFactor`
    - `features=[]`: A list of names of categorical features given as symbols to exclude or include from encoding
    - `ignore=true`: Whether to exclude or includes the features given in `features`
    - `ordered_factor=false`: Whether to encode `OrderedFactor` or ignore them
    - `feature_mapper`: Defined above. 

# Returns

    - `mapping_per_feat_level`: Maps each level for each feature in a subset of the categorical features of
     X into a scalar or a vector. 
    - `encoded_features`: The subset of the categorical features of X that were encoded
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
    mapping_per_feat_level = Dict()

    # 4. Use feature mapper to compute the mapping of each level in each column
    encoded_features = Symbol[]# to store column that were actually encoded
    for feat_name in feat_names
        feat_col = Tables.getcolumn(X, feat_name)
        feat_type = elscitype(feat_col)
        feat_has_allowed_type =
            feat_type <: Union{Missing, Multiclass} ||
            (ordered_factor && feat_type <: Union{Missing, OrderedFactor})
        if feat_has_allowed_type  # then should be encoded
            push!(encoded_features, feat_name)
            # Compute the dict using the given feature_mapper function
            mapping_per_feat_level[feat_name] =
                feature_mapper(feat_col, feat_name, args...; kwargs...)
        end
    end
    return mapping_per_feat_level, encoded_features
end



"""
**Private method.**

Function to generate new feature names: feat_name_0, feat_name_1,..., feat_name_n or if possible,
feat_name_level_0, feat_name_level_1,..., feat_name_level_n
"""
function generate_new_feat_names(
    feat_name,
    num_inds,
    levels,
    existing_names;
    use_levelnames = true,
)
    # Convert levels (e.g. KeySet or Tuple) to an indexable vector
    levels_vec = collect(levels)

    conflict = true        # true while there's a name clash
    count = 1              # number of underscores in the suffix
    new_column_names = Symbol[]

    while conflict
        suffix = repeat("_", count)
        if use_levelnames
            # Always use the first num_inds level names
            new_column_names = [ Symbol("$(feat_name)$(suffix)$(levels_vec[i])") for i in 1:num_inds ]
        else
            # Always use numeric indices
            new_column_names = [ Symbol("$(feat_name)$(suffix)$i") for i in 1:num_inds ]
        end
        # Check for collisions
        conflict = any(name -> name in existing_names, new_column_names)
        count += 1
    end

    return new_column_names
end



"""
**Private method.**

Given a table `X` and a dictionary `mapping_per_feat_level` which maps each level for each column in
a subset of categorical features of X into a scalar or a vector (as specified in single_feat)

  - transforms each value (some level) in each column in `X` using the function in `mapping_per_feat_level`
    into a scalar (single_feat=true)

  - transforms each value (some level) in each column in `X` using the function in `mapping_per_feat_level`
    into a set of k features where k is the length of the vector (single_feat=false)
  - In both cases it attempts to preserve the type of the table.
  - In the latter case, it assumes that all levels under the same category are mapped to vectors of the same length. Such
    assumption is necessary because any column in X must correspond to a constant number of features
    in the output table (which is equal to k).
  - Features not in the dictionary are mapped to themselves (i.e., not changed).
  - Levels not in the nested dictionary are mapped to themselves if `identity_map_unknown` is true else raise an error.
  - use_levelnames: if true, the new feature names are generated using the level names when the transform generates multiple features;
    else they are generated using the indices of the levels.
  - custom_levels: if not nothing, then the levels of the categorical features are replaced by the custom_levels
"""
function generic_transform(
    X,
    mapping_per_feat_level;
    single_feat = true,
    ignore_unknown = false,
    use_levelnames = false,
    custom_levels = nothing,
)
    feat_names = Tables.schema(X).names
    new_feat_names = Symbol[]
    new_cols = []
    for feat_name in feat_names
        col = Tables.getcolumn(X, feat_name)
        # Create the transformation function for each column
        if feat_name in keys(mapping_per_feat_level)
            if !ignore_unknown
                train_levels = keys(mapping_per_feat_level[feat_name])
                test_levels = levels(col)
                # test levels must be a subset of train levels
                if !issubset(test_levels, train_levels)
                    # get the levels in test that are not in train
                    lost_levels = setdiff(test_levels, train_levels)
                    error(
                        "While transforming, found novel levels for the column $(feat_name): $(lost_levels) that were not seen while training.",
                    )
                end
            end

            if single_feat
                level2scalar = mapping_per_feat_level[feat_name]
                new_col = !isempty(level2scalar) ? recode(col, level2scalar...) : col
                push!(new_cols, new_col)
                push!(new_feat_names, feat_name)
            else
                level2vector = mapping_per_feat_level[feat_name]
                new_multi_col = map(x -> get(level2vector, x, x), col)
                new_multi_col = [col for col in eachrow(hcat(new_multi_col...))]
                push!(new_cols, new_multi_col...)

                feat_names_with_inds = generate_new_feat_names(
                    feat_name,
                    length(first(mapping_per_feat_level[feat_name])[2]),
                    (custom_levels === nothing) ? keys(mapping_per_feat_level[feat_name]) : custom_levels,
                    feat_names;
                    use_levelnames = use_levelnames,
                )
                push!(new_feat_names, feat_names_with_inds...)
            end
        else
            # Not to be transformed => left as is
            push!(new_feat_names, feat_name)
            push!(new_cols, col)
        end
    end

    transformed_X = NamedTuple{tuple(new_feat_names...)}(tuple(new_cols)...)
    # Attempt to preserve table type
    transformed_X = Tables.materializer(X)(transformed_X)
    return transformed_X
end
