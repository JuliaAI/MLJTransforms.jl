# generic functions go here; such function can be used  throughout multiple methods

"""
```julia
generic_fit(X,
    features = Symbol[],
    args...;
    ignore::Bool = true,
    ordered_factor::Bool = false,
    feature_mapper,
    kwargs...,
)
```

Given a `feature_mapper` (see definition below), this method applies 
    `feature_mapper` across a specified subset of categorical columns in X and returns a dictionary 
    whose keys are the feature names, and each value is the corresponding 
    level‑to‑value mapping produced by `feature_mapper`. 

In essence, it spares effort of looping over each column and applying the `feature_mapper` function manually as well as handling the feature selection logic.


# Arguments

$X_doc
$features_doc
$ignore_doc
$ordered_factor_doc
- feature_mapper: function that, for a given vector (eg, corresponding to a categorical column from the dataset `X`), 
    produces a mapping from each category level name in this vector to a scalar or vector according to specified transformation logic.

# Note

- Any additional arguments (whether keyword or not) provided to this function are passed to the `feature_mapper` function which
    is helpful when `feature_mapper` requires additional arguments to compute the mapping (eg, hyperparameters).

# Returns
- `mapping_per_feat_level`: Maps each level for each feature in a subset of the categorical features of
    X into a scalar or a vector. 
$encoded_features_doc
"""
function generic_fit(X,
    features = Symbol[],
    args...;
    ignore::Bool = true,
    ordered_factor::Bool = false,
    feature_mapper,
    kwargs...,
)
    # 1. Get X column types and names
    feat_names = Tables.schema(X).names

    #2.  Modify column_names based on features 
    if features isa Symbol
        features = [features]
    end
    
    if features isa AbstractVector{Symbol}
        # Original behavior for vector of symbols
        feat_names =
            (ignore) ? setdiff(feat_names, features) : intersect(feat_names, features)
    else
        # If features is a callable, apply it to each feature name
        if ignore
            feat_names = filter(name -> !features(name), feat_names)
        else
            feat_names = filter(features, feat_names)
        end
    end

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
```julia
generic_transform(
    X,
    mapping_per_feat_level;
    single_feat::Bool = true,
    ignore_unknown::Bool = false,
    use_levelnames::Bool = false,
    custom_levels = nothing,
    ensure_categorical::Bool = false,
)
```


Apply a per‐level feature mapping to selected categorical columns in `X`, returning a new table of the same type.

# Arguments

$X_doc
- `mapping_per_feat_level::Dict{Symbol,Dict}`:
    A dict whose keys are feature names (`Symbol`) and values are themselves dictionaries 
    mapping each observed level to either a scalar (if `single_feat=true`) or a fixed‐length vector 
        (if `single_feat=false`). Only columns whose names appear in `mapping_per_feat_level` are 
            transformed; others pass through unchanged.
- `single_feat::Bool=true`:
    If `true`, each input level is mapped to a single scalar feature; if `false`,
    each input level is mapped to a length‑`k` vector, producing `k` output columns.
- `ignore_unknown::Bool=false`:
    If `false`, novel levels in `X` (not seen during fit) will raise an error; 
    if `true`, novel levels will be left unchanged (identity mapping).
- `use_levelnames::Bool=false`:
    When `single_feat=false`, controls naming of the expanded columns: `true`: use actual level names (e.g. `:color_red`, `:color_blue`), 
    `false`: use numeric indices (e.g. `:color_1`, `:color_2`).
- `custom_levels::Union{Nothing,Vector}`:
    If not `nothing`, overrides the names of levels used to generate feature names when `single_feat=false`.
- `ensure_categorical::Bool=false`:
    Only when `single_feat=true` and if `true`, preserves the categorical type of the column after 
        recoding (eg, feature should still be recognized as `Multiclass` after transformation)

# Returns

A new table of potentially similar to `X` but with categorical columns transformed according to `mapping_per_feat_level`.
"""
function generic_transform(
    X,
    mapping_per_feat_level;
    single_feat = true,
    ignore_unknown = false,
    use_levelnames = false,
    custom_levels = nothing,
    ensure_categorical = false,
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
                if ensure_categorical
                    new_col = !isempty(level2scalar) ? recode(col, level2scalar...) : col
                else 
                    new_col = !isempty(level2scalar) ? unwrap.(recode(col, level2scalar...)) : col
                end
               
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
