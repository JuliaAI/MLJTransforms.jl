
@with_kw_noshow mutable struct OneHotEncoder <: Unsupervised
    features::Vector{Symbol}   = Symbol[]
    drop_last::Bool            = false
    ordered_factor::Bool       = true
    ignore::Bool               = false
end

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MMI.MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Union{Pair{<:Unsigned,Symbol}, Pair{Missing, Symbol}}}}
    fitted_levels_given_feature::Dict{Symbol, CategoricalArray}
end

# join feature and level into new label without clashing with anything
# in all_features:
function compound_label(all_features, feature, level)
    label = Symbol(string(feature, "__", level))
    # in the (rare) case subft is not a new feature label:
    while label in all_features
        label = Symbol(string(label,"_"))
    end
    return label
end

function MMI.fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector

    if isempty(transformer.features)
        specified_features = collect(all_features)
    else
        if transformer.ignore
            specified_features = filter(all_features |> collect) do ftr
                !(ftr in transformer.features)
            end
        else
            specified_features = transformer.features
        end
    end


    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()

    allowed_scitypes = ifelse(
        transformer.ordered_factor,
        Union{Missing, Finite},
        Union{Missing, Multiclass}
    )
    fitted_levels_given_feature = Dict{Symbol, CategoricalArray}()
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MMI.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = classes(col)
            fitted_levels_given_feature[ftr] = levels
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MMI.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features),
                                    ref_name_pairs_given_feature,
                                    fitted_levels_given_feature)

    # get new feature names
    d = ref_name_pairs_given_feature
    new_features = Symbol[]
    features_to_be_transformed = keys(d)
    for ftr in all_features
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
        else
            push!(new_features, ftr)
        end
    end

    report = (features_to_be_encoded=
              collect(keys(ref_name_pairs_given_feature)),
              new_features=new_features)
    cache = nothing

    return fitresult, cache, report
end

MMI.fitted_params(::OneHotEncoder, fitresult) = (
    all_features = fitresult.all_features,
    fitted_levels_given_feature = fitresult.fitted_levels_given_feature,
    ref_name_pairs_given_feature = fitresult.ref_name_pairs_given_feature,
)

# If v=categorical('a', 'a', 'b', 'a', 'c') and MMI.int(v[1]) = ref
# then `_hot(v, ref) = [true, true, false, true, false]`
hot(v::AbstractVector{<:CategoricalValue}, ref) = map(v) do c
    MMI.int(c) == ref
end

function hot(col::AbstractVector{<:Union{Missing, CategoricalValue}}, ref) map(col) do c
    if ismissing(ref)
        missing
    else
        MMI.int(c) == ref
    end
end
end

function MMI.transform(transformer::OneHotEncoder, fitresult, X)
    features = Tables.schema(X).names     # tuple not vector

    d = fitresult.ref_name_pairs_given_feature

    # check the features match the fit result
    all(e -> e in fitresult.all_features, features) ||
        error("Attempting to transform table with feature "*
              "names not seen in fit. ")
    new_features = Symbol[]
    new_cols = [] # not Vector[] !!
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MMI.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            Set(fitresult.fitted_levels_given_feature[ftr]) ==
                Set(classes(col)) ||
            error("Found category level mismatch in feature `$(ftr)`. "*
            "Consider using `levels!` to ensure fitted and transforming "*
            "features have the same category levels.")
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                if ismissing(ref) missing
                else float.(hot(col, ref))
                end
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)
    return MMI.table(named_cols, prototype=X)
end

metadata_model(OneHotEncoder,
    input_scitype   = Table,
    output_scitype = Table,
    human_name = "one-hot encoder",
    load_path    = "MLJModels.OneHotEncoder")

"""
$(MLJModelInterface.doc_header(OneHotEncoder))

Use this model to one-hot encode the `Multiclass` and `OrderedFactor`
features (columns) of some table, leaving other columns unchanged.

New data to be transformed may lack features present in the fit data,
but no *new* features can be present.

**Warning:** This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column, `col`, is the same for
training data and new data to be transformed.

To ensure *all* features are transformed into `Continuous` features, or
dropped, use [`ContinuousEncoder`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table. Columns can be of mixed type
  but only those with element scitype `Multiclass` or `OrderedFactor`
  can be encoded. Check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: a vector of symbols (feature names). If empty (default)
  then all `Multiclass` and `OrderedFactor` features are
  encoded. Otherwise, encoding is further restricted to the specified
  features (`ignore=false`) or the unspecified features
  (`ignore=true`). This default behavior can be modified by the
  `ordered_factor` flag.

- `ordered_factor=false`: when `true`, `OrderedFactor` features are
  universally excluded

- `drop_last=true`: whether to drop the column corresponding to the
  final class of encoded features. For example, a three-class feature
  is spawned into three new features if `drop_last=false`, but just
  two features otherwise.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `all_features`: names of all features encountered in training

- `fitted_levels_given_feature`: dictionary of the levels associated
  with each feature encoded, keyed on the feature name

- `ref_name_pairs_given_feature`: dictionary of pairs `r => ftr` (such
  as `0x00000001 => :grad__A`) where `r` is a CategoricalArrays.jl
  reference integer representing a level, and `ftr` the corresponding
  new feature name; the dictionary is keyed on the names of features that
  are encoded


# Report

The fields of `report(mach)` are:

- `features_to_be_encoded`: names of input features to be encoded

- `new_features`: names of all output features


# Example

```
using MLJ

X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical(["A", "B", "A", "C"], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3])

julia> schema(X)
┌───────────┬──────────────────┐
│ names     │ scitypes         │
├───────────┼──────────────────┤
│ name      │ Multiclass{4}    │
│ grade     │ OrderedFactor{3} │
│ height    │ Continuous       │
│ n_devices │ Count            │
└───────────┴──────────────────┘

hot = OneHotEncoder(drop_last=true)
mach = fit!(machine(hot, X))
W = transform(mach, X)

julia> schema(W)
┌──────────────┬────────────┐
│ names        │ scitypes   │
├──────────────┼────────────┤
│ name__Danesh │ Continuous │
│ name__John   │ Continuous │
│ name__Lee    │ Continuous │
│ grade__A     │ Continuous │
│ grade__B     │ Continuous │
│ height       │ Continuous │
│ n_devices    │ Count      │
└──────────────┴────────────┘
```

See also [`ContinuousEncoder`](@ref).

"""
OneHotEncoder