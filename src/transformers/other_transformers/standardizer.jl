
"""
    UnivariateStandardizer()

Transformer type for standardizing (whitening) single variable data.

This model may be deprecated in the future. Consider using
[`Standardizer`](@ref), which handles both tabular *and* univariate data.

"""
mutable struct UnivariateStandardizer <: Unsupervised end

function MMI.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    stdv = std(v)
    stdv > eps(typeof(stdv)) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), stdv)
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

MMI.fitted_params(::UnivariateStandardizer, fitresult) =
    (mean=fitresult[1], std=fitresult[2])


# for transforming single value:
function MMI.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MMI.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]


# # STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

mutable struct Standardizer <: Unsupervised
    # features to be standardized; empty means all
    features::Union{AbstractVector{Symbol}, Function}
    ignore::Bool # features to be ignored
    ordered_factor::Bool
    count::Bool
end

# keyword constructor
function Standardizer(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false,
    ordered_factor::Bool=false,
    count::Bool=false
)
    transformer = Standardizer(features, ignore, ordered_factor, count)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::Standardizer)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::Standardizer, verbosity::Int, X)

    # if not a table, it must be an abstract vector, eltpye AbstractFloat:
    is_univariate = !Tables.istable(X)

    # are we attempting to standardize Count or OrderedFactor?
    is_invertible = !transformer.count && !transformer.ordered_factor

    # initialize fitresult:
    fitresult_given_feature = LittleDict{Symbol,Tuple{AbstractFloat,AbstractFloat}}()

    # special univariate case:
    if is_univariate
        fitresult_given_feature[:unnamed] =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, X)[1]
        return (is_univariate=true,
                is_invertible=true,
                fitresult_given_feature=fitresult_given_feature),
        nothing, nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes =
        collect(elscitype(selectcols(X, c)) for c in all_features)
    scitypes = Vector{Type}([Continuous])
    transformer.ordered_factor && push!(scitypes, OrderedFactor)
    transformer.count && push!(scitypes, Count)
    AllowedScitype = Union{scitypes...}

    # determine indices of all_features to be transformed
    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                feature_scitypes[j] <: AllowedScitype
            end
        else
            !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn "Some specified features not present in table to be fit. "
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                ifelse(
                    transformer.ignore,
                    !(all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype,
                    (all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype
                )
            end
        end
    else
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            ifelse(
                transformer.ignore,
                !(transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype,
                (transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype
            )
        end
    end

    isempty(cols_to_fit) && verbosity > -1 &&
        @warn "No features to standarize."

    # fit each feature and add result to above dict
    verbosity > 1 && @info "Features standarized: "
    for j in cols_to_fit
        col_data = if (feature_scitypes[j] <: OrderedFactor)
            coerce(selectcols(X, j), Continuous)
        else
            selectcols(X, j)
        end
        col_fitresult, _, _ =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity > 1 &&
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  "*
            "sigma=$(col_fitresult[2])"
    end

    fitresult = (is_univariate=false, is_invertible=is_invertible,
                 fitresult_given_feature=fitresult_given_feature)
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

function MMI.fitted_params(::Standardizer, fitresult)
    is_univariate, _, dic = fitresult
    is_univariate &&
        return fitted_params(UnivariateStandardizer(), dic[:unnamed])
    features_fit = keys(dic) |> collect
    zipped = map(ftr->dic[ftr], features_fit)
    means, stds = zip(zipped...) |> collect
    return (; features_fit, means, stds)
end

MMI.transform(::Standardizer, fitresult, X) =
    _standardize(transform, fitresult, X)

function MMI.inverse_transform(::Standardizer, fitresult, X)
    fitresult.is_invertible ||
        error("Inverse standardization is not supported when `count=true` "*
              "or `ordered_factor=true` during fit. ")
    return _standardize(inverse_transform, fitresult, X)
end

function _standardize(operation, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names
    is_univariate, _, fitresult_given_feature = fitresult

    if is_univariate
        univariate_fitresult = fitresult_given_feature[:unnamed]
        return operation(UnivariateStandardizer(), univariate_fitresult, X)
    end

    features_to_be_transformed = keys(fitresult_given_feature)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        ftr_data = selectcols(X, ftr)
        if ftr in features_to_be_transformed
            col_to_transform = coerce(ftr_data, Continuous)
            operation(col_transformer,
                      fitresult_given_feature[ftr],
                      col_to_transform)
        else
            ftr_data
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MMI.table(named_cols, prototype=X)
end

metadata_model(UnivariateStandardizer,
    input_scitype   = AbstractVector{<:Infinite},
    output_scitype = AbstractVector{Continuous},
    human_name = "single variable discretizer",
    load_path    = "MLJTransforms.UnivariateStandardizer")

metadata_model(Standardizer,
    input_scitype   = Union{Table, AbstractVector{<:Continuous}},
    output_scitype = Union{Table, AbstractVector{<:Continuous}},
    load_path = "MLJTransforms.Standardizer")

"""
$(MLJModelInterface.doc_header(Standardizer))

Use this model to standardize (whiten) a `Continuous` vector, or
relevant columns of a table. The rescalings applied by this
transformer to new data are always those learned during the training
phase, which are generally different from what would actually
standardize the new data.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be standardized;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: one of the following, with the behavior indicated below:

  - `[]` (empty, the default): standardize all features (columns)
    having `Continuous` element scitype

  - non-empty vector of feature names (symbols): standardize only the
    `Continuous` features in the vector (if `ignore=false`) or
    `Continuous` features *not* named in the vector (`ignore=true`).

  - function or other callable: standardize a feature if the callable
    returns `true` on its name. For example, `Standardizer(features =
    name -> name in [:x1, :x3], ignore = true, count=true)` has the
    same effect as `Standardizer(features = [:x1, :x3], ignore = true,
    count=true)`, namely to standardize all `Continuous` and `Count`
    features, with the exception of `:x1` and `:x3`.

  Note this behavior is further modified if the `ordered_factor` or `count` flags
  are set to `true`; see below

- `ignore=false`: whether to ignore or standardize specified `features`, as
  explained above

- `ordered_factor=false`: if `true`, standardize any `OrderedFactor`
  feature wherever a `Continuous` feature would be standardized, as
  described above

- `count=false`: if `true`, standardize any `Count` feature wherever a
  `Continuous` feature would be standardized, as described above


# Operations

- `transform(mach, Xnew)`: return `Xnew` with relevant features
  standardized according to the rescalings learned during fitting of
  `mach`.

- `inverse_transform(mach, Z)`: apply the inverse transformation to
  `Z`, so that `inverse_transform(mach, transform(mach, Xnew))` is
  approximately the same as `Xnew`; unavailable if `ordered_factor` or
  `count` flags were set to `true`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_fit` - the names of features that will be standardized

- `means` - the corresponding untransformed mean values

- `stds` - the corresponding untransformed standard deviations


# Report

The fields of `report(mach)` are:

- `features_fit`: the names of features that will be standardized


# Examples

```
using MLJ

X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce([:x, :y, :x], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));

julia> schema(X)
┌──────────┬──────────────────┐
│ names    │ scitypes         │
├──────────┼──────────────────┤
│ ordinal1 │ Count            │
│ ordinal2 │ OrderedFactor{2} │
│ ordinal3 │ Continuous       │
│ ordinal4 │ Continuous       │
│ nominal  │ Multiclass{3}    │
└──────────┴──────────────────┘

stand1 = Standardizer();

julia> transform(fit!(machine(stand1, X)), X)
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [-1.0, 0.0, 1.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)

stand2 = Standardizer(features=[:ordinal3, ], ignore=true, count=true);

julia> transform(fit!(machine(stand2, X)), X)
(ordinal1 = [-1.0, 0.0, 1.0],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [10.0, 20.0, 30.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)
```

See also [`OneHotEncoder`](@ref), [`ContinuousEncoder`](@ref).
"""
Standardizer