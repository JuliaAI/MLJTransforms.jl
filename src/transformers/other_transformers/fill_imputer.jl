
round_median(v::AbstractVector) = v -> round(eltype(v), median(v))

_median(e)       = skipmissing(e) |> median
_round_median(e) = skipmissing(e) |> (f -> round(eltype(f), median(f)))
_mode(e)         = skipmissing(e) |> mode

@with_kw_noshow mutable struct UnivariateFillImputer <: Unsupervised
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::UnivariateFillImputer,
                      verbosity::Integer,
                      v)

    filler(v, ::Type) = throw(ArgumentError(
        "Imputation is not supported for vectors "*
        "of elscitype $(elscitype(v))."))
    filler(v, ::Type{<:Union{Continuous,Missing}}) =
        transformer.continuous_fill(v)
    filler(v, ::Type{<:Union{Count,Missing}}) =
        transformer.count_fill(v)
    filler(v, ::Type{<:Union{Finite,Missing}}) =
        transformer.finite_fill(v)

    fitresult = (filler=filler(v, elscitype(v)),)
    cache = nothing
    report = NamedTuple()

    return fitresult, cache, report

end

function replace_missing(::Type{<:Finite}, vnew, filler)
   all(in(levels(filler)), levels(vnew)) ||
        error(ArgumentError("The `column::AbstractVector{<:Finite}`"*
                            " to be transformed must contain the same levels"*
                            " as the categorical value to be imputed"))
   replace(vnew, missing => filler)

end

function replace_missing(::Type, vnew, filler)
   T = promote_type(nonmissing(eltype(vnew)), typeof(filler))
   w_tight = similar(vnew, T)
   @inbounds for i in eachindex(vnew)
        if ismissing(vnew[i])
           w_tight[i] = filler
        else
           w_tight[i] = vnew[i]
        end
   end
   return w_tight
end

function MMI.transform(transformer::UnivariateFillImputer,
                           fitresult,
                           vnew)

    filler = fitresult.filler

    scitype(filler) <: elscitype(vnew) ||
    error("Attempting to impute a value of scitype $(scitype(filler)) "*
    "into a vector of incompatible elscitype, namely $(elscitype(vnew)). ")

    if elscitype(vnew) >: Missing
        w_tight = replace_missing(nonmissing(elscitype(vnew)), vnew, filler)
    else
        w_tight = vnew
    end

    return w_tight
end

MMI.fitted_params(::UnivariateFillImputer, fitresult) = fitresult

@with_kw_noshow mutable struct FillImputer <: Unsupervised
    features::Vector{Symbol}  = Symbol[]
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::FillImputer, verbosity::Int, X)

    s = schema(X)
    features_seen = s.names |> collect # "seen" = "seen in fit"
    scitypes_seen = s.scitypes |> collect

    features = isempty(transformer.features) ? features_seen :
        transformer.features

    issubset(features, features_seen) || throw(ArgumentError(
    "Some features specified do not exist in the supplied table. "))

    # get corresponding scitypes:
    mask = map(features_seen) do ftr
        ftr in features
    end
    features = @view features_seen[mask] # `features` re-ordered
    scitypes = @view scitypes_seen[mask]
    features_and_scitypes = zip(features, scitypes) #|> collect

    # now keep those features that are imputable:
    function isimputable(ftr, T::Type)
        if verbosity > 0 && !isempty(transformer.features)
            @info "Feature $ftr will not be imputed "*
            "(imputation for $T not supported). "
        end
        return false
    end
    isimputable(ftr, ::Type{<:Union{Continuous,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Count,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Finite,Missing}}) = true

    mask = map(features_and_scitypes) do tup
        isimputable(tup...)
    end
    features_to_be_imputed = @view features[mask]

    univariate_transformer =
        UnivariateFillImputer(continuous_fill=transformer.continuous_fill,
                              count_fill=transformer.count_fill,
                              finite_fill=transformer.finite_fill)
    univariate_fitresult(ftr) = MMI.fit(univariate_transformer,
                                            verbosity - 1,
                                            selectcols(X, ftr))[1]

    fitresult_given_feature =
        Dict(ftr=> univariate_fitresult(ftr) for ftr in features_to_be_imputed)

    fitresult = (features_seen=features_seen,
                 univariate_transformer=univariate_transformer,
                 fitresult_given_feature=fitresult_given_feature)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::FillImputer, fitresult, X)

    features_seen = fitresult.features_seen # seen in fit
    univariate_transformer = fitresult.univariate_transformer
    fitresult_given_feature = fitresult.fitresult_given_feature

    all_features = Tables.schema(X).names

    # check that no new features have appeared:
    all(e -> e in features_seen, all_features) || throw(ArgumentError(
        "Attempting to transform table with "*
        "feature labels not seen in fit.\n"*
        "Features seen in fit = $features_seen.\n"*
        "Current features = $([all_features...]). "))

    features = keys(fitresult_given_feature)

    cols = map(all_features) do ftr
        col = MMI.selectcols(X, ftr)
        if ftr in features
            fr = fitresult_given_feature[ftr]
            return transform(univariate_transformer, fr, col)
        end
        return col
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))
    return MMI.table(named_cols, prototype=X)

end

function MMI.fitted_params(::FillImputer, fr)
    dict = fr.fitresult_given_feature
    filler_given_feature = Dict(ftr=>dict[ftr].filler for ftr in keys(dict))
    return (features_seen_in_fit=fr.features_seen,
            univariate_transformer=fr.univariate_transformer,
            filler_given_feature=filler_given_feature)
end

metadata_model(UnivariateFillImputer,
    input_scitype = Union{AbstractVector{<:Union{Continuous,Missing}},
                  AbstractVector{<:Union{Count,Missing}},
                  AbstractVector{<:Union{Finite,Missing}}},
    output_scitype= Union{AbstractVector{<:Continuous},
                  AbstractVector{<:Count},
                  AbstractVector{<:Finite}},
    human_name = "single variable fill imputer",
    load_path  = "MLJTransforms.UnivariateFillImputer")

metadata_model(FillImputer,
    input_scitype   = Table,
    output_scitype = Table,
    load_path = "MLJTransforms.FillImputer")

"""
$(MLJModelInterface.doc_header(UnivariateFillImputer))

Use this model to imputing `missing` values in a vector with a fixed
value learned from the non-missing values of training vector.

For imputing missing values in tabular data, use [`FillImputer`](@ref)
instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector with element scitype `Union{Missing, T}`
  where `T` is a subtype of `Continuous`, `Multiclass`,
  `OrderedFactor` or `Count`; check scitype using `scitype(x)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `continuous_fill`: function or other callable to determine value to
  be imputed in the case of `Continuous` (abstract float) data;
  default is to apply `median` after skipping `missing` values

- `count_fill`: function or other callable to determine value to be
  imputed in the case of `Count` (integer) data; default is to apply
  rounded `median` after skipping `missing` values

- `finite_fill`: function or other callable to determine value to be
  imputed in the case of `Multiclass` or `OrderedFactor` data
  (categorical vectors); default is to apply `mode` after skipping
  `missing` values


# Operations

- `transform(mach, xnew)`: return `xnew` with missing values imputed
  with the fill values learned when fitting `mach`


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `filler`: the fill value to be imputed in all new data


# Examples

```
using MLJ
imputer = UnivariateFillImputer()

x_continuous = [1.0, 2.0, missing, 3.0]
x_multiclass = coerce(["y", "n", "y", missing, "y"], Multiclass)
x_count = [1, 1, 1, 2, missing, 3, 3]

mach = machine(imputer, x_continuous)
fit!(mach)

julia> fitted_params(mach)
(filler = 2.0,)

julia> transform(mach, [missing, missing, 101.0])
3-element Vector{Float64}:
 2.0
 2.0
 101.0

mach2 = machine(imputer, x_multiclass) |> fit!

julia> transform(mach2, x_multiclass)
5-element CategoricalArray{String,1,UInt32}:
 "y"
 "n"
 "y"
 "y"
 "y"

mach3 = machine(imputer, x_count) |> fit!

julia> transform(mach3, [missing, missing, 5])
3-element Vector{Int64}:
 2
 2
 5
```

For imputing tabular data, use [`FillImputer`](@ref).

"""
UnivariateFillImputer



"""
$(MLJModelInterface.doc_header(FillImputer))

Use this model to impute `missing` values in tabular data. A fixed
"filler" value is learned from the training data, one for each column
of the table.

For imputing missing values in a vector, use
[`UnivariateFillImputer`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any table of input features (eg, a `DataFrame`) whose features
  each have element scitypes `Union{Missing, T}`, where `T` is a
  subtype of `Continuous`, `Multiclass`, `OrderedFactor` or
  `Count`. Check scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `features`: a vector of names of features (symbols) for which
  imputation is to be attempted; default is empty, which is
  interpreted as "impute all".

- `continuous_fill`: function or other callable to determine value to
  be imputed in the case of `Continuous` (abstract float) data; default is to apply
  `median` after skipping `missing` values

- `count_fill`: function or other callable to determine value to
  be imputed in the case of `Count` (integer) data; default is to apply
  rounded `median` after skipping `missing` values

- `finite_fill`: function or other callable to determine value to be
  imputed in the case of `Multiclass` or `OrderedFactor` data
  (categorical vectors); default is to apply `mode` after skipping `missing` values


# Operations

- `transform(mach, Xnew)`: return `Xnew` with missing values imputed with
  the fill values learned when fitting `mach`


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_seen_in_fit`: the names of features (features) encountered
  during training

- `univariate_transformer`: the univariate model applied to determine
    the fillers (it's fields contain the functions defining the filler computations)

- `filler_given_feature`: dictionary of filler values, keyed on
  feature (column) names


# Examples

```
using MLJ
imputer = FillImputer()

X = (a = [1.0, 2.0, missing, 3.0, missing],
     b = coerce(["y", "n", "y", missing, "y"], Multiclass),
     c = [1, 1, 2, missing, 3])

schema(X)
julia> schema(X)
┌───────┬───────────────────────────────┐
│ names │ scitypes                      │
├───────┼───────────────────────────────┤
│ a     │ Union{Missing, Continuous}    │
│ b     │ Union{Missing, Multiclass{2}} │
│ c     │ Union{Missing, Count}         │
└───────┴───────────────────────────────┘

mach = machine(imputer, X)
fit!(mach)

julia> fitted_params(mach).filler_given_feature
(filler = 2.0,)

julia> fitted_params(mach).filler_given_feature
Dict{Symbol, Any} with 3 entries:
  :a => 2.0
  :b => "y"
  :c => 2

julia> transform(mach, X)
(a = [1.0, 2.0, 2.0, 3.0, 2.0],
 b = CategoricalValue{String, UInt32}["y", "n", "y", "y", "y"],
 c = [1, 1, 2, 2, 3],)
```

See also [`UnivariateFillImputer`](@ref).

"""
FillImputer