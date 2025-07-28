
@with_kw_noshow mutable struct ContinuousEncoder <: Unsupervised
    drop_last::Bool                = false
    one_hot_ordered_factors::Bool  = false
end

function MMI.fit(transformer::ContinuousEncoder, verbosity::Int, X)

    # what features can be converted and therefore kept?
    s = schema(X)
    features = s.names
    scitypes = s.scitypes
    Convertible = Union{Continuous, Finite, Count}
    feature_scitype_tuples = zip(features, scitypes) |> collect
    features_to_keep  =
        first.(filter(t -> last(t) <: Convertible, feature_scitype_tuples))
    features_to_be_dropped = setdiff(collect(features), features_to_keep)

    if verbosity > 0
        if !isempty(features_to_be_dropped)
            @info "Some features cannot be replaced with "*
            "`Continuous` features and will be dropped: "*
            "$features_to_be_dropped. "
        end
    end

    # fit the one-hot encoder:
    hot_encoder =
        OneHotEncoder(ordered_factor=transformer.one_hot_ordered_factors,
                      drop_last=transformer.drop_last)
    hot_fitresult, _, hot_report = MMI.fit(hot_encoder, verbosity - 1, X)

    new_features = setdiff(hot_report.new_features, features_to_be_dropped)

    fitresult = (features_to_keep=features_to_keep,
                 one_hot_encoder=hot_encoder,
                 one_hot_encoder_fitresult=hot_fitresult)

    # generate the report:
    report = (features_to_keep=features_to_keep,
              new_features=new_features)

    cache = nothing

    return fitresult, cache, report

end

MMI.fitted_params(::ContinuousEncoder, fitresult) = fitresult

function MMI.transform(transformer::ContinuousEncoder, fitresult, X)

    features_to_keep, hot_encoder, hot_fitresult = values(fitresult)

    # dump unseen or untransformable features:
    if !issubset(features_to_keep, MMI.schema(X).names)
        throw(
            ArgumentError(
                "Supplied frame does not admit previously selected features."
            )
        )
    end
    X0 = MMI.selectcols(X, features_to_keep)

    # one-hot encode:
    X1 = transform(hot_encoder, hot_fitresult, X0)

    # convert remaining to continuous:
    return coerce(X1, Count=>Continuous, OrderedFactor=>Continuous)

end

metadata_model(ContinuousEncoder,
    input_scitype   = Table,
    output_scitype = Table(Continuous),
    load_path    = "MLJTransforms.ContinuousEncoder")

"""
$(MLJModelInterface.doc_header(ContinuousEncoder))

Use this model to arrange all features (features) of a table to have
`Continuous` element scitype, by applying the following protocol to
each feature `ftr`:

- If `ftr` is already `Continuous` retain it.

- If `ftr` is `Multiclass`, one-hot encode it.

- If `ftr` is `OrderedFactor`, replace it with `coerce(ftr,
  Continuous)` (vector of floating point integers), unless
  `ordered_factors=false` is specified, in which case one-hot encode
  it.

- If `ftr` is `Count`, replace it with `coerce(ftr, Continuous)`.

- If `ftr` has some other element scitype, or was not observed in
  fitting the encoder, drop it from the table.

**Warning:** This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column, `col`, is the same for
training data and new data to be transformed.

To selectively one-hot-encode categorical features (without dropping
features) use [`OneHotEncoder`](@ref) instead.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table. features can be of mixed type
  but only those with element scitype `Multiclass` or `OrderedFactor`
  can be encoded. Check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `drop_last=true`: whether to drop the column corresponding to the
  final class of one-hot encoded features. For example, a three-class
  feature is spawned into three new features if `drop_last=false`, but
  two just features otherwise.

- `one_hot_ordered_factors=false`: whether to one-hot any feature
  with `OrderedFactor` element scitype, or to instead coerce it
  directly to a (single) `Continuous` feature using the order


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `features_to_keep`: names of features that will not be dropped from
  the table

- `one_hot_encoder`: the `OneHotEncoder` model instance for handling
  the one-hot encoding

- `one_hot_encoder_fitresult`: the fitted parameters of the
  `OneHotEncoder` model


# Report

- `features_to_keep`: names of input features that will not be dropped
  from the table

- `new_features`: names of all output features


# Example

```julia
X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical(["A", "B", "A", "C"], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3],
     comments=["the force", "be", "with you", "too"])

julia> schema(X)
┌───────────┬──────────────────┐
│ names     │ scitypes         │
├───────────┼──────────────────┤
│ name      │ Multiclass{4}    │
│ grade     │ OrderedFactor{3} │
│ height    │ Continuous       │
│ n_devices │ Count            │
│ comments  │ Textual          │
└───────────┴──────────────────┘

encoder = ContinuousEncoder(drop_last=true)
mach = fit!(machine(encoder, X))
W = transform(mach, X)

julia> schema(W)
┌──────────────┬────────────┐
│ names        │ scitypes   │
├──────────────┼────────────┤
│ name__Danesh │ Continuous │
│ name__John   │ Continuous │
│ name__Lee    │ Continuous │
│ grade        │ Continuous │
│ height       │ Continuous │
│ n_devices    │ Continuous │
└──────────────┴────────────┘

julia> setdiff(schema(X).names, report(mach).features_to_keep) # dropped features
1-element Vector{Symbol}:
 :comments

```

See also [`OneHotEncoder`](@ref)
"""
ContinuousEncoder
