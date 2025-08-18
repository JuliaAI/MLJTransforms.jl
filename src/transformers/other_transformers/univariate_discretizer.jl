
# helper function:
reftype(::CategoricalArray{<:Any,<:Any,R}) where R = R

@with_kw_noshow mutable struct UnivariateDiscretizer <:Unsupervised
    n_classes::Int = 512
end

struct UnivariateDiscretizerResult{C}
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
    element::C
end

function MMI.fit(transformer::UnivariateDiscretizer, verbosity::Int, X)
    n_classes = transformer.n_classes
    quantiles = quantile(X, Array(range(0, stop=1, length=2*n_classes+1)))
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles

    # odd_quantiles for transforming, even_quantiles used for
    # inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    # determine optimal reference type for encoding as categorical:
    R = reftype(categorical(1:n_classes, compress=true))
    output_prototype = categorical(R(1):R(n_classes), compress=true, ordered=true)
    element = output_prototype[1]

    cache  = nothing
    report = NamedTuple()

    res = UnivariateDiscretizerResult(odd_quantiles, even_quantiles, element)
    return res, cache, report
end

# acts on scalars:
function transform_to_int(
            result::UnivariateDiscretizerResult{<:CategoricalValue{R}},
            r::Real) where R
    k = oneR = R(1)
    @inbounds for q in result.odd_quantiles
        if r > q
            k += oneR
        end
    end
    return k
end

# transforming scalars:
MMI.transform(::UnivariateDiscretizer, result, r::Real) =
    transform(result.element, transform_to_int(result, r))

# transforming vectors:
function MMI.transform(::UnivariateDiscretizer, result, v)
   w = [transform_to_int(result, r) for r in v]
   return transform(result.element, w)
end

# inverse_transforming raw scalars:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result , k::Integer)
    k <= transformer.n_classes && k > 0 ||
        error("Cannot transform an integer outside the range "*
              "`[1, n_classes]`, where `n_classes = $(transformer.n_classes)`")
    return result.even_quantiles[k]
end

# inverse transforming a categorical value:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result, e::CategoricalValue)
    k = CategoricalArrays.DataAPI.unwrap(e)
    return inverse_transform(transformer, result, k)
end

# inverse transforming raw vectors:
MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          w::AbstractVector{<:Integer}) =
      [inverse_transform(transformer, result, k) for k in w]

# inverse transforming vectors of categorical elements:
function MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          wcat::AbstractVector{<:CategoricalValue})
    w = MMI.int(wcat)
    return [inverse_transform(transformer, result, k) for k in w]
end

MMI.fitted_params(::UnivariateDiscretizer, fitresult) = (
    odd_quantiles=fitresult.odd_quantiles,
    even_quantiles=fitresult.even_quantiles
)

metadata_model(UnivariateDiscretizer,
    input_scitype   = AbstractVector{<:Continuous},
    output_scitype = AbstractVector{<:OrderedFactor},
    human_name = "single variable discretizer",
    load_path = "MLJTransforms.UnivariateDiscretizer")


"""
$(MLJModelInterface.doc_header(UnivariateDiscretizer))

Discretization converts a `Continuous` vector into an `OrderedFactor`
vector. In particular, the output is a `CategoricalVector` (whose
reference type is optimized).

The transformation is chosen so that the vector on which the
transformer is fit has, in transformed form, an approximately uniform
distribution of values. Specifically, if `n_classes` is the level of
discretization, then `2*n_classes - 1` ordered quantiles are computed,
the odd quantiles being used for transforming (discretization) and the
even quantiles for inverse transforming.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, x)

where

- `x`: any abstract vector with `Continuous` element scitype; check
  scitype with `scitype(x)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `n_classes`: number of discrete classes in the output


# Operations

- `transform(mach, xnew)`: discretize `xnew` according to the
  discretization learned when fitting `mach`

- `inverse_transform(mach, z)`: attempt to reconstruct from `z` a
  vector that transforms to give `z`


# Fitted parameters

The fields of `fitted_params(mach).fitesult` include:

- `odd_quantiles`: quantiles used for transforming (length is `n_classes - 1`)

- `even_quantiles`: quantiles used for inverse transforming (length is `n_classes`)


# Example

```
using MLJ
using Random
Random.seed!(123)

discretizer = UnivariateDiscretizer(n_classes=100)
mach = machine(discretizer, randn(1000))
fit!(mach)

julia> x = rand(5)
5-element Vector{Float64}:
 0.8585244609846809
 0.37541692370451396
 0.6767070590395461
 0.9208844241267105
 0.7064611415680901

julia> z = transform(mach, x)
5-element CategoricalArrays.CategoricalArray{UInt8,1,UInt8}:
 0x52
 0x42
 0x4d
 0x54
 0x4e

x_approx = inverse_transform(mach, z)
julia> x - x_approx
5-element Vector{Float64}:
 0.008224506144777322
 0.012731354778359405
 0.0056265330571125816
 0.005738175684445124
 0.006835652575801987
```

"""
UnivariateDiscretizer
