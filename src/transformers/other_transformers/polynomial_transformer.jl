# # STRUCT AND CONSTRUCTORS

const WARN_DEGREE = "The `degree` must be at least 1. "*
    "Reset `degree=2`. "


mutable struct PolynomialTransformer <: Static
    degree::Int
    features::Union{Nothing, Vector{Symbol}}
    interactions_only::Bool
end

function MMI.clean!(model::PolynomialTransformer)
    message = ""
    if model.degree ≤ 0
        model.degree = 2
        message *= WARN_DEGREE
    end
    return message
end

function PolynomialTransformer(
    ; order=2,
    degree=order,
    features=nothing,
    interactions_only=false,
    )
    model = PolynomialTransformer(degree, features, interactions_only)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


# # HELPERS

"""
    orderedwords(alphabet, len)

If `alphabet = [:x, y:], then `[:x', :x']`, `[:x', :y]`, and `[:y, :y]` are
*ordered* words (of length two) but `[:y, :x']` is not.

# Example

```julia-repl
julia> orderedwords([:x, :y, :z], 2)
6-element Vector{Vector{String}}:
 6-element Vector{Vector{Symbol}}:
 [:x, :x]
 [:x, :y]
 [:x, :z]
 [:y, :y]
 [:y, :z]
 [:z, :z]
```
"""
orderedwords(alphabet::Union{AbstractVector{T},NTuple{N,T}}, len) where {T,N} =
    _orderedwords(alphabet, len, Vector{T}[])
# recursive part:
function _orderedwords(
    alphabet::Union{AbstractVector{T},NTuple{N,T}},
    len,
    shorter_words,
    ) where {T,N}
    isempty(shorter_words) && len == 0 && return shorter_words
    if isempty(shorter_words)
        longer_words = map(letter->T[letter], alphabet)
    else
        length(first(shorter_words)) == len && return shorter_words
        longer_words = Vector{T}[]
        for wrd in shorter_words
            idx = findfirst(==(last(wrd)), alphabet)
            for letter in alphabet[idx:end]
                push!(longer_words, [wrd..., letter])
            end
        end
    end
    return _orderedwords(alphabet, len, longer_words)
end

abstract type Selection end
struct WithRepetitions <: Selection end
struct WithoutRepetitions <: Selection end

"""
    premonomials(alphabet, degree, kind_of_selection::Selection)

*Private method* to help generate monomials. A **pre-monomial** is a vector with elements
 from the alphabet, with possible repetitions, but with no element predecessor coming
 *after* the element itself in the alphaget.

Note degree one "pre-monomials" are excluded.

# Example

```julia-repl
julia> premonomials((:x, :y, :z), 3, WithoutRepetitions())
4-element Vector{Vector{Symbol}}:
 [:x, :y]
 [:x, :z]
 [:y, :z]
 [:x, :y, :z]

julia> premonomials((:x, :y, :z), 3, WithRepetitions())
16-element Vector{Vector{Symbol}}:
 [:x, :x]
 [:x, :y]
 [:x, :z]
 [:y, :y]
 [:y, :z]
 [:z, :z]
 [:x, :x, :x]
 [:x, :x, :y]
 [:x, :x, :z]
 [:x, :y, :y]
 [:x, :y, :z]
 [:x, :z, :z]
 [:y, :y, :y]
 [:y, :y, :z]
 [:y, :z, :z]
 [:z, :z, :z]
```
"""
premonomials(alphabet, degree, ::WithoutRepetitions) =
    premonomials(alphabet, degree, Combinatorics.combinations)
premonomials(alphabet, degree, ::WithRepetitions) =
    premonomials(alphabet, degree, orderedwords)
premonomials(alphabet, degree, fnctn) =
    collect(Iterators.flatten(fnctn(alphabet, i) for i in 2:degree))

column_product(columns, premonomial...) =
    .*((Tables.getcolumn(columns, feature) for feature in premonomial)...)


# # CORE IMPLEMENTATION

function MMI.transform(model::PolynomialTransformer, _, X)
    features = MLJTransforms.actualfeatures(model.features, X)
    kind_of_selection = model.interactions_only ? WithoutRepetitions() : WithRepetitions()
    premonomials = MLJTransforms.premonomials(features, model.degree, kind_of_selection)
    new_features = Tuple(Symbol(join(premon, "_")) for premon in premonomials)
    materializer = Tables.materializer(X)
    columns = Tables.Columns(X)
    table_addendum =
        NamedTuple{new_features}(
            [column_product(columns, premon...) for premon in premonomials],
        )
    return merge(Tables.columntable(X), table_addendum) |> materializer
end


# # TRAITS

metadata_model(PolynomialTransformer,
    input_scitype   = Tuple{Table},
    output_scitype = Table,
    human_name = "polynomial transformer",
    load_path = "MLJTransforms.PolynomialTransformer")

# Package metadata for docstring generation
metadata_pkg(PolynomialTransformer,
    package_name = "MLJTransforms",
    package_uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6",
    package_url = "https://github.com/JuliaAI/MLJTransforms.jl",
    is_pure_julia = true,
    package_license = "MIT")

"""
$(MLJModelInterface.doc_header(PolynomialTransformer))

This `Static` transformer generates new features comprised of monomials in existing
features that have `Continuous` or `Count` scitype, up to some specified degree. A
restricted set of features may be specified, and one may elect to generate only
interaction monomials (no feature appearing with degree higher than one).

In MLJ or MLJBase, you can transform features `X` with the single call

    transform(machine(model), X)

See also the example below.


# Hyper-parameters

- `degree=2`: maximum degree of monomials to be generated

- `features=nothing`: vector of features for which monomials should be generated; if
  `nothing` (unspecified) then all `Continuous` and `Count` features are used.

# Operations

- `transform(machine(model), X)`: Generate a new table from `X` with the monomial columnn
  specified by hyper-parameters.

# Example

```
using MLJ

X = (
    A = [1, 2, 3],
    B = [4, 5, 6],
    C = [7, 8, 9],
    D = ["cat", "dog", "rat"]
)

transformer = PolynomialTransformer(degree=2, features=[:A, :B])
mach = machine(transformer)
julia> transform(mach, X) |> pretty
┌───────┬───────┬───────┬─────────┬───────┬───────┬───────┐
│ A     │ B     │ C     │ D       │ A_A   │ A_B   │ B_B   │
│ Int64 │ Int64 │ Int64 │ String  │ Int64 │ Int64 │ Int64 │
│ Count │ Count │ Count │ Textual │ Count │ Count │ Count │
├───────┼───────┼───────┼─────────┼───────┼───────┼───────┤
│ 1     │ 4     │ 7     │ cat     │ 1     │ 4     │ 16    │
│ 2     │ 5     │ 8     │ dog     │ 4     │ 10    │ 25    │
│ 3     │ 6     │ 9     │ rat     │ 9     │ 18    │ 36    │
└───────┴───────┴───────┴─────────┴───────┴───────┴───────┘

transformer = PolynomialTransformer(degree=3, interactions_only=true)
mach = machine(transformer)
julia> transform(mach, X) |> pretty
┌───────┬───────┬───────┬─────────┬───────┬───────┬───────┬───────┐
│ A     │ B     │ C     │ D       │ A_B   │ A_C   │ B_C   │ A_B_C │
│ Int64 │ Int64 │ Int64 │ String  │ Int64 │ Int64 │ Int64 │ Int64 │
│ Count │ Count │ Count │ Textual │ Count │ Count │ Count │ Count │
├───────┼───────┼───────┼─────────┼───────┼───────┼───────┼───────┤
│ 1     │ 4     │ 7     │ cat     │ 4     │ 7     │ 28    │ 28    │
│ 2     │ 5     │ 8     │ dog     │ 10    │ 16    │ 40    │ 80    │
│ 3     │ 6     │ 9     │ rat     │ 18    │ 27    │ 54    │ 162   │
└───────┴───────┴───────┴─────────┴───────┴───────┴───────┴───────┘

```

"""
PolynomialTransformer
