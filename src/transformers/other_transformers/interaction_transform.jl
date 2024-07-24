
@mlj_model mutable struct InteractionTransformer <: Static
    order::Int                                          = 2::(_ > 1)
    features::Union{Nothing, Vector{Symbol}}            = nothing::(_ !== nothing ? length(_) > 1 : true)
end

infinite_scitype(col) = eltype(scitype(col)) <: Infinite

actualfeatures(features::Nothing, table) =
    filter(feature -> infinite_scitype(Tables.getcolumn(table, feature)), Tables.columnnames(table))

function actualfeatures(features::Vector{Symbol}, table)
    diff = setdiff(features, Tables.columnnames(table))
    diff != [] && throw(ArgumentError(string("Column(s) ", join([x for x in diff], ", "), " are not in the dataset.")))

    for feature in features
        infinite_scitype(Tables.getcolumn(table, feature)) || throw(ArgumentError("Column $feature's scitype is not Infinite."))
    end
    return Tuple(features)
end

interactions(columns, order::Int) =
    collect(Iterators.flatten(combinations(columns, i) for i in 2:order))

interactions(columns, variables...) =
    .*((Tables.getcolumn(columns, var) for var in variables)...)

function MMI.transform(model::InteractionTransformer, _, X)
    features = actualfeatures(model.features, X)
    interactions_ = interactions(features, model.order)
    interaction_features = Tuple(Symbol(join(inter, "_")) for inter in interactions_)
    columns = Tables.Columns(X)
    interaction_table = NamedTuple{interaction_features}([interactions(columns, inter...) for inter in interactions_])
    return merge(Tables.columntable(X), interaction_table)
end

metadata_model(InteractionTransformer,
    input_scitype   = Tuple{Table},
    output_scitype = Table,
    human_name = "interaction transformer",
    load_path    = "MLJModels.InteractionTransformer")

"""
$(MLJModelInterface.doc_header(InteractionTransformer))

Generates all polynomial interaction terms up to the given order for the subset of chosen
columns.  Any column that contains elements with scitype `<:Infinite` is a valid basis to
generate interactions.  If `features` is not specified, all such columns with scitype
`<:Infinite` in the table are used as a basis.

In MLJ or MLJBase, you can transform features `X` with the single call

    transform(machine(model), X)

See also the example below.


# Hyper-parameters

- `order`: Maximum order of interactions to be generated.
- `features`: Restricts interations generation to those columns

# Operations

- `transform(machine(model), X)`: Generates polynomial interaction terms out of table `X`
  using the hyper-parameters specified in `model`.

# Example

```
using MLJ

X = (
    A = [1, 2, 3],
    B = [4, 5, 6],
    C = [7, 8, 9],
    D = ["x₁", "x₂", "x₃"]
)
it = InteractionTransformer(order=3)
mach = machine(it)

julia> transform(mach, X)
(A = [1, 2, 3],
 B = [4, 5, 6],
 C = [7, 8, 9],
 D = ["x₁", "x₂", "x₃"],
 A_B = [4, 10, 18],
 A_C = [7, 16, 27],
 B_C = [28, 40, 54],
 A_B_C = [28, 80, 162],)

it = InteractionTransformer(order=2, features=[:A, :B])
mach = machine(it)

julia> transform(mach, X)
(A = [1, 2, 3],
 B = [4, 5, 6],
 C = [7, 8, 9],
 D = ["x₁", "x₂", "x₃"],
 A_B = [4, 10, 18],)

```

"""
InteractionTransformer


