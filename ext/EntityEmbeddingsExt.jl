module EntityEmbeddingsExt

using MLJFlux
using Tables
using ScientificTypes
using MLJModelInterface
using TableOperations
using Optimisers
using Flux
using MLJBase
using MLJTransforms
using MLJTransforms: EntityEmbedder
const MMI = MLJModelInterface

# activations
function MLJTransforms.get_activation(func_symbol::Symbol)
    if hasproperty(Flux, func_symbol)
        return getproperty(Flux, func_symbol)
    else
        error("Function $func_symbol not found in Flux.")
    end
end

function MLJTransforms.entity_embedder_fit(
    X,
    y,
    features::AbstractVector{Symbol} = Symbol[];
    ignore::Bool = true,
    hidden_layer_sizes::Tuple{Vararg{Int}} = (5,),
    activation::Symbol = :relu,
    epochs = 100,
    batch_size = 32,
    learning_rate = 0.01,
    embedding_dims::Dict{Symbol, Real} = Dict{Symbol, Real}(),
    verbosity::Int = 0,
    kwargs...,
)

    # Figure out task
    y_scitype = elscitype(y)
    classification_types = (y_scitype <: Multiclass || y_scitype <: OrderedFactor)
    regression_types = (y_scitype <: Continuous || y_scitype <: Count)
    task =
        regression_types ? :Regression :
        classification_types ? :Classification : :Unknown
    task == :Unknown && error(
        "Your target must be Continuous/Count for regression or Multiclass/OrderedFactor for classification",
    )

    # Handle ignore and given feat names
    feat_names_org = Tables.schema(X).names
    feat_names =
        (ignore) ? setdiff(feat_names_org, features) : intersect(feat_names_org, features)

    feat_inds_cat = [
        findfirst(feat_names .== feat_name) for
        feat_name in feat_names if elscitype(Tables.getcolumn(X, feat_name)) <: Finite
    ]

    # Select only the relevant columns in `X` based on `feat_names`
    X = X |> TableOperations.select(feat_names...) |> Tables.columntable


    # Setup builder
    builder = MLJFlux.MLP(;
        hidden = hidden_layer_sizes,
        σ = MLJTransforms.get_activation(activation),
    )

    # Accordingly fit NeuralNetworkRegressor, NeuralNetworkClassifier
    clf =
        (task == :Classification) ?
        MLJFlux.NeuralNetworkClassifier(
            builder = builder,
            optimiser = Optimisers.Adam(learning_rate),
            batch_size = batch_size,
            epochs = epochs,
            embedding_dims = embedding_dims;
            kwargs...,
        ) :
        MLJFlux.NeuralNetworkRegressor(
            builder = builder,
            optimiser = Optimisers.Adam(learning_rate),
            batch_size = batch_size,
            epochs = epochs,
            embedding_dims = embedding_dims;
            kwargs...,
        )

    # Fit the model
    mach = machine(clf, X, y)
    fit!(mach, verbosity = verbosity)

    # Get mappings

    mapping_matrices = MLJFlux.get_embedding_matrices(
        fitted_params(mach).chain,
        feat_inds_cat,
        feat_names,
    )
    ordinal_mappings = mach.fitresult[3]
    cache = (
        mapping_matrices = mapping_matrices,
        ordinal_mappings = ordinal_mappings,
        task = task,
        machine = mach,
    )
    return cache
end


"""
Given X and a dict of mapping_matrices that map each categorical column to a matrix, use the matrix to transform
each level in each categorical columns using the columns of the matrix.

This is used with the embedding matrices of the entity embedding layer in entity enabled models to implement entity embeddings.
"""
function MLJTransforms.entity_embedder_transform(X, cache)
    mach = cache[:machine]
    Xnew = MLJFlux.transform(mach, X)
    return Xnew
end

include("EntityEmbeddingsInterface.jl")


end