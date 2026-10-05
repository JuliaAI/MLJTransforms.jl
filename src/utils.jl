# add utility functions here

has_infinite_scitype(col) = scitype(col) <:AbstractVector{<:Union{Missing,Infinite}}

# method to extrac, `Infinite` scitype features from a table, given a subset of features.
actualfeatures(features::Nothing, table) =
    filter(Tables.columnnames(table)) do feature
        MLJTransforms.has_infinite_scitype(Tables.getcolumn(table, feature))
    end
function actualfeatures(features::Vector{Symbol}, table)
    diff = setdiff(features, Tables.columnnames(table))
    diff != [] &&
        throw(ArgumentError(string(
            "Column(s) ",
            join([x for x in diff], ", "),
            " are not in the dataset."),
                            )
              )
    for feature in features
        MLJTransforms.has_infinite_scitype(Tables.getcolumn(table, feature)) ||
            throw(ArgumentError("Column $feature's scitype is not Infinite."))
    end
    return Tuple(features)
end
