# To go from e.g., Union{Integer, String} to (Integer, String)
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)