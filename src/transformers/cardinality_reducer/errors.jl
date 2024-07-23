UNSUPPORTED_COL_TYPE(col_type) =
    "In CardinalityReducer, elements have type $(col_type). The supported types are `Union{Char, Number, AbstractString}`"
VALID_TYPES_NEW_VAL(possible_col_type) =
    "In CardinalityReducer, label_for_infrequent keys have type $(possible_col_type). The supported types are `Union{Char, Number, AbstractString}`"
COLLISION_NEW_VAL(value) =
    "In CardinalityReducer, label_for_infrequent specifies new column name $(value). However, this name already exists in one of the columns. Please respecify label_for_infrequent."
UNSPECIFIED_COL_TYPE(col_type, label_for_infrequent) =
    "In CardinalityReducer, $(col_type) does not appear in label_for_infrequent which only has keys $(keys(label_for_infrequent))"
