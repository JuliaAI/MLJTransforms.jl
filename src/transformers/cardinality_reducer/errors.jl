UNSUPPORTED_COL_TYPE(col_type) =
    "In CardinalityReducer, elements have type $(col_type). The supported types are $(ScientificTypes.SupportedTypes)"
VALID_TYPES_NEW_VAL(possible_col_type) =
    "In CardinalityReducer, infreq_val keys have type $(possible_col_type). The supported types are $(ScientificTypes.SupportedTypes)"
COLLISION_NEW_VAL(value) =
    "In CardinalityReducer, infreq_val specifies new column name $(value). However, this name already exists in one of the columns. Please respecify infreq_val."
UNSPECIFIED_COL_TYPE(col_type, infreq_val) =
    "In CardinalityReducer, $(col_type) does not appear in infreq_val which only has keys $(keys(infreq_val))"
