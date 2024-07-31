UNSUPPORTED_COL_TYPE_ME(col_type) =
    "In MissingnessEncoder, elements have type $(col_type). The supported types are `Char`, `AbstractString`, and `Number`"
VALID_TYPES_NEW_VAL_ME(possible_col_type) =
    "In MissingnessEncoder, label_for_missing keys have type $(possible_col_type). The supported types are `Char`, `AbstractString`, and `Number`"
COLLISION_NEW_VAL_ME(value) =
    "In MissingnessEncoder, label_for_missing specifies new feature name $(value). However, this name already exists in one of the features. Please respecify label_for_missing."
UNSPECIFIED_COL_TYPE_ME(col_type, label_for_missing) =
    "In MissingnessEncoder, $(col_type) does not appear in label_for_missing which only has keys $(keys(label_for_missing))"

