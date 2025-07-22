using MLJTransforms: generate_new_feat_names, generic_fit, generic_transform

# Initial setup
classification_forms = []
multiclassification_forms = []
regression_forms = []
dataset_forms = []          # X only (above is X,y)

# Add datasets to the classification forms vector
for form in [:binary, :binary_str]
    push!(classification_forms, create_dummy_dataset(form, as_dataframe = false))
    push!(classification_forms, create_dummy_dataset(form, as_dataframe = true))
end

# Add datasets to the multiclassification forms vector
for form in [:multiclass, :multiclass_str]
    push!(multiclassification_forms, create_dummy_dataset(form, as_dataframe = false))
    push!(multiclassification_forms, create_dummy_dataset(form, as_dataframe = true))
end

# Add datasets to the regression forms vector
push!(regression_forms, create_dummy_dataset(:regression, as_dataframe = false))
push!(regression_forms, create_dummy_dataset(:regression, as_dataframe = true))

# Add datasets to the dataset forms vector
push!(
    dataset_forms,
    create_dummy_dataset(:regression, as_dataframe = false, return_y = false),
)
push!(
    dataset_forms,
    create_dummy_dataset(:regression, as_dataframe = true, return_y = false),
)

@testset "Generate New feature names Function Tests" begin
    levels = ("A", "B", "C")

    # Test 1: No initial conflicts, indices mode (use_levelnames=false)
    @testset "No Initial Conflicts (Indices)" begin
        existing_names = Symbol[]
        names = generate_new_feat_names(
            "feat",
            2,
            levels,
            existing_names;
            use_levelnames = false,
        )
        @test names == [Symbol("feat_1"), Symbol("feat_2")]
    end

    # Test 2: No conflicts, level-names mode (default use_levelnames=true)
    @testset "No Initial Conflicts (Level Names)" begin
        existing_names = Symbol[]
        names = generate_new_feat_names("feat", 3, levels, existing_names)
        @test names == [Symbol("feat_A"), Symbol("feat_B"), Symbol("feat_C")]
    end

    # Test 3: Handle initial conflict by adding underscores (indices)
    @testset "Initial Conflict Resolution (Indices)" begin
        existing_names = [Symbol("feat_1"), Symbol("feat_2")]
        names = generate_new_feat_names(
            "feat",
            2,
            levels,
            existing_names;
            use_levelnames = false,
        )
        @test names == [Symbol("feat__1"), Symbol("feat__2")]
    end

    # Test 4: Handle initial conflict by adding underscores (level names)
    @testset "Initial Conflict Resolution (Level Names)" begin
        existing_names = [Symbol("feat_A"), Symbol("feat_B"), Symbol("feat_C")]
        names = generate_new_feat_names("feat", 3, levels, existing_names)
        @test names == [Symbol("feat__A"), Symbol("feat__B"), Symbol("feat__C")]
    end
end


# Dummy encoder that maps each level to its hash (some arbitrary function)
function dummy_encoder_fit(
    X,
    features = Symbol[];
    ignore::Bool = true,
    ordered_factor::Bool = false,
)
    # 1. Define feature mapper
    function feature_mapper(col, name)
        feat_levels = levels(col)
        hash_given_feat_val =
            Dict{Any, Integer}(value => hash(value) for value in feat_levels)
        return hash_given_feat_val
    end

    # 2. Pass it to generic_fit
    hash_given_feat_val, encoded_features = generic_fit(
        X, features; ignore = ignore, ordered_factor = ordered_factor,
        feature_mapper = feature_mapper,
    )
    cache = (
        hash_given_feat_val = hash_given_feat_val,
        encoded = encoded_features,
    )
    return cache
end

function dummy_encoder_transform(X, cache::NamedTuple)
    hash_given_feat_val = cache.hash_given_feat_val
    return generic_transform(X, hash_given_feat_val)
end



@testset "Column inclusion and exclusion for fit" begin
    X = dataset_forms[1]

    # test exclude features
    feat_names = Tables.schema(X).names
    ignore_cols = [rand(feat_names), rand(feat_names)]
    hash_given_feat_val =
        dummy_encoder_fit(X, ignore_cols; ignore = true, ordered_factor = false)[:hash_given_feat_val]
    @test intersect(keys(hash_given_feat_val), ignore_cols) == Set()

    # test include features
    feat_names = [:A, :C, :D, :F]        # these are multiclass
    include_cols = [rand(feat_names), rand(feat_names)]
    hash_given_feat_val2 =
        dummy_encoder_fit(X, include_cols; ignore = false, ordered_factor = false)[:hash_given_feat_val]
    @test intersect(keys(hash_given_feat_val2), include_cols) == Set(include_cols)

    # test types of encoded features
    feat_names = Tables.schema(X).names
    hash_given_feat_val =
        dummy_encoder_fit(X, Symbol[]; ignore = true, ordered_factor = false)[:hash_given_feat_val]
    @test !(:E in keys(hash_given_feat_val))
    hash_given_feat_val =
        dummy_encoder_fit(X, Symbol[]; ignore = true, ordered_factor = true)[:hash_given_feat_val]
    @test (:E in keys(hash_given_feat_val))
end



@testset "Test generic fit output" begin
    X = dataset_forms[1]
    A_col, C_col, D_col, F_col = selectcols(X, [1, 3, 4, 6])
    result = dummy_encoder_fit(X)[:hash_given_feat_val]
    enc = (col, level) -> (hash(level))
    true_output = Dict{Symbol, Dict{Any, Any}}(
        :F => Dict(
            "m" => enc(F_col, "m"),
            "l" => enc(F_col, "l"),
            "s" => enc(F_col, "s"),
        ),
        :A => Dict(
            "g" => enc(A_col, "g"),
            "b" => enc(A_col, "b"),
            "r" => enc(A_col, "r"),
        ),
        :D => Dict(
            false => enc(D_col, false),
            true => enc(D_col, true),
        ),
        :C => Dict(
            "f" => enc(C_col, "f"),
            "m" => enc(C_col, "m"),
        ),
    )
    @test result == true_output
end

@testset "Test generic transform" begin
    X = dataset_forms[1]
    cache = dummy_encoder_fit(X)
    X_tr = dummy_encoder_transform(X, cache)

    enc = (col, level) -> (hash(level))

    target = (
        A = [enc(:A, X[:A][i]) for i in 1:10],
        B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        C = [enc(:C, X[:C][i]) for i in 1:10],
        D = [enc(:D, X[:D][i]) for i in 1:10],
        E = [1, 2, 3, 4, 5, 6, 6, 3, 2, 1],
        F = [enc(:F, X[:F][i]) for i in 1:10],
    )
    @test X_tr == target
end

@testset "Callable feature functionality tests" begin
    X = dataset_forms[1]
    feat_names = Tables.schema(X).names

    # Define a predicate: include only columns with name in uppercase list [:A, :C, :E]
    predicate = name -> name in [:A, :C, :E]

    # Test 1: ignore=true should exclude predicate columns
    cache1 = dummy_encoder_fit(X, predicate; ignore = true, ordered_factor = false)
    @test !(:A in cache1[:encoded]) && !(:C in cache1[:encoded]) &&
          !(:E in cache1[:encoded])

    # Test 2: ignore=false should include only predicate columns
    cache2 = dummy_encoder_fit(X, predicate; ignore = false, ordered_factor = false)
    @test Set(cache2[:encoded]) == Set([:A, :C])

    # Test 3: predicate with ordered_factor=true picks up ordered factors (e.g., :E)
    cache3 = dummy_encoder_fit(X, predicate; ignore = false, ordered_factor = true)
    @test Set(cache3[:encoded]) == Set([:A, :C, :E])
end

@testset "Single Symbol and list of one symbol equivalence" begin
    X = dataset_forms[1]
    feat_names = Tables.schema(X).names

    # Test 1: Single Symbol
    single_symbol = :A
    cache1 = dummy_encoder_fit(X, single_symbol; ignore = true, ordered_factor = false)
    @test !(:A in cache1[:encoded])
    # Test 2: List of one symbol
    single_symbol_list = [:A]
    cache2 = dummy_encoder_fit(X, single_symbol_list; ignore = true, ordered_factor = false)
    @test !(:A in cache2[:encoded])
    # Test 3: Both should yield the same result
    @test cache1[:encoded] == cache2[:encoded]
end
