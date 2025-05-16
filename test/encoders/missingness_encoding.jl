using MLJTransforms: missingness_encoder_fit, missingness_encoder_transform

@testset "Missingness Encoder Error Handling" begin
    # Test COLLISION_NEW_VAL_ME error - when label_for_missing value already exists in levels
    @test_throws MLJTransforms.COLLISION_NEW_VAL_ME("missing") begin
        X = generate_X_with_missingness(;john_name="missing")
        cache = missingness_encoder_fit(
            X;
            label_for_missing = Dict(AbstractString => "missing", Char => 'm'),
        )
    end
    
    # Test VALID_TYPES_NEW_VAL_ME error - when label_for_missing key is not a supported type
    @test_throws MLJTransforms.VALID_TYPES_NEW_VAL_ME(Bool) begin
        X = generate_X_with_missingness()
        cache = missingness_encoder_fit(
            X;
            label_for_missing = Dict(AbstractString => "Other", Bool => 'X'),
        )
    end
    
    # Test UNSPECIFIED_COL_TYPE_ME error - when column type isn't in label_for_missing
    @test_throws MLJTransforms.UNSPECIFIED_COL_TYPE_ME(Char, Dict(AbstractString => "X")) begin
        X = generate_X_with_missingness()
        cache = missingness_encoder_fit(
            X;
            label_for_missing = Dict(AbstractString => "X"),
        )
    end
end


@testset "Default for Numbers Set Correctly" begin
    X = generate_X_with_missingness()
    cache = missingness_encoder_fit(X)
    label_for_missing_given_feature = cache[:label_for_missing_given_feature]
    @test label_for_missing_given_feature[:C][missing] == minimum(levels(X.C)) - 1
end


@testset "End-to-end test" begin
    X = generate_X_with_missingness()
    
    cache = missingness_encoder_fit(X; label_for_missing = Dict(AbstractString => "missing-item", Char => 'i', Number => -99))
    X_tr = missingness_encoder_transform(X, cache)

    for col in [:A, :B, :C, :D, :E]
        @test issubset(levels(X[col]), levels(X_tr[col]))
    end

    @test Set(push!(levels(X[:A]), "missing-item")) == Set(levels(X_tr[:A]))
    @test Set(push!(levels(X[:C]), -99)) == Set(levels(X_tr[:C]))
    @test Set(push!(levels(X[:E]), 'i')) == Set(levels(X_tr[:E]))
    @test levels(X[:B]) == levels(X_tr[:B])
    @test levels(X[:D]) == levels(X_tr[:D])
end
 

@testset "Missingness Encoder Fit" begin
    X = generate_X_with_missingness()

    result = missingness_encoder_fit(
        X;
        label_for_missing = Dict(AbstractString => "MissingOne", Char => 'X', Number => -90),
    )[:label_for_missing_given_feature]

    true_output = Dict{Symbol, Dict{Any, Any}}(
        :A => Dict([(missing, "MissingOne")]),
        :C => Dict([(missing, -90)]),
        :E => Dict([(missing, 'X')]),
    )
    @test result == true_output
end

# Redundant because it must work if generic transform work which has been tested before
@testset "Missingness Encoder Transform" begin
    X = generate_X_with_missingness()
    cache = missingness_encoder_fit(
        X;
        label_for_missing = Dict(AbstractString => "MissingOne", Char => 'X', Number => -90),
    )

    enc_char = (col, level) ->  ismissing(level) ? 'X' : level
    enc_num = (col, level) ->  ismissing(level) ? -90 : level
    enc_str = (col, level) ->  ismissing(level) ? "MissingOne" : level
    enc_idn = (col, level) -> level

    X_tr = missingness_encoder_transform(X, cache)

    target = (
        A = [
            enc_str(X[:A], X[:A][i]) for i in 1:7
        ],
        B = [
            enc_idn(X[:B], X[:B][i]) for i in 1:7
        ],
        C = [
            enc_num(X[:C], X[:C][i]) for i in 1:7
        ],
        D = [
            enc_idn(X[:D], X[:D][i]) for i in 1:7
        ],
        E = [
            enc_char(X[:E], X[:E][i]) for i in 1:7
        ]
    )

    @test isequal(target, X_tr)
end

@testset "Schema doesn't change after transform" begin
    X = generate_X_with_missingness()
    
    cache = missingness_encoder_fit(
        X;
        label_for_missing = Dict(AbstractString => "MissingOne", Char => 'X', Number => -90),
    )

    X_tr = missingness_encoder_transform(X, cache)

    @test elscitype(X_tr[:A]) <: Multiclass
    @test elscitype(X_tr[:B]) <: Union{Missing, Continuous}
    @test elscitype(X_tr[:C]) <: Multiclass
    @test elscitype(X_tr[:D]) <: Count
    @test elscitype(X_tr[:E]) <: Multiclass
end

@testset "Adding new levels" begin
    X = generate_X_with_missingness()
    levels!(Tables.getcolumn(X, :A), ["Ben", "John", "Mary", "Max"])

    cache = missingness_encoder_fit(
        X;
        label_for_missing = Dict(AbstractString => "MissingOne", Char => 'X', Number => -90),
    )
    X_tr = missingness_encoder_transform(X, cache)

    @test issubset(levels(X[:A]), levels(X_tr[:A])) # Will have "MissingOne" added
end

@testset "MLJ Interface Missingness Encoder" begin
    X = generate_X_with_missingness()
    # functional api
    generic_cache = missingness_encoder_fit(X; ignore = true, ordered_factor = false)
    X_transf = missingness_encoder_transform(X, generic_cache)
    # mlj api
    encoder = MissingnessEncoder(ignore = true, ordered_factor = false)
    mach = machine(encoder, X)
    fit!(mach)
    Xnew_transf = MMI.transform(mach, X)

    # same output
    @test isequal(X_transf, Xnew_transf)

    # fitted parameters is correct
    label_for_missing_given_feature = fitted_params(mach).label_for_missing_given_feature
    @test label_for_missing_given_feature == generic_cache[:label_for_missing_given_feature]

    # Test report
    @test report(mach) == (encoded_features = generic_cache[:encoded_features],)
end