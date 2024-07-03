using MLJTransforms: union_types, cardinality_reducer_fit, cardinality_reducer_transform

@testset "Union_types" begin
    @test union_types(Union{Integer, String}) == (Integer, String)
end

@testset "Throws errors when needed" begin
    @test_throws ArgumentError begin
        X = generate_high_cardinality_table(1000; obj = false, special_cat = 'X')
        cache = cardinality_reducer_fit(
            X;
            label_for_infrequent = Dict(AbstractString => "Other", Char => 'X'),
        )
    end
    @test_throws ArgumentError begin
        X = generate_high_cardinality_table(1000; obj = false, special_cat = 'O')
        cache = cardinality_reducer_fit(
            X;
            label_for_infrequent = Dict(AbstractString => "Other", Bool => 'X'),
        )
    end
    @test_throws ArgumentError begin
        X = generate_high_cardinality_table(1000)
        cache = cardinality_reducer_fit(
            X;
            min_frequency = 20,
            label_for_infrequent = Dict(AbstractString => "X"),
        )
    end
end


@testset "Default for Numbers Set Correctly" begin
    X = generate_high_cardinality_table(1000)
    cache = cardinality_reducer_fit(X; min_frequency = 0.2)
    new_cat_given_col_val = cache[:new_cat_given_col_val]

    convert(Integer, minimum(values(new_cat_given_col_val[:HighCardFeature1])))
    convert(Integer, minimum(X.HighCardFeature1)) - 1
    
    @test convert(Integer, minimum(values(new_cat_given_col_val[:HighCardFeature1]))) ==
          (convert(Integer, minimum(X.HighCardFeature1)) - 1)
end


@testset "Equivalence of float and integer min_frequency" begin
    X = generate_high_cardinality_table(1000)
    for min_frequency in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        cache1 = cardinality_reducer_fit(X; min_frequency = min_frequency)
        new_cat_given_col_val1 = cache1[:new_cat_given_col_val]

        cache2 = cardinality_reducer_fit(X; min_frequency = Int.(min_frequency * 1000))
        new_cat_given_col_val2 = cache2[:new_cat_given_col_val]

        @test new_cat_given_col_val1 == new_cat_given_col_val2
    end
end

@testset "End-to-end test" begin
    X = generate_high_cardinality_table(1000)
    
    for min_frequency in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        cache = cardinality_reducer_fit(X; min_frequency = min_frequency)
        X_tr = cardinality_reducer_transform(X, cache)

        for col in [:LowCardFeature, :HighCardFeature1, :HighCardFeature2]
            new_prop_map = proportionmap(X_tr[!, col])
            for val in values(new_prop_map)
                @test ((val >= min_frequency) || (length(values(new_prop_map)) == 2))  # e.g., [A=0.7, O=0.3] is correct
            end
        end

    end

end



@testset "Cardinality Reducer Fit" begin
    X = generate_high_cardinality_table(1000)
    LowCardFeature_col, HighCardFeature1_col, HighCardFeature2_col =
        X[!, :1], X[:, :2], X[:, :3]
    result = cardinality_reducer_fit(
        X;
        min_frequency = 0.3,
        label_for_infrequent = Dict(AbstractString => "OtherOne", Char => 'X', Number => -99),
    )[:new_cat_given_col_val]

    enc_char = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : 'X')
    enc_num = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : -99)
    enc_str = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : "OtherOne")

    true_output = Dict{Symbol, Dict{Any, Any}}(
        :LowCardFeature => Dict(
            [
            (level, enc_char(LowCardFeature_col, level)) for
            level in levels(LowCardFeature_col)
        ],
        ),
        :HighCardFeature1 => Dict(
            [
            (level, enc_num(HighCardFeature1_col, level)) for
            level in levels(HighCardFeature1_col)
        ],
        ),
        :HighCardFeature2 => Dict(
            [
            (level, enc_str(HighCardFeature2_col, level)) for
            level in levels(HighCardFeature2_col)
        ],
        ),
    )
    @test result == true_output
end

# Redundant because it must work if generic transform work which has been tested before
@testset "Cardinality Reducer Transform" begin
    X = Tables.columntable(generate_high_cardinality_table(10))
    LowCardFeature_col, HighCardFeature1_col, HighCardFeature2_col =
        X[:LowCardFeature], X[:HighCardFeature1], X[:HighCardFeature2]
    cache = cardinality_reducer_fit(
        X;
        min_frequency = 0.3,
        label_for_infrequent = Dict(AbstractString => "OtherOne", Char => 'X', Number => -99),
    )

    enc_char = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : 'X')
    enc_num = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : -99)
    enc_str = (col, level) -> (proportionmap(col)[level] >= 0.3 ? level : "OtherOne")

    X_tr = cardinality_reducer_transform(X, cache)

    target = (
        LowCardFeature = [
            enc_char(X[:LowCardFeature], X[:LowCardFeature][i]) for i in 1:10
        ],
        HighCardFeature1 = [
            enc_num(X[:HighCardFeature1], X[:HighCardFeature1][i]) for i in 1:10
        ],
        HighCardFeature2 = [
            enc_str(X[:HighCardFeature2], X[:HighCardFeature2][i]) for i in 1:10
        ],
    )
    @test target == X_tr
end




@testset "MLJ Interface Cardinality Reducer" begin
    X = generate_high_cardinality_table(1000)
    # functional api
    generic_cache = cardinality_reducer_fit(X; ignore = true, ordered_factor = false)
    X_transf = cardinality_reducer_transform(X, generic_cache)
    # mlj api
    encoder = CardinalityReducer(ignore = true, ordered_factor = false)
    mach = machine(encoder, X)
    fit!(mach)
    Xnew_transf = MMI.transform(mach, X)

    # same output
    @test X_transf == Xnew_transf

    # fitted parameters is correct
    new_cat_given_col_val = fitted_params(mach).new_cat_given_col_val
    @test new_cat_given_col_val == generic_cache[:new_cat_given_col_val]

    # Test report
    @test report(mach) == (encoded_features = generic_cache[:encoded_features],)
end

# Look into MLJModelInterfaceTest
# Add tests to ensure categorical column properties are as expected