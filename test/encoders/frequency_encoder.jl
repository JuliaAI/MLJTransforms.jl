using MLJTransforms: frequency_encoder_fit, frequency_encoder_transform

@testset "Frequency Encoding Fit" begin
    @test frequency_encoder_fit(dataset_forms[1]) == frequency_encoder_fit(dataset_forms[2])

    X = dataset_forms[1]
    normalize = [false, true]
    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    for norm in normalize
        result = frequency_encoder_fit(X; normalize = norm)[:statistic_given_feat_val]
        enc =
            (col, level) ->
                Float32((norm) ? sum(col .== level) / length(col) : sum(col .== level))
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
end


# Redundant because it must work if fit works (only relies on generic_transform)
@testset "Frequency Encoding Transform" begin
    X = dataset_forms[1]
    normalize = [false, true]
    for norm in normalize
        cache = frequency_encoder_fit(X; normalize = norm)
        X_tr = frequency_encoder_transform(X, cache)
        enc =
            (col, level) ->
                Float32((norm) ? sum(X[col] .== level) / length(X[col]) : sum(X[col] .== level))

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
end


@testset "MLJ Interface Frequency Encoding" begin
    Xs = [dataset_forms[1], dataset_forms[2]]
    for X in Xs
        # functional api
        generic_cache = frequency_encoder_fit(X; ignore = true, ordered_factor = false)
        X_transf = frequency_encoder_transform(X, generic_cache)
        # mlj api
        encoder = FrequencyEncoder(ignore = true, ordered_factor = false)
        mach = machine(encoder, X)
        fit!(mach)
        Xnew_transf = MMI.transform(mach, X)

        # same output
        @test X_transf == Xnew_transf

        # fitted parameters is correct
        statistic_given_feat_val = fitted_params(mach).statistic_given_feat_val
        @test statistic_given_feat_val == generic_cache[:statistic_given_feat_val]

        # Test report
        @test report(mach) == (encoded_features = generic_cache[:encoded_features],)
    end
end

@testset "Test Frequency Encoding Output Types" begin
    # Define categorical features
    A = ["g", "b", "g", "r", "r"]
    B = [1.0, 2.0, 3.0, 4.0, 5.0]
    C = ["f", "f", "f", "m", "f"]
    D = [true, false, true, false, true]
    E = [1, 2, 3, 4, 5]

    # Combine into a named tuple
    X = (A = A, B = B, C = C, D = D, E = E)

    # Coerce A, C, D to multiclass and B to continuous and E to ordinal
    X = coerce(X,
        :A => Multiclass,
        :B => Continuous,
        :C => Multiclass,
        :D => Multiclass,
        :E => OrderedFactor,
    )

    # Check scitype coercions:
    schema(X)

    encoder = FrequencyEncoder(ordered_factor = false, normalize = false)
    mach = fit!(machine(encoder, X))
    Xnew = MMI.transform(mach, X)


    scs = schema(Xnew).scitypes
    ts  = schema(Xnew).types
    # Check scitypes correctness
    @test all(scs[1:end-1] .== Continuous)
    @test all(t -> (t <: AbstractFloat) && isconcretetype(t), ts[1:end-1])
    # Ordinal column should be intact
    @test scs[end] === schema(X).scitypes[end]
    @test ts[end] == schema(X).types[end]
end
