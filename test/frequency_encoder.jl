using MLJTransforms: frequency_encoder_fit, frequency_encoder_transform

# Possible dataset forms (X,y)
dataset_forms = []
push!(dataset_forms, create_dummy_dataset(:regression, as_dataframe=false, return_y=false))
push!(dataset_forms, create_dummy_dataset(:regression, as_dataframe=true, return_y=false))

@testset "Frequency Encoding Fit" begin
    @test frequency_encoder_fit(dataset_forms[1]) == frequency_encoder_fit(dataset_forms[2])

    X = dataset_forms[1]
    normalize = [false, true]
    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    for norm in normalize
        result = frequency_encoder_fit(X; normalize=norm)[:statistic_given_feat_val]
        enc = (col, level) -> ((norm) ? sum(col .== level)/length(col) : sum(col .== level))
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
@testset "Ordinal Encoding Transform" begin
    X = dataset_forms[1]
    normalize = [false, true]
    for norm in normalize
        cache = frequency_encoder_fit(X; normalize=norm)

        X_tr = frequency_encoder_transform(X, cache)

        enc = (col, level) -> ((norm) ? sum(X[col] .== level)/length(X[col]) : sum(X[col] .== level))

        target = (
            A = [enc(:A, X[:A][i]) for i in 1:10],
            B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            C = [enc(:C, X[:C][i]) for i in 1:10],
            D = [enc(:D, X[:D][i]) for i in 1:10],
            E = [1, 2, 3, 4, 5, 6, 6, 3, 2, 1],
            F = [enc(:F, X[:F][i]) for i in 1:10]
        )
        @test X_tr == target
    end
end