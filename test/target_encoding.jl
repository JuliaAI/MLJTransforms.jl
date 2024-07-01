using MLJTransforms: compute_label_freq_for_level, compute_label_freqs_for_level,
    compute_target_mean_for_level, compute_shrinkage, compute_m_auto, mix_stats,
    target_encoder_fit, generate_new_feat_names, target_encoder_transform, TargetEncoder

@testset "Test statistic computation" begin
    targets_for_level = [1, 0, 1, 1, 0, 0, 1]
    y_classes = [1, 0]
    @test compute_label_freq_for_level(targets_for_level, y_classes) == 4 / 7

    targets_for_level = [3, 3, 3, 3, 3]
    y_classes = [3, 2]
    @test compute_label_freq_for_level(targets_for_level, y_classes) == 1.0

    targets_for_level = [1, 2, 3, 4, 5]
    @test compute_target_mean_for_level(targets_for_level) == 3.0

    targets_for_level = [1, 2, 2, 3, 3, 3]
    y_classes = [1, 2, 3]
    expected_freqs = [1 / 6, 2 / 6, 3 / 6]
    @test compute_label_freqs_for_level(targets_for_level, y_classes) == expected_freqs
end

@testset "Compute_shrinkage tests" begin
    targets_for_level = [1, 2, 3, 4, 5]
    
    @test compute_shrinkage(targets_for_level, m=0) == 1.0
    @test compute_shrinkage(targets_for_level, m=10) == 5 / 15
    
    targets_for_level = [10, 20, 30]
    
    @test compute_shrinkage(targets_for_level, m=5) == 3 / 8
    @test compute_shrinkage(targets_for_level, m=3, λ=0.5) == 0.5
end

@testset "Compute_m_auto tests" begin
    targets_for_level = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_var = 2.0
    @test compute_m_auto("Regression", targets_for_level, y_var=y_var) ≈ 2.0 / 2.5
    
    targets_for_level = [10.0, 20.0, 30.0, 40.0, 50.0]
    y_var = 100.0
    @test compute_m_auto("Regression", targets_for_level, y_var=y_var) ≈ 100.0 / 250.0
    @test_throws ErrorException compute_m_auto("Classification", targets_for_level, y_var=y_var)
end


@testset "Fit for binary classification dataset" begin
    # Test consistency of output
    results = []
    for (X, y) in classification_forms
        y_stat_given_feat_level =
            target_encoder_fit(X, y; ignore = true, ordered_factor = false)[:y_stat_given_feat_level]
        push!(results, y_stat_given_feat_level)
    end
    
    @test all(x -> x == results[1], results)

    # Test correctness of output
    X, y = classification_forms[1]
    n = length(y)

    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    true_output = Dict{Symbol, Dict{Any, AbstractFloat}}(
        :F => Dict(
            "m" => sum(y[F_col.=="m"] .== 0) / length(y[F_col.=="m"]),
            "l" => sum(y[F_col.=="l"] .== 0) / length(y[F_col.=="l"]),
            "s" => sum(y[F_col.=="s"] .== 0) / length(y[F_col.=="s"]),
        ),
        :A => Dict(
            "g" => sum(y[A_col.=="g"] .== 0) / length(y[A_col.=="g"]),
            "b" => sum(y[A_col.=="b"] .== 0) / length(y[A_col.=="b"]),
            "r" => sum(y[A_col.=="r"] .== 0) / length(y[A_col.=="r"]),
            ),
        :D => Dict(
            false => sum(y[D_col.==false] .== 0) / length(y[D_col.==false]),
            true => sum(y[D_col.==true] .== 0) / length(y[D_col.==true]),
            ),
        :C => Dict(
            "f" => sum(y[C_col.=="f"] .== 0) / length(y[C_col.=="f"]),
            "m" => sum(y[C_col.=="m"] .== 0) / length(y[C_col.=="m"]),
            ),
    )
    @test results[1] == true_output

    # Test mixing works in the edge case
    P̂ = length(y[y .== 0])/length(y)

    y_stat_given_feat_level =
        target_encoder_fit(X, y; ignore = true, ordered_factor = false, m = Inf)[:y_stat_given_feat_level]

    true_output = Dict{Symbol, Dict{Any, AbstractFloat}}(
        :F => Dict("m" => P̂, "l" => P̂, "s" => P̂,),
        :A => Dict("g" => P̂, "b" => P̂, "r" => P̂,),
        :D => Dict(false => P̂, true => P̂,),
        :C => Dict("f" => P̂, "m" => P̂,),
    )
    @test y_stat_given_feat_level == true_output

end


@testset "Fit for regression dataset" begin
    # Test output consistency
    results = []
    for (X, y) in regression_forms
        y_stat_given_feat_level =
            target_encoder_fit(X, y; ignore = true, ordered_factor = false)[:y_stat_given_feat_level]
        push!(results, y_stat_given_feat_level)
    end
    
    @test all(x -> x == results[1], results)

    # Test output corretness
    X, y = regression_forms[1]
    n = length(y)
    μ̂ = mean(y)

    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    true_output = Dict{Symbol, Dict{Any, AbstractFloat}}(
        :F => Dict(
            "m" => mean(y[F_col.=="m"]),
            "l" => mean(y[F_col.=="l"]),
            "s" => mean(y[F_col.=="s"]),
        ),
        :A => Dict(
            "g" => mean(y[A_col.=="g"]),
            "b" => mean(y[A_col.=="b"]),
            "r" => mean(y[A_col.=="r"]),
            ),
        :D => Dict(
            false => mean(y[D_col.==false]) ,
            true => mean(y[D_col.==true]),
            ),
        :C => Dict(
            "f" => mean(y[C_col.=="f"]),
            "m" => mean(y[C_col.=="m"]),
            ),
    )
    @test results[1] == true_output

    # Test mixing in the edge case
    y_stat_given_feat_level =
    target_encoder_fit(X, y; ignore = true, ordered_factor = false, m = Inf)[:y_stat_given_feat_level]

    true_output = Dict{Symbol, Dict{Any, AbstractFloat}}(
        :F => Dict("m" => μ̂, "l" => μ̂, "s" => μ̂,),
        :A => Dict("g" => μ̂, "b" => μ̂, "r" => μ̂,),
        :D => Dict(false => μ̂, true => μ̂,),
        :C => Dict("f" => μ̂, "m" => μ̂,),
    )
    @test y_stat_given_feat_level == true_output
end

@testset "Fit for multiclassification dataset" begin

    # Test output consistency
    results = []
    for (X, y) in multiclassification_forms
        y_stat_given_feat_level =
            target_encoder_fit(X, y; ignore = true, ordered_factor = false)[:y_stat_given_feat_level]
        push!(results, y_stat_given_feat_level)
    end

    @test all(x -> x == results[1], results)

    # Test output corretness
    X, y = multiclassification_forms[1]
    y_classes = classes(y)
    n = length(y)

    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    true_output = Dict{Symbol, Dict{Any, AbstractVector{AbstractFloat}}}(
        :F => Dict(
            "m" => [sum(y[F_col.=="m"] .== l) for l in y_classes] ./ length(y[F_col.=="m"]),
            "l" => [sum(y[F_col.=="l"] .== l) for l in y_classes] ./ length(y[F_col.=="l"]),
            "s" => [sum(y[F_col.=="s"] .== l) for l in y_classes] ./ length(y[F_col.=="s"]),
        ),
        :A => Dict(
            "g" => [sum(y[A_col.=="g"] .== l) for l in y_classes] ./ length(y[A_col.=="g"]),
            "b" => [sum(y[A_col.=="b"] .== l) for l in y_classes] ./ length(y[A_col.=="b"]),
            "r" => [sum(y[A_col.=="r"] .== l) for l in y_classes] ./ length(y[A_col.=="r"]),
            ),
        :D => Dict(
            false => [sum(y[D_col.==false] .== l) for l in y_classes] ./ length(y[D_col.==false]),
            true => [sum(y[D_col.==true] .== l) for l in y_classes] ./ length(y[D_col.==true]),
            ),
        :C => Dict(
            "f" => [sum(y[C_col.=="f"] .== l) for l in y_classes] ./ length(y[C_col.=="f"]),
            "m" => [sum(y[C_col.=="m"] .== l) for l in y_classes] ./ length(y[C_col.=="m"]),
            ),
    )
    @test results[1] == true_output

    # Text mixing in the edge case
    P̂ = [length(y[y .== l])/length(y) for l in y_classes]
    y_stat_given_feat_level =
        target_encoder_fit(X, y; ignore = true, ordered_factor = false, m = Inf)[:y_stat_given_feat_level]

    true_output = Dict{Symbol, Dict{Any, AbstractVector{AbstractFloat}}}(
        :F => Dict("m" => P̂, "l" => P̂, "s" => P̂,),
        :A => Dict("g" => P̂, "b" => P̂, "r" => P̂,),
        :D => Dict(false => P̂, true => P̂,),
        :C => Dict("f" => P̂, "m" => P̂,),
    )
    @test y_stat_given_feat_level == true_output
end


@testset "Test binary classification target encoding transforms" begin
    X, y = classification_forms[1]
    cache =
        target_encoder_fit(X, y; ignore = true, ordered_factor = false)
    X_tr = target_encoder_transform(X, cache)

    enc = (col, level) -> cache[:y_stat_given_feat_level][col][level]

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

@testset "Test preserving output type" begin
    X, y = classification_forms[4]
    cache =
        target_encoder_fit(X, y; ignore = true, ordered_factor = false)
    X_tr = target_encoder_transform(X, cache)
    
    @test typeof(X_tr) == typeof(X)
end



@testset "Test regression target encoding transforms" begin
    X, y = regression_forms[1]
    cache =
        target_encoder_fit(X, y)
    X_tr = target_encoder_transform(X, cache)

    enc = (col, level) -> cache[:y_stat_given_feat_level][col][level]

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

@testset "Test multiclassification target encoding transforms" begin
    X, y = multiclassification_forms[1]
    cache =
        target_encoder_fit(X, y)
    X_tr = target_encoder_transform(X, cache)
    
    enc = (col, level) -> cache[:y_stat_given_feat_level][col][level]
    
    target = (
        A_1 = [enc(:A, X[:A][i])[1] for i in 1:10],
        A_2 = [enc(:A, X[:A][i])[2] for i in 1:10],
        A_3 = [enc(:A, X[:A][i])[3] for i in 1:10],
        B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        C_1 = [enc(:C, X[:C][i])[1] for i in 1:10],
        C_2 = [enc(:C, X[:C][i])[2] for i in 1:10],
        C_3 = [enc(:C, X[:C][i])[3] for i in 1:10],
        D_1 = [enc(:D, X[:D][i])[1] for i in 1:10],
        D_2 = [enc(:D, X[:D][i])[2] for i in 1:10],
        D_3 = [enc(:D, X[:D][i])[3] for i in 1:10],
        E = [1, 2, 3, 4, 5, 6, 6, 3, 2, 1],
        F_1 = [enc(:F, X[:F][i])[1] for i in 1:10],
        F_2 = [enc(:F, X[:F][i])[2] for i in 1:10],
        F_3 = [enc(:F, X[:F][i])[3] for i in 1:10]
    )
    for col in keys(target)
        @test all(X_tr[col] .== target[col])
    end
end


@testset "mlj-interface" begin
    X1, y1 = classification_forms[1]
    X2, y2 = regression_forms[1]
    X3, y3 = multiclassification_forms[1]
    Xys = vcat(classification_forms, regression_forms, multiclassification_forms)
    for (X, y) in Xys
        # functional api
        fit_res = target_encoder_fit(X, y; ignore=true, ordered_factor=false, lambda=0.5, m=1)
        X_transf = target_encoder_transform(X, fit_res)
        # mlj api
        encoder = TargetEncoder( ignore=true, ordered_factor=false, lambda=0.5, m=1.0)
        mach = machine(encoder, X, y)
        fit!(mach)
        Xnew_transf = MMI.transform(mach, X)

        # same output
        @test X_transf == Xnew_transf

        # fitted parameters is correct
        fitresult = fitted_params(mach)
        @test fitresult.y_statistic_given_feat_level == fit_res[:y_stat_given_feat_level]
        @test fitresult.task == fit_res[:task]

        # Test invalid `m`
        @test_throws ArgumentError begin
            t = TargetEncoder(ignore=true, ordered_factor=false, lambda=0.5, m=-5)
        end

        # Test invalid `lambda`
        @test_throws ArgumentError begin
             t = TargetEncoder(ignore=true, ordered_factor=false, lambda=1.1, m=1)
        end

        # Test report
        @test report(mach) == Dict(:encoded_features => fit_res[:encoded_features])
    end
end