
@testset "UnivariateFillImputer" begin
    vpure = rand(stable_rng, 10)
    v = vcat([missing, ], vpure)
    filler = median(vpure)
    imp = MLJTransforms.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = [missing, 1.0, missing, 2.0, 3.0]
    @test MLJBase.transform(imp, f, vnew) ≈ [filler, 1.0, filler, 2.0, 3.0]

    vpure = MLJBase.coerce(rand(stable_rng, "abc", 100), OrderedFactor);
    v = vcat([missing, ], vpure)
    filler = mode(vpure)
    imp = MLJTransforms.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = vcat([missing, ], vpure[end-10:end], [missing, ])
    @test MLJBase.transform(imp, f, vnew) ==
        vcat([filler, ], vpure[end-10:end], [filler, ])

    vpure = rand(stable_rng, Int, 10)
    v = vcat([missing, ], vpure)
    filler = round(Int, median(vpure))
    imp = MLJTransforms.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = [missing, 1, missing, 2, 3]
    @test MLJBase.transform(imp, f, vnew) == [filler, 1, filler, 2, 3]

    @test_throws Exception MLJBase.transform(imp, f, [missing, "1", "2"])

    @test_throws ArgumentError MLJBase.fit(imp, 1, [missing, "1", "2"])

end


@testset "FillImputer" begin
    X = (
        x = [missing,ones(10)...],
        y = [missing,ones(10)...],
        z = [missing,ones(10)...]
        )

    imp = FillImputer()
    f,  = MLJBase.fit(imp, 1, X)

    fp = MLJBase.fitted_params(imp, f)
    @test fp.features_seen_in_fit == [:x, :y, :z]
    @test fp.univariate_transformer == MLJTransforms.UnivariateFillImputer()
    @test fp.filler_given_feature[:x] ≈ 1.0
    @test fp.filler_given_feature[:x] ≈ 1.0
    @test fp.filler_given_feature[:x] ≈ 1.0

    Xnew = MLJBase.selectrows(X, 1:5)
    Xt  = MLJBase.transform(imp, f, Xnew)
    @test all(.!ismissing.(Xt.x))
    @test Xt.x isa Vector{Float64} # no missing
    @test all(Xt.x .== 1.0)

    imp = FillImputer(features=[:x,:y])
    f,  = MLJBase.fit(imp, 1, X)
    Xt = MLJBase.transform(imp, f, Xnew)
    @test all(Xt.x .== 1.0)
    @test all(Xt.y .== 1.0)
    @test ismissing(Xt.z[1])

    # adding a new feature not seen in fit:
    Xnew = (x = X.x, y=X.y, a=X.x)
    @test_throws ArgumentError  MLJBase.transform(imp, f, Xnew)

    # mixture of features:
    X = (x = categorical([missing, missing, missing, missing,
                          "Old", "Young", "Middle", "Young",
                          "Old", "Young", "Middle", "Young"]),
         y = [missing, ones(11)...],
         z = [missing, missing, 1,1,1,1,1,5,1,1,1,1],
         a = rand("abc", 12))

    imp = FillImputer()
    f, = MLJBase.fit(imp, 1, X)
    Xnew = MLJBase.selectrows(X, 1:4)
    Xt = MLJBase.transform(imp, f, Xnew)

    @test all(.!ismissing.(Xt.x))
    @test all(.!ismissing.(Xt.y))
    @test all(.!ismissing.(Xt.z))
    @test all(.!ismissing.(Xt.a))

    @test Xt.x[1] == mode(skipmissing(X.x))
    @test Xt.y[1] == 1
    @test Xt.z[1] == 1

    # user specifies a feature explicitly that's not supported:
    imp = FillImputer(features=[:x, :a]) # :a of Unknown scitype not supported
    @test_logs (:info, r"Feature a will not") MLJBase.fit(imp, 1, X)

end
