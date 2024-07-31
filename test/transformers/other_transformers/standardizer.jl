
@testset begin "standardization"

    # UnivariateStandardizer:
    stand = UnivariateStandardizer()
    f,    = MLJBase.fit(stand, 1, [0, 2, 4])
    @test round.(Int, MLJBase.transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
    @test round.(Int, MLJBase.inverse_transform(stand, f, [-1, 1, 3])) ==
        [0, 4, 8]
    # infos = MLJTransforms.info_dict(stand)

    N = 5
    rand_char = rand("abcefgh", N)
    while length(unique(rand_char)) < 2
        rand_char = rand("abcefgh", N)
    end
    X = (OverallQual  = rand(UInt8, N),
         GrLivArea    = rand(N),
         Neighborhood = categorical(rand_char, ordered=true),
         x1stFlrSF    = sample(1:10, N, replace=false),
         TotalBsmtSF  = rand(N))

    # introduce a field of type `Char`:
    x1 = categorical(map(Char, (X.OverallQual |> collect)))

    X = (x1=x1, x2=X[2], x3=X[3], x4=X[4], x5=X[5])

    stand = Standardizer()
    f,    = MLJBase.fit(stand, 1, X)
    Xnew  = MLJBase.transform(stand, f, X)

    # test inverse:
    XX = MLJBase.inverse_transform(stand, f, Xnew)
    @test MLJBase.schema(X) == MLJBase.schema(XX)
    @test XX.x1 == X.x1
    @test XX.x2 ≈ X.x2
    @test XX.x3 == X.x3
    @test XX.x4 == X.x4
    @test XX.x5 ≈ X.x5

    # test transformation:
    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test MLJBase.std(Xnew[5]) ≈ 1.0

    # test feature specification (ignore=false):
    stand.features = [:x1, :x5]
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[[5,]]))
    Xt = MLJBase.transform(stand, f, X)
    @test Xnew[1] == X[1]
    @test Xnew[2] == X[2]
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test MLJBase.std(Xnew[5]) ≈ 1.0

    # test on ignoring a feature, even if it's listed in the `features`
    stand.ignore = true
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[[2,]]))
    Xt = MLJBase.transform(stand, f, X)
    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test Xnew[5] == X[5]

    # test warnings about features not encountered in fit or no
    # features need transforming:
    stand = Standardizer(features=[:x1, :mickey_mouse])
    @test_logs(
        (:warn, r"Some specified"),
        (:warn, r"No features"),
        MLJBase.fit(stand, 1, X)
    )
    stand.ignore = true
    @test_logs (:warn, r"Some specified") MLJBase.fit(stand, 1, X)

    # features must be specified if ignore=true
    @test_throws ArgumentError Standardizer(ignore=true)

    # test count, ordered_factor options:
    stand = Standardizer(features=[:x3, :x4], count=true, ordered_factor=true)
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[3:4,]))
    Xt = MLJBase.transform(stand, f, X)
    @test_throws Exception MLJBase.inverse_transform(stand, f, Xt)

    @test Xnew[1] == X[1]
    @test Xnew[2] == X[2]
    @test elscitype(X[3]) <: OrderedFactor
    @test elscitype(Xnew[3]) <: Continuous
    @test MLJBase.std(Xnew[3]) ≈ 1.0
    @test elscitype(X[4]) == Count
    @test elscitype(Xnew[4]) <: Continuous
    @test MLJBase.std(Xnew[4]) ≈ 1.0
    @test Xnew[5] == X[5]

    stand = Standardizer(features= x-> x == (:x2))
    f,    = MLJBase.fit(stand, 1, X)
    Xnew  = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test Xnew[5] == X[5]

    # infos = MLJTransforms.info_dict(stand)

    # @test infos[:name] == "Standardizer"
    # @test infos[:input_scitype] ==
    #     Union{MLJBase.Table, AbstractVector{<:Continuous}}
    # @test infos[:output_scitype] ==
    #     Union{MLJBase.Table, AbstractVector{<:Continuous}}

    # univariate case
    stand = Standardizer()
    f, _, _   = MLJBase.fit(stand, 1, [0, 2, 4])
    @test round.(Int, MLJBase.transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
    fp = MLJBase.fitted_params(stand, f)
    @test fp.mean ≈ 2.0
    @test fp.std ≈ 2.0
end