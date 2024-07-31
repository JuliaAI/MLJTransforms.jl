
@testset "One-Hot" begin
    X = (name   = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])

    t  = OneHotEncoder()
    f, _, report = @test_logs((:info, r"Spawning 3"),
                    (:info, r"Spawning 3"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test Xt.name__John == float.([false, true, false, true])
    @test Xt.height == X.height
    @test Xt.favourite_number__10 == float.([false, false, true, false])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                               :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

    @test report.new_features == collect(MLJBase.schema(Xt).names)

    # test that *entire* pool of categoricals is used in fit, including
    # unseen levels:
    f, = @test_logs((:info, r"Spawning 3"), (:info, r"Spawning 3"),
                          MLJBase.fit(t, 1, MLJBase.selectrows(X,1:2)))
    Xtsmall = MLJBase.transform(t, f, X)
    @test Xt == Xtsmall

    # test that transform can be applied to subset of the data:
    @test MLJBase.transform(t, f, MLJBase.selectcols(X, [:name, :age])) ==
        MLJBase.selectcols(MLJBase.transform(t, f, X),
                           [:name__Ben, :name__John, :name__Mary, :age])

    # test ignore
    t = OneHotEncoder(features=[:name,], ignore=true)
    f, = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test MLJBase.schema(Xt).names == (:name, :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

    # test exclusion of ordered factors:
    t  = OneHotEncoder(ordered_factor=false)
    f, = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test keys(Xt) == (:name, :height, :favourite_number__5,
                       :favourite_number__7, :favourite_number__10, :age)

    @test :name in Tables.schema(Xt).names
    @test :favourite_number__5 in Tables.schema(Xt).names
    @test MLJBase.schema(Xt).scitypes == (OrderedFactor{3}, Continuous,
                                          Continuous, Continuous,
                                          Continuous, Count)

    # test that one may not add new columns:
    X = (name = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height     = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age        = [23, 23, 14, 23],
         gender     = categorical(['M', 'M', 'F', 'M']))
    @test_throws Exception MLJBase.transform(t, f, X)

    # test to throw exception when category level mismatch is found
    X = (name   = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])
    Xmiss = (name   = categorical(["John", "Mary", "John"], ordered=true),
             height = X.height,
             favourite_number = X.favourite_number,
             age    = X.age)
    t  = OneHotEncoder()
    f, = MLJBase.fit(t, 0, X)
    @test_throws Exception MLJBase.transform(t, f, Xmiss)

#     infos = MLJTransforms.info_dict(t)

#     @test infos[:name] == "OneHotEncoder"
#     @test infos[:input_scitype] == MLJBase.Table
#     @test infos[:output_scitype] == MLJBase.Table

    # test the work on missing values
    X = (name   = categorical(["Ben", "John", "Mary", "John", missing], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67, 1.56],
         favourite_number = categorical([7, 5, 10, missing, 5]),
         age    = [23, 23, 14, 23, 21])

    t  = OneHotEncoder()
    f, _, report = @test_logs((:info, r"Spawning 3"),
                         (:info, r"Spawning 3"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test length(Xt.name__John) == 5
    @test collect(skipmissing(Xt.name__John)) == float.([false, true, false, true])
    @test ismissing(Xt.name__John[5])
    @test Xt.height == X.height
    @test length(Xt.favourite_number__10) == 5
    @test collect(skipmissing(Xt.favourite_number__10)) == float.([false, false, true, false])
    @test ismissing(Xt.favourite_number__10[4])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                               :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

    @test report.new_features == collect(MLJBase.schema(Xt).names)

    # test the work on missing values with drop_last = true

    X = (name   = categorical(["Ben", "John", "Mary", "John", missing], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67, 1.56],
         favourite_number = categorical([7, 5, 10, missing, 5]),
         age    = [23, 23, 14, 23, 21])

    t  = OneHotEncoder(drop_last = true)
    f, _, report = @test_logs((:info, r"Spawning 2"),
                        (:info, r"Spawning 2"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test length(Xt.name__John) == 5
    @test collect(skipmissing(Xt.name__John)) == float.([false, true, false, true])
    @test ismissing(Xt.name__John[5])
    @test Xt.height == X.height
    @test ismissing(Xt.favourite_number__5[4])
    @test collect(skipmissing(Xt.favourite_number__5)) == float.([false, true, false, true])
    @test ismissing(Xt.favourite_number__5[4])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John,
                            :height, :favourite_number__5,
                            :favourite_number__7,
                            :age)

    @test_throws Exception Xt.favourite_number__10
    @test_throws Exception Xt.name__Mary
    @test report.new_features == collect(MLJBase.schema(Xt).names)

    # Test when the first value is missing
    X = (name=categorical([missing, "John", "Mary", "John"]),)
    t  = OneHotEncoder()
    f, _, _ = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test Xt.name__John[1] === Xt.name__Mary[1] === missing
    @test Xt.name__John[2:end] == Union{Missing, Float64}[1.0, 0.0, 1.0]
    @test Xt.name__Mary[2:end] == Union{Missing, Float64}[0.0, 1.0, 0.0]

end
