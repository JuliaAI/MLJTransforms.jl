
@testset "Continuous encoder" begin

    X = (name  = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         rubbish = ["a", "b", "c", "a"],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])

    t  = ContinuousEncoder()
    f, _, _ = @test_logs((:info, r"Some.*dropped\:.*\:rubbish\]"),
                              MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)
    @test scitype(Xt) <: MLJBase.Table(MLJBase.Continuous)
    s = MLJBase.schema(Xt)
    @test s.names == (:name, :height, :favourite_number__5,
                      :favourite_number__7, :favourite_number__10, :age)

    t  = ContinuousEncoder(drop_last=true, one_hot_ordered_factors=true)
    f, _, r = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test scitype(Xt) <: MLJBase.Table(MLJBase.Continuous)
    s = MLJBase.schema(Xt)
    @test s.names == (:name__Ben, :name__John, :height, :favourite_number__5,
                      :favourite_number__7, :age)

end