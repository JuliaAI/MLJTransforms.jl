
@testset "U-boxcox" begin
    # create skewed non-negative vector with a zero value:
    Random.seed!(1551)
    v = abs.(randn(1000))
    v = v .- minimum(v)

    t  = UnivariateBoxCoxTransformer(shift=true)
    f, = MLJBase.fit(t, 2, v)

    e = v - MLJBase.inverse_transform(t, f, MLJBase.transform(t, f, v))
    @test sum(abs, e) <= 5000*eps()

    # infos = MLJTransforms.info_dict(t)

    # @test infos[:name] == "UnivariateBoxCoxTransformer"
    # @test infos[:input_scitype] == AbstractVector{MLJBase.Continuous}
    # @test infos[:output_scitype] == AbstractVector{MLJBase.Continuous}
end