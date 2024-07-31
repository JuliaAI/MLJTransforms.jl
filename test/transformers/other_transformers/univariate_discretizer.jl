
@testset "U-Discr" begin
    v = randn(10000)
    t = UnivariateDiscretizer(n_classes=100);
    result, = MLJBase.fit(t, 1, v)
    w = MLJBase.transform(t, result, v)
    bad_values = filter(v - MLJBase.inverse_transform(t, result, w)) do x
        abs(x) > 0.05
    end
    @test length(bad_values)/length(v) < 0.06

    # scalars:
    @test MLJBase.transform(t, result, v[42]) == w[42]
    r =  MLJBase.inverse_transform(t, result, w)[43]
    @test MLJBase.inverse_transform(t, result, w[43]) ≈ r

    # test of permitted abuses of argument:
    @test MLJBase.inverse_transform(t, result, _get(w[43])) ≈ r
    @test MLJBase.inverse_transform(t, result, map(_get, w)) ≈
        MLJBase.inverse_transform(t, result, w)

    # all transformed vectors should have an identical pool (determined in
    # call to fit):
    v2 = v[1:3]
    w2 = MLJBase.transform(t, result, v2)
    @test levels(w2) == levels(w)

end