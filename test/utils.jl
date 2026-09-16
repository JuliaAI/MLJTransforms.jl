@testset "actualfeatures" begin
    # No column provided, A has scitype Continuous, B has scitype Count
    table = (A = [1., 2., 3.], B = [4, 5, 6], C = ["x₁", "x₂", "x₃"])
    @test MLJTransforms.actualfeatures(nothing, table) == (:A, :B)
    # Column provided
    @test MLJTransforms.actualfeatures([:A, :B], table) == (:A, :B)
    # Column provided, not in table
    @test_throws(
        ArgumentError("Column(s) D are not in the dataset."),
        MLJTransforms.actualfeatures([:A, :D], table),
    )
    # Non Infinite scitype column provided
    @test_throws(
        ArgumentError("Column C's scitype is not Infinite."),
        MLJTransforms.actualfeatures([:A, :C], table),
    )
end

@testset "has_infinite_scitype" begin
    @test MLJTransforms.has_infinite_scitype([1.6, 1.7])
    @test MLJTransforms.has_infinite_scitype([42, missing])
    @test !MLJTransforms.has_infinite_scitype(["cat", "dog"])
    @test !MLJTransforms.has_infinite_scitype(["cat", missing])
end

true
