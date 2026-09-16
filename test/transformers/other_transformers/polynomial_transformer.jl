import MLJTransforms
import MLJBase
using Test
import DataFrames.DataFrame

@testset "helper functions" begin
    @test MLJTransforms.orderedwords((:x, :y, :z), 2) ==
        [[:x, :x], [:x, :y], [:x, :z], [:y, :y], [:y, :z], [:z, :z]]
    @test MLJTransforms.premonomials(
        (:x, :y, :z),
        3,
        MLJTransforms.WithoutRepetitions(),
    ) == [[:x, :y], [:x, :z], [:y, :z], [:x, :y, :z]]
    @test MLJTransforms.premonomials(
        (:x, :y, :z),
        3,
        MLJTransforms.WithRepetitions(),
    ) == [[:x, :x], [:x, :y], [:x, :z], [:y, :y], [:y, :z], [:z, :z], [:x, :x, :x],
          [:x, :x, :y], [:x, :x, :z], [:x, :y, :y], [:x, :y, :z], [:x, :z, :z],
          [:y, :y, :y], [:y, :y, :z], [:y, :z, :z], [:z, :z, :z]]
end


# # INTERACTIONS ONLY

@testset "interactions only" begin
    # Check constructor sanity checks:
    @test_logs(
        (:warn, MLJTransforms.WARN_DEGREE),
        PolynomialTransformer(interactions_only=true, degree = 0),
    )

    X = (A = [1, 2, 3], B = [4, 5, 6], C = [7, 8, 9])
    # Default degree=2, features=nothing, i.e., all columns
    Xt = MLJBase.transform(PolynomialTransformer(interactions_only=true), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54]
    )
    # degree=3, features=nothing, ie all columns
    Xt = MLJBase.transform(
        PolynomialTransformer(interactions_only=true,degree=3),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
        A_B_C = [28, 80, 162]
    )
    # degree=2, features=[:A, :B], ie all columns
    Xt =MLJBase.transform(
        PolynomialTransformer(interactions_only=true, degree=2, features=[:A, :B]),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18]
    )
    # degree=3, features=[:A, :B, :C], some non continuous columns
    X = merge(X, (D = ["x₁", "x₂", "x₃"],))
    Xt = MLJBase.transform(
        PolynomialTransformer(interactions_only=true, degree=3, features=[:A, :B, :C]),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        D = ["x₁", "x₂", "x₃"],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
        A_B_C = [28, 80, 162]
    )
    # degree=2, features=nothing, only continuous columns are dealt with
    Xt = MLJBase.transform(
        PolynomialTransformer(interactions_only=true, degree=2),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        D = ["x₁", "x₂", "x₃"],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
    )
end

@testset "all terms" begin
    # Check constructor sanity checks:
    @test_logs(
        (:warn, MLJTransforms.WARN_DEGREE),
        PolynomialTransformer(degree = 0),
    )

    X = (A = [1, 2, 3], B = [4, 5, 6])
    # Default degree=2, features=nothing, i.e., all columns
    Xt = MLJBase.transform(PolynomialTransformer(), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        A_A = [1, 4, 9],
        A_B = [4, 10, 18],
        B_B = [16, 25, 36],
    )

    # degree=3, features=nothing, ie all columns
    Xt = MLJBase.transform(
        PolynomialTransformer(degree=3),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        A_A = [1, 4, 9],
        A_B = [4, 10, 18],
        B_B = [16, 25, 36],
        A_A_A = [1, 8, 27],
        A_A_B = [4, 20, 54],
        A_B_B = [16, 50, 108],
        B_B_B = [64, 125, 216],
    )

    # degree=2, some non continuous columns
    X = merge(X, (D = ["x₁", "x₂", "x₃"],))
    Xt = MLJBase.transform(
        PolynomialTransformer(degree=2),
        nothing,
        X,
    )
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        D = ["x₁", "x₂", "x₃"],
        A_A = [1, 4, 9],
        A_B = [4, 10, 18],
        B_B = [16, 25, 36],
    )
end

@testset "non-native table types" begin
    X = (A = [1, 2, 3], B = [4, 5, 6], C = [7, 8, 9]) |> DataFrame
    Xt = MLJBase.transform(PolynomialTransformer(interactions_only=true), nothing, X)
    @test Xt == DataFrame((
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54]
    ))
end

true
