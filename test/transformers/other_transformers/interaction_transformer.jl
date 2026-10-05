@testset "Interaction Transformer" begin
    # Check constructor sanity checks: order > 1, length(features) > 1
    @test_logs (:warn, "Constraint `model.order > 1` failed; using default: order=2.") InteractionTransformer(order = 1)
    @test_logs (:warn, "Constraint `if model.features !== nothing\n"*
                       "    length(model.features) > 1\nelse\n    true\nend` failed; "*
                       "using default: features=nothing.") InteractionTransformer(features = [:A])

    X = (A = [1, 2, 3], B = [4, 5, 6], C = [7, 8, 9])
    # Default order=2, features=nothing, ie all columns
    Xt = MLJBase.transform(InteractionTransformer(), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54]
    )
    # order=3, features=nothing, ie all columns
    Xt = MLJBase.transform(InteractionTransformer(order=3), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
        A_B_C = [28, 80, 162]
    )
    # order=2, features=[:A, :B], ie all columns
    Xt =MLJBase.transform(InteractionTransformer(order=2, features=[:A, :B]), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18]
    )
    # order=3, features=[:A, :B, :C], some non continuous columns
    X = merge(X, (D = ["x₁", "x₂", "x₃"],))
    Xt = MLJBase.transform(InteractionTransformer(order=3, features=[:A, :B, :C]), nothing, X)
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
    # order=2, features=nothing, only continuous columns are dealt with
    Xt = MLJBase.transform(InteractionTransformer(order=2), nothing, X)
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


