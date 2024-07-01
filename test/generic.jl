using MLJTransforms: generate_new_feat_names

@testset "Generate New Column Names Function Tests" begin
    # Test 1: No initial conflicts
    @testset "No Initial Conflicts" begin
        existing_names = []
        names = generate_new_feat_names("feat", 3, existing_names)
        @test names == [Symbol("feat_1"), Symbol("feat_2"), Symbol("feat_3")]
    end

    # Test 2: Handle initial conflict by adding underscores
    @testset "Initial Conflict Resolution" begin
        existing_names = [Symbol("feat_1"), Symbol("feat_2"), Symbol("feat_3")]
        names = generate_new_feat_names("feat", 3, existing_names)
        @test names == [Symbol("feat__1"), Symbol("feat__2"), Symbol("feat__3")]
    end
end
