using MLJTransforms: contrast_encoder_fit, contrast_encoder_transform, get_dummy_contrast, get_sum_contrast, 
create_backward_vector, get_backward_diff_contrast, get_forward_diff_contrast, create_helmert_vector, get_helmert_contrast, ContrastEncoder

stable_rng = StableRNGs.StableRNG(123)
X =     (name   = categorical(["Ben", "John", "Mary", "John"]),
height = [1.85, 1.67, 1.5, 1.67],
favnum = categorical([7, 5, 10, 1]),
age    = [23, 23, 14, 23])


@testset "Contrast Encoder Error Handling" begin

    # Example definitions to allow the test to run
    function dummy_buildmatrix(colname, k)
        # Simple dummy function to generate a matrix of correct size
        return randn(k, k-1)  # Adjust dimensions as needed for the test
    end

    # Define a DataFrame or appropriate data structure to test with
    data = DataFrame(
        A = categorical(["level1", "level2", "level3"]),
        B = categorical(["levelX", "levelY", "levelZ"]),
    )

    # Test IGNORE_MUST_FALSE_VEC_MODE error
    @test_throws ArgumentError contrast_encoder_fit(data, [:A], mode=[:contrast], ignore=true)

    # Test LENGTH_MISMATCH_VEC_MODE error
    @test_throws ArgumentError contrast_encoder_fit(data, [:A], mode=[:contrast, :dummy], buildmatrix=dummy_buildmatrix, ignore=false)

    # Test BUILDFUNC_MUST_BE_SPECIFIED error
    @test_throws ArgumentError contrast_encoder_fit(data, [:A], mode=:contrast, ignore=false)

    # Test MATRIX_SIZE_ERROR
    wrong_buildmatrix = (levels, k) -> randn(k, k)  # Incorrect dimensions
    @test_throws ArgumentError contrast_encoder_fit(data, [:A], mode=:contrast, buildmatrix=wrong_buildmatrix, ignore=false)

    # Test MATRIX_SIZE_ERROR_HYP
    wrong_buildmatrix_hyp = (levels, k) -> randn(k, k+1)  # Incorrect dimensions for hypothesis matrix
    @test_throws ArgumentError contrast_encoder_fit(data, [:A], mode=:hypothesis, buildmatrix=wrong_buildmatrix_hyp, ignore=false)
end

@testset "Dummy Coding Tests" begin
    for k in 2:5  # Testing for various numbers of levels
        contrast_matrix = get_dummy_contrast(k)
        expected_matrix = Matrix(1.0I, k, k-1)
        @test contrast_matrix == expected_matrix
        @test size(contrast_matrix) == (k, k-1)
    end
    # test that fit is correct for dummy Coding
    cache = contrast_encoder_fit(X, [:name]; ignore=false, mode = :dummy)
    k = length(levels(X.name))
    contrast_matrix = get_dummy_contrast(k)
    print()
    for (i, level) in enumerate(levels(X.name))
        println(cache[:vec_given_feat_val])
        @test cache[:vec_given_feat_val][:name][level] == contrast_matrix[i, :]
    end
end


@testset "Sum Coding Tests" begin
    # Manually define the expected matrix for a 4-level categorical variable
    expected_matrix_4 = [1.0  0.0  0.0;
                         0.0  1.0  0.0;
                         0.0  0.0  1.0;
                        -1.0 -1.0 -1.0]  # Sum of each column for the first three rows is zeroed by the last row
    contrast_matrix_4 = get_sum_contrast(4)
    @test contrast_matrix_4 == expected_matrix_4
    @test size(contrast_matrix_4) == (4, 3)

    # Additional tests can be included for different levels, with each matrix defined manually
    # Example for 3 levels
    expected_matrix_3 = [1.0  0.0;
                         0.0  1.0;
                        -1.0 -1.0]
    contrast_matrix_3 = get_sum_contrast(3)
    @test contrast_matrix_3 == expected_matrix_3
    @test size(contrast_matrix_3) == (3, 2)
    # test that fit is correct for sum Coding
    cache = contrast_encoder_fit(X, [:name, :favnum]; ignore=false, mode = :sum)
    k = length(levels(X.favnum))
    contrast_matrix = get_sum_contrast(k)
    for (i, level) in enumerate(levels(X.favnum))
        @test cache[:vec_given_feat_val][:favnum][level] == contrast_matrix[i, :]
    end
end

@testset "Backward Difference Coding Tests" begin
    # Manually define the expected matrix for a 4 level categorical variable
    expected_matrix_4 = [-0.75  -0.5  -0.25;
                          0.25  -0.5  -0.25;
                          0.25   0.5  -0.25;
                          0.25   0.5   0.75]
    contrast_matrix_4 = get_backward_diff_contrast(4)
    @test contrast_matrix_4 == expected_matrix_4
    @test size(contrast_matrix_4) == (4, 3)

    # Test that fit is correct for backward Coding
    cache = contrast_encoder_fit(X, [:name, :favnum]; ignore=false, mode = :backward_diff)
    k = length(levels(X.favnum))
    contrast_matrix = get_backward_diff_contrast(k)
    for (i, level) in enumerate(levels(X.favnum))
        @test cache[:vec_given_feat_val][:favnum][level] == contrast_matrix[i, :]
    end
end

@testset "Forward Difference Coding Tests" begin
    for k in 2:5
        backward_matrix = get_backward_diff_contrast(k)
        forward_matrix = get_forward_diff_contrast(k)
        @test forward_matrix == -backward_matrix
        @test size(forward_matrix) == (k, k-1)
    end

    # Test that fit is correct for forward Coding
    cache = contrast_encoder_fit(X, [:name, :favnum]; ignore=false, mode = :forward_diff)
    k = length(levels(X.favnum))
    contrast_matrix = get_forward_diff_contrast(k)
    for (i, level) in enumerate(levels(X.favnum))
        @test cache[:vec_given_feat_val][:favnum][level] == contrast_matrix[i, :]
    end
end

@testset "helmert_vector function tests" begin
    @test create_helmert_vector(1, 5) == [-1.0, 1.0, 0.0, 0.0, 0.0]
    @test create_helmert_vector(2, 5) == [-1.0, -1.0, 2.0, 0.0, 0.0]
    @test create_helmert_vector(3, 5) == [-1.0, -1.0, -1.0, 3.0, 0.0]
    @test create_helmert_vector(4, 5) == [-1.0, -1.0, -1.0, -1.0, 4.0]
    @test create_helmert_vector(1, 3) == [-1.0, 1.0, 0.0]
    @test create_helmert_vector(2, 3) == [-1.0, -1.0, 2.0]
    k = 4
    @test get_helmert_contrast(k) ==  [
    -1.0  -1.0  -1.0
     1.0  -1.0  -1.0
     0.0   2.0  -1.0
     0.0   0.0   3.0]
     # test that fit is correct for helmert Coding
     cache = contrast_encoder_fit(X, [:name, :favnum]; ignore=false, mode = :helmert)
     k = length(levels(X.name))
     contrast_matrix = get_helmert_contrast(k)
     for (i, level) in enumerate(levels(X.name))
         @test cache[:vec_given_feat_val][:name][level] == contrast_matrix[i, :]
     end
end


# @testset "contrast matrix end-to-end test"
@testset "contrast mode end-to-end test" begin


    function buildrandomcontrast(colname, k)
        return rand(StableRNGs.StableRNG(123), k, k-1)
    end

    cache = contrast_encoder_fit(X; mode=:contrast, buildmatrix=buildrandomcontrast)

    X_tr = contrast_encoder_transform(X, cache)
    X_tr_mlj = Tables.matrix(X_tr)[:,1:end-1]


    df = DataFrame(X)

    mf = ModelFrame(@formula(age ~ (name + height + favnum)), df, contrasts = Dict(
        :name => StatsModels.ContrastsCoding(buildrandomcontrast(nothing, 3)),
        :favnum=> StatsModels.ContrastsCoding(buildrandomcontrast(nothing, 4))
        ))

    X_tr_sm =  ModelMatrix(mf).m[:, 2:end]

    @test X_tr_mlj == X_tr_sm
end

# @testset "hypothesis matrix end-to-end test"
@testset "hypothesis mode end-to-end test" begin

    function buildrandomhypothesis(colname, k)
        return rand(StableRNGs.StableRNG(123), k-1, k)
    end    

    cache = contrast_encoder_fit(X; mode=:hypothesis, buildmatrix=buildrandomhypothesis)
    X_tr = contrast_encoder_transform(X, cache)
    X_tr_mlj = Tables.matrix(X_tr)[:,1:end-1]
    df = DataFrame(X)

    mf = ModelFrame(@formula(age ~ (name + height + favnum)), df, contrasts = Dict(
        :name => HypothesisCoding(buildrandomhypothesis(nothing, 3); levels=levels(X.name), labels=[]),
        :favnum=> HypothesisCoding(buildrandomhypothesis(nothing, 4); levels=levels(X.favnum), labels=[])
        ))

    X_tr_sm =  ModelMatrix(mf).m[:, 2:end]

    @test X_tr_mlj == X_tr_sm
end


function buildrandomhypothesis(colname, k)
    return rand(StableRNGs.StableRNG(123), k-1, k)
end    

function buildrandomcontrast(colname, k)
    return rand(StableRNGs.StableRNG(123), k, k-1)
end

@testset "single-mode end-to-end test with StatsModels" begin
    # test end-to-end single_column transformations
    for ind in 1:6
        stats_models(k, ind) = [
            StatsModels.ContrastsCoding(buildrandomcontrast(nothing, k)),
            DummyCoding(; base=(k == 3) ? "Mary" : 10),
            EffectsCoding(; base=(k == 3) ? "Mary" : 10),
            SeqDiffCoding(),
            HelmertCoding(),
            HypothesisCoding(buildrandomhypothesis(nothing, k); levels=(k == 3) ? levels(X.name) : levels(X.favnum), labels=[]),
        ][ind]
        modes = [:contrast, :dummy, :sum, :backward_diff, :helmert, :hypothesis]
        matrix_func = [buildrandomcontrast, nothing, nothing, nothing, nothing, buildrandomhypothesis]

        # Try MLJTransforms
        cache = contrast_encoder_fit(X; mode=modes[ind], buildmatrix=matrix_func[ind])
        X_tr = contrast_encoder_transform(X, cache)

        df = DataFrame(X)

        mf = ModelFrame(@formula(age ~ (name + height + favnum)), df, contrasts = Dict(
            :name => stats_models(3, ind),
            :favnum=> stats_models(4, ind),
            ))

        X_tr_mlj = Tables.matrix(X_tr)[:,1:end-1]
        X_tr_sm =  ModelMatrix(mf).m[:, 2:end]
        @test X_tr_mlj ≈  X_tr_sm
    end
end

@testset "multi-mode end-to-end test with StatsModels" begin
    # test end-to-end single_column transformations
    for ind1 in 1:6
        for ind2 in 2:5
            stats_models(k, ind) = [
                StatsModels.ContrastsCoding(buildrandomcontrast(nothing, k)),
                DummyCoding(; base=(k == 3) ? "Mary" : 10),
                EffectsCoding(; base=(k == 3) ? "Mary" : 10),
                SeqDiffCoding(),
                HelmertCoding(),
                HypothesisCoding(buildrandomhypothesis(nothing, k); levels=(k == 3) ? levels(X.name) : levels(X.favnum), labels=[]),
            ][ind]

            modes = [:contrast, :dummy, :sum, :backward_diff, :helmert, :hypothesis]
            matrix_func = [buildrandomcontrast, nothing, nothing, nothing, nothing, buildrandomhypothesis]

            # Try MLJTransforms
            cache = contrast_encoder_fit(X, [:name, :favnum]; ignore=false, mode=[modes[ind1], modes[ind2]], buildmatrix=matrix_func[ind1])
            X_tr = contrast_encoder_transform(X, cache)

            df = DataFrame(X)

            mf = ModelFrame(@formula(age ~ (name + height + favnum)), df, contrasts = Dict(
                :name => stats_models(3, ind1),
                :favnum=> stats_models(4, ind2),
                ))

            X_tr_mlj = Tables.matrix(X_tr)[:,1:end-1]
            X_tr_sm =  ModelMatrix(mf).m[:, 2:end]

            @test X_tr_mlj ≈  X_tr_sm
        end
    end
end



@testset "MLJ Interface Contrast Encoding" begin
    # functional api
    generic_cache = contrast_encoder_fit(X; ignore = true, ordered_factor = false)
    X_transf = contrast_encoder_transform(X, generic_cache)
    # mlj api
    encoder = ContrastEncoder(ignore = true, ordered_factor = false)
    mach = machine(encoder, X)
    fit!(mach)
    Xnew_transf = MMI.transform(mach, X)

    # same output
    @test X_transf == Xnew_transf

    # fitted parameters is correct
    vec_given_feat_val = fitted_params(mach).vec_given_feat_val
    @test vec_given_feat_val == generic_cache[:vec_given_feat_val]

    # Test report
    @test report(mach) == (encoded_features = generic_cache[:encoded_features],)
end