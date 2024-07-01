using MLJTransforms: ordinal_encoder_fit, ordinal_encoder_transform

# Possible dataset forms (X,y)
dataset_forms = []
push!(dataset_forms, create_dummy_dataset(:regression, as_dataframe=false, return_y=false))
push!(dataset_forms, create_dummy_dataset(:regression, as_dataframe=true, return_y=false))

@testset "Ordinal Encoding Fit" begin
    @test ordinal_encoder_fit(dataset_forms[1]) == ordinal_encoder_fit(dataset_forms[2])
    X = dataset_forms[1]
    result = ordinal_encoder_fit(X)[:index_given_feat_level]
    A_col, C_col, D_col, F_col = MMI.selectcols(X, [1, 3, 4, 6])
    true_output = Dict{Symbol, Dict{Any, AbstractFloat}}(
        :F => Dict(
            "m" => findfirst(==("m"), levels(F_col)),
            "l" => findfirst(==("l"), levels(F_col)),
            "s" => findfirst(==("s"), levels(F_col)),
        ),
        :A => Dict(
            "g" => findfirst(==("g"), levels(A_col)),
            "b" => findfirst(==("b"), levels(A_col)),
            "r" => findfirst(==("r"), levels(A_col)),
            ),
        :D => Dict(
            false => findfirst(==(false), levels(D_col)),
            true => findfirst(==(true), levels(D_col)),
            ),
        :C => Dict(
            "f" => findfirst(==("f"), levels(C_col)),
            "m" => findfirst(==("m"), levels(C_col)),
            ),
    )
    @test result == true_output	
end

# Redundant because it must work if fit works (only relies on generic_transform)
@testset "Ordinal Encoding Transform" begin
    X = dataset_forms[1]
    cache = ordinal_encoder_fit(X)

    X_tr = ordinal_encoder_transform(X, cache)
    
    enc = (col, level) -> findfirst(==(level), levels(X[col]))
    
    target = (
        A = [enc(:A, X[:A][i]) for i in 1:10],
        B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        C = [enc(:C, X[:C][i]) for i in 1:10],
        D = [enc(:D, X[:D][i]) for i in 1:10],
        E = [1, 2, 3, 4, 5, 6, 6, 3, 2, 1],
        F = [enc(:F, X[:F][i]) for i in 1:10]
    )
    @test X_tr == target
end


@testset "MLJ Interface Ordinal Encoding" begin
    Xs = [dataset_forms[1], dataset_forms[2]]
    for X in Xs
        # functional api
        fit_res = ordinal_encoder_fit(X; ignore=true, ordered_factor=false)
        X_transf = ordinal_encoder_transform(X, fit_res)
        # mlj api
        encoder = OrdinalEncoder( ignore=true, ordered_factor=false)
        mach = machine(encoder, X)
        fit!(mach)
        Xnew_transf = MMI.transform(mach, X)

        # same output
        @test X_transf == Xnew_transf

        # fitted parameters is correct
        fitresult = fitted_params(mach)
        @test fitresult.index_given_feat_level == fit_res[:index_given_feat_level]

        # Test report
        @test report(mach) == Dict(:encoded_features => fit_res[:encoded_features])
    end
end