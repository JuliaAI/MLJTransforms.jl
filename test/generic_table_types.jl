using Test
using Tables
using CategoricalArrays



@testset "Generic Table Types Support" begin

    # Create test data as in the issue
    x = vcat(collect("abc"), fill('d', 100))
    x = coerce(x, Multiclass)
    
    # Column table (NamedTuple of vectors) - this already works
    coltable = (; x)
    
    # Row table (Vector of NamedTuples) - this was failing
    rowtable = Tables.rowtable(coltable)
    
    # List of models that were affected by the issue
    models_to_test = [
        CardinalityReducer(),
        FrequencyEncoder(),
        MissingnessEncoder(),
        OrdinalEncoder(),
    ]
    
    @testset "Model: $(string(typeof(model)))" for model in models_to_test
        
        @testset "Column Table Support" begin
            mach_col = machine(model, coltable)
            MLJBase.fit!(mach_col, verbosity=0)
            result_col = MLJBase.transform(mach_col, coltable)
            
            @test !isempty(Tables.columntable(result_col))
        end
        
        @testset "Row Table Support" begin
            # This should now work after the fix
            mach_row = machine(model, rowtable)
            MLJBase.fit!(mach_row, verbosity=0)
            result_row = MLJBase.transform(mach_row, rowtable)
            
            @test !isempty(Tables.columntable(result_row))
        end
        
        @testset "Consistency Between Table Types" begin
            # Results should be equivalent regardless of table type
            mach_col = machine(model, coltable)
            MLJBase.fit!(mach_col, verbosity=0)
            result_col = MLJBase.transform(mach_col, coltable)
            
            mach_row = machine(model, rowtable) 
            MLJBase.fit!(mach_row, verbosity=0)
            result_row = MLJBase.transform(mach_row, rowtable)
            
            # Convert both to column tables for comparison
            result_col_ct = Tables.columntable(result_col)
            result_row_ct = Tables.columntable(result_row)
            
            # Should have same column names
            @test keys(result_col_ct) == keys(result_row_ct)
            
            # Should have same values (allowing for potential ordering differences in table types)
            for col_name in keys(result_col_ct)
                @test Set(result_col_ct[col_name]) == Set(result_row_ct[col_name])
            end
        end
    end
end