#=
@testset "Persistency: save/load_ivfadc_index" begin
    ivfadc = build_index_random_data()
    filepath, io = mktemp()

    # write to disk
    try
        save_ivfadc_index(filepath, ivfadc)

        # load from disk and compare
        ivfadc2 = load_ivfadc_index(filepath)

        @test typeof(ivfadc) == typeof(ivfadc2)
        # coarse quantizer
        @test ivfadc.coarse_quantizer.vectors == ivfadc2.coarse_quantizer.vectors
        @test ivfadc.coarse_quantizer.distance == ivfadc2.coarse_quantizer.distance
        # residual quantizer
        @test ivfadc.residual_quantizer.k== ivfadc2.residual_quantizer.k
        @test ivfadc.residual_quantizer.dims == ivfadc2.residual_quantizer.dims
        @test ivfadc.residual_quantizer.distance== ivfadc2.residual_quantizer.distance
        @test ivfadc.residual_quantizer.quantization == ivfadc2.residual_quantizer.quantization
        @test ivfadc.residual_quantizer.rot == ivfadc2.residual_quantizer.rot
        for i in 1:length(ivfadc.residual_quantizer.codebooks)
            @test ivfadc.residual_quantizer.codebooks[i].codes == ivfadc2.residual_quantizer.codebooks[i].codes
            @test ivfadc.residual_quantizer.codebooks[i].vectors == ivfadc2.residual_quantizer.codebooks[i].vectors
        end
        #index
        for i in 1:length(ivfadc.inverse_index)
            @test ivfadc.inverse_index[i].idxs == ivfadc2.inverse_index[i].idxs
            @test ivfadc.inverse_index[i].codes == ivfadc2.inverse_index[i].codes
        end
        rm(filepath, force=true, recursive=true)
    catch e
        rm(filepath, force=true, recursive=true)
        @test e
    end
end
=#
