@testset "Persistency (naive coarse quantizer): save/load_ivfadc_index" begin
    ivfadc = build_index_random_data(coarse_quantizer=:naive)
    filepath, io = mktemp()

    # write to disk
    try
        save_ivfadc_index(filepath, ivfadc)

        # load from disk and compare
        ivfadc2 = load_ivfadc_index(filepath)
        @test typeof(ivfadc) == typeof(ivfadc2)

        # coarse quantizer
        @test ivfadc.coarse_quantizer.vectors == ivfadc2.coarse_quantizer.vectors
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


@testset "Persistency (hnsw coarse quantizer)" begin
    ivfadc = build_index_random_data(coarse_quantizer=:hnsw)
    filepath, io = mktemp()
    # try to write to disk and test for failure
    try
        save_ivfadc_index(filepath, ivfadc)

        # load from disk and compare
        ivfadc2 = load_ivfadc_index(filepath)
        @test typeof(ivfadc) == typeof(ivfadc2)

        # brevity assignments
        hnsw_orig = ivfadc.coarse_quantizer.hnsw
        hnsw_loaded = ivfadc2.coarse_quantizer.hnsw
        # coarse quantizer
        @test hnsw_orig.lgraph.linklist == hnsw_loaded.lgraph.linklist
        @test hnsw_orig.lgraph.M == hnsw_loaded.lgraph.M
        @test hnsw_orig.lgraph.M0 == hnsw_loaded.lgraph.M0
        @test hnsw_orig.lgraph.m_L == hnsw_loaded.lgraph.m_L
        @test hnsw_orig.added == hnsw_loaded.added
        @test hnsw_orig.data == hnsw_loaded.data
        @test hnsw_orig.ep == hnsw_loaded.ep
        @test hnsw_orig.entry_level == hnsw_loaded.entry_level
        @test hnsw_orig.vlp.num_elements == hnsw_loaded.vlp.num_elements
        for i in 1:length(hnsw_orig.vlp.pool)
            @test hnsw_orig.vlp.pool[i].list == hnsw_loaded.vlp.pool[i].list
            @test hnsw_orig.vlp.pool[i].visited_value == hnsw_loaded.vlp.pool[i].visited_value
        end
        @test hnsw_orig.metric == hnsw_loaded.metric
        @test hnsw_orig.efConstruction == hnsw_loaded.efConstruction
        @test hnsw_orig.ef == hnsw_loaded.ef
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
