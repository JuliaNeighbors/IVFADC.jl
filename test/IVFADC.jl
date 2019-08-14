const nvectors = 243
const nrows = 10


function build_index_random_data(;index_type=UInt32)
    data = rand(nrows, nvectors)
    kc = 100    # number of coarse quantizer vectors
    k = 16     # quantization levels (for residuals)
    m = 2       # number of quantizers (for residuals)
    coarse_distance = Distances.SqEuclidean()
    quantization_distance = Distances.SqEuclidean()
    quantization_method = :pq
    coarse_maxiter = 25
    quantization_maxiter = 25

    # Build index
    ivfadc = IVFADCIndex(data,
                         kc=kc, k=k, m=m,
                         coarse_distance=coarse_distance,
                         quantization_distance=quantization_distance,
                         quantization_method=quantization_method,
                         coarse_maxiter=coarse_maxiter,
                         quantization_maxiter=quantization_maxiter,
                         index_type=index_type)
    return ivfadc
end


@testset "IVFADCIndex" begin
    ivfadc =  build_index_random_data()
    @test ivfadc isa IVFADCIndex

    data = rand(2, 300)
    @test_throws AssertionError IVFADCIndex(data, kc=1, k=2, m=1)    # kc fail
    @test_throws AssertionError IVFADCIndex(data, kc=2, k=301, m=1)  # k fail
    @test_throws AssertionError IVFADCIndex(data, kc=2, k=300, m=3)  # m fail
    @test_throws AssertionError IVFADCIndex(data, index_type=UInt8)  # index_type fail

end


@testset "push! / pushfirst!" begin
    ivfadc = build_index_random_data(; index_type=UInt8)
    ol = length(ivfadc)
    nnv = 256 - nvectors

    # push!
    for i in 1:nnv
        push!(ivfadc, rand(nrows))
    end
    @test length(ivfadc) == ol + nnv
    @test_throws AssertionError push!(ivfadc, rand(nrows))  # index is full
    delete_from_index!(ivfadc, [1])
    @test_throws AssertionError push!(ivfadc, rand(nrows+1))  # wrong dimension

    # pushfirst!
    for i in 1:nnv-1
        delete_from_index!(ivfadc, [i])
    end
    for i in 1:nnv
        pushfirst!(ivfadc, rand(nrows))
    end
    @test length(ivfadc) == ol + nnv
    @test_throws AssertionError pushfirst!(ivfadc, rand(nrows))  # index is full
    delete_from_index!(ivfadc, [1])
    @test_throws AssertionError pushfirst!(ivfadc, rand(nrows+1))  # wrong dimension
end


@testset "pop! / popfirst!" begin
    ivfadc = build_index_random_data(; index_type=UInt8)
    ol = length(ivfadc)
    npops = 1
    # pop!
    for i in 1:npops
        v = pop!(ivfadc)
        @test v isa Vector{Float64}
        @test length(v) == size(ivfadc, 1)
    end
    @test length(ivfadc) == ol - npops

    # popfirst!
    ol = length(ivfadc)
    for i in 1:npops
        v=popfirst!(ivfadc)
        @test v isa Vector{Float64}
        @test length(v) == size(ivfadc, 1)
    end
    @test length(ivfadc) == ol - npops
end


@testset "knn_search" begin
    INDEX_TYPE = UInt32
    ivfadc = build_index_random_data(;index_type=INDEX_TYPE)

    # Single query
    K = 3  # number of neighbors
    query = rand(nrows)
    idxs, dists = knn_search(ivfadc, query, K, w=2)
    @test idxs isa Vector{INDEX_TYPE}
    @test dists isa Vector{eltype(query)}
    @test_throws AssertionError knn_search(ivfadc, query, 0)
    @test_throws AssertionError knn_search(ivfadc, query, 1, w=0)

    # Multiple queries
    queries = [rand(nrows) for _ in 1:10]
    idxs, dists = knn_search(ivfadc, queries, K, w=2)
    @test idxs isa Vector{Vector{INDEX_TYPE}}
    @test dists isa Vector{Vector{eltype(query)}}
end


@testset "delete_from_index!" begin
    ivfadc = build_index_random_data()
    ivfadc_copy = deepcopy(ivfadc)
    n = length(ivfadc)
    L1s, L1e = 1, 5
    L2s, L2e = 10, 30
    L3s, L3e = n-5, n
    first_list= collect(L1s : L1e)
    second_list = collect(L2s : L2e)
    third_list = collect(L3s : L3e)
    idxs_to_delete = vcat(first_list...,
                          second_list...,
                          third_list...)

    # Test that points are deleted
    delete_from_index!(ivfadc, idxs_to_delete)
    @test length(ivfadc) == n - length(idxs_to_delete)

    # Test that indexes are correctly updated
    mismatches = 0
    for (cl, ivlist) in enumerate(ivfadc_copy.inverse_index)
        cluster_indexes = ivlist.idxs
        cluster_indexes_del = ivfadc.inverse_index[cl].idxs  # after deletion
        found = intersect(cluster_indexes, idxs_to_delete .- 1)
        @test length(cluster_indexes) == length(cluster_indexes_del) + length(found)
        for (i, idx) in enumerate(cluster_indexes .+ 1)
            if idx > L1e && idx < L2s
                shift = L1e - L1s + 1
            elseif idx > L2e && idx < L3s
                shift = (L1e - L1s + 1) + (L2e - L2s + 1)
            else
                shift = nothing
            end

            if shift != nothing
                oldpos = i
                newval = idx - shift - 1
                newpos = findfirst(x->x==newval, cluster_indexes_del)
                if ivfadc_copy.inverse_index[cl].codes[oldpos] !=
                        ivfadc.inverse_index[cl].codes[newpos]
                    mismatches += 1
                end
            end
        end
    end
    @test mismatches == 0
end

@testset "persistency" begin
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
