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


@testset "push!" begin
    ivfadc = build_index_random_data(; index_type=UInt8)
    ol = length(ivfadc)
    nnv = 256 - nvectors
    for i in 1:nnv
        push!(ivfadc, rand(nrows))
    end

    @test length(ivfadc) == ol + nnv
    @test_throws AssertionError push!(ivfadc, rand(nrows))  # index is full

    delete_from_index!(ivfadc, [1])
    @test_throws AssertionError push!(ivfadc, rand(nrows+1))  # wrong dimension
end


@testset "knn_search" begin
    INDEX_TYPE = UInt32
    ivfadc = build_index_random_data(;index_type=INDEX_TYPE)
    # Search single vector
    K = 3  # number of neighbors
    query = rand(nrows)
    idxs, dists = knn_search(ivfadc, query, K, w=2)
    @test idxs isa Vector{INDEX_TYPE}
    @test dists isa Vector{eltype(query)}

    @test_throws AssertionError knn_search(ivfadc, query, 0)
    @test_throws AssertionError knn_search(ivfadc, query, 1, w=0)
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
