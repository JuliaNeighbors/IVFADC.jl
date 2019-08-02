const nvectors = 243
const nrows = 10


function build_index_random_data()
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
    ivfadc = build_index(data,
                         kc=kc, k=k, m=m,
                         coarse_distance=coarse_distance,
                         quantization_distance=quantization_distance,
                         quantization_method=quantization_method,
                         coarse_maxiter=coarse_maxiter,
                         quantization_maxiter=quantization_maxiter)
    return ivfadc
end


@testset "IVFADC: build_index" begin
    ivfadc =  build_index_random_data()
    @test ivfadc isa IVFADCIndex
end


@testset "IVFADC: add_to_index!" begin
    ivfadc = build_index_random_data()
    ol = length(ivfadc)
    nnv = 15
    for i in 1:nnv
        add_to_index!(ivfadc, rand(nrows))
    end
    @test length(ivfadc) == ol + nnv
end


@testset "IVFADC: knn_search" begin
    ivfadc = build_index_random_data()
    # Search single vector
    K = 3  # number of neighbors
    query = rand(nrows)
    idxs, dists = knn_search(ivfadc, query, K, w=2)
    @test idxs isa Vector{Int}
    @test dists isa Vector{eltype(query)}

    # Search multiple vectors
    #queries = [rand(nrows) for _ in 1:10]
    #idxs, dists = knn_search(ivfadc, queries, K)
end


@testset "IVFADC: delete_from_index!" begin
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
    @test length(ivfadc) == n-length(idxs_to_delete)

    # Test that indexes are correctly updated
    for (cl, ivlist) in enumerate(ivfadc_copy.inverse_index)
        cluster_indexes = ivlist.idxs
        cluster_indexes_del = ivfadc.inverse_index[cl].idxs  # after deletion
        found = intersect(cluster_indexes, idxs_to_delete)
        @test length(cluster_indexes) == length(cluster_indexes_del) + length(found)
        for (i, idx) in enumerate(cluster_indexes)
            if idx > L1e && idx < L2s
                shift = L1e - L1s + 1
            elseif idx > L2e && idx < L3s
                shift = (L1e - L1s + 1) + (L2e - L2s + 1)
            else
                shift = nothing
            end

            if shift != nothing
                oldpos = i
                newval = idx - shift
                newpos = findfirst(x->x==newval, cluster_indexes_del)
                @test ivfadc_copy.inverse_index[cl].codes[oldpos] ==
                    ivfadc.inverse_index[cl].codes[newpos]
            end
        end
    end
end
