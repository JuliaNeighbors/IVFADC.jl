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


@testset "IVFADC: search" begin
    @test true
    # Search single vector
    ### K = 3  # number of neighbors
    ### query = rand(nrows)
    ### idxs, dists = knn_search(ivfadc, query, K)

    ### # Search multiple vectors
    ### queries = [rand(nrows) for _ in 1:10]
    ### idxs, dists = knn_search(ivfadc, queries, K)
end
