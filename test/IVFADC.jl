@testset "IVFADC" begin
    nrows, nvectors = 20, 10_000
    data = rand(nrows, nvectors)
    kc = 100    # number of coarse quantizer vectors
    k = 256     # quantization levels (for residuals)
    m = 5       # number of quantizers (for residuals)
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

    @test ivfadc isa IVFADCIndex

    # Search single vector
    ### K = 3  # number of neighbors
    ### query = rand(nrows)
    ### idxs, dists = knn_search(ivfadc, query, K)

    ### # Search multiple vectors
    ### queries = [rand(nrows) for _ in 1:10]
    ### idxs, dists = knn_search(ivfadc, queries, K)
end
