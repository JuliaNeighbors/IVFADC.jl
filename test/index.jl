const nvectors = 243
const nrows = 10


function build_index_random_data(;coarse_quantizer=:naive,
                                 index_type=UInt32)
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
                         coarse_quantizer=coarse_quantizer,
                         coarse_distance=coarse_distance,
                         quantization_distance=quantization_distance,
                         quantization_method=quantization_method,
                         coarse_maxiter=coarse_maxiter,
                         quantization_maxiter=quantization_maxiter,
                         index_type=index_type)
    return ivfadc
end


@testset "Index: IVFADCIndex" begin
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc =  build_index_random_data(coarse_quantizer=coarse_quantizer)
        @test ivfadc isa IVFADCIndex

        data = rand(2, 300)
        @test_throws AssertionError IVFADCIndex(data, kc=1, k=2, m=1)    # kc fail
        @test_throws AssertionError IVFADCIndex(data, kc=2, k=301, m=1)  # k fail
        @test_throws AssertionError IVFADCIndex(data, kc=2, k=300, m=3)  # m fail
        @test_throws AssertionError IVFADCIndex(data, index_type=UInt8)  # index_type fail
    end
end
