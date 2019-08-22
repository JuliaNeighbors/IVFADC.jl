@testset "Search: types, methods" begin
    INDEX_TYPE = UInt32
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc = build_index_random_data(;
                    coarse_quantizer=coarse_quantizer,
                    index_type=INDEX_TYPE)

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
end


@testset "Search: results" begin
    data = [0    0     0     1    1  1    1    1  20  20    20    20    20;
            0.1  0.11  0.12  8  10  15  14  16  5  5.1  5.2  5.4  5.5]
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc = IVFADCIndex(data, kc=3, k=8, m=2,
                             coarse_quantizer=coarse_quantizer)
        points = [[1.0,10.0], [0.0, 0.0], [20, 5.0]]

        neighbors_w1 = [[5, 4, 7, 6, 8],
                        [1, 2, 3],
                        [9, 10, 11, 12, 13]]
        for (point, result) in zip(points, neighbors_w1)
            neighbors = Int.(knn_search(ivfadc, point, 5, w=1)[1]) .+1
            @test isempty(setdiff(neighbors, result))
        end
        neighbors_w2 = [[5, 4, 7, 6, 8],
                        [1, 2, 3, 4, 5],
                        [9, 10, 11, 12, 13]]
        for (point, result) in zip(points, neighbors_w2)
            neighbors = Int.(knn_search(ivfadc, point, 5, w=2)[1]) .+1
            @test isempty(setdiff(neighbors, result))
        end
    end
end
