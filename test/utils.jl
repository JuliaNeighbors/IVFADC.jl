@testset "Utils: push!, pushfirst!" begin
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc = build_index_random_data(;coarse_quantizer=coarse_quantizer,
                                         index_type=UInt8)
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
end


@testset "Utils: pop!, popfirst!" begin
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc = build_index_random_data(;coarse_quantizer=coarse_quantizer,
                                         index_type=UInt8)
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
end


@testset "Utils: delete_from_index!" begin
    for coarse_quantizer in [:naive, :hnsw]
        ivfadc = build_index_random_data(coarse_quantizer=coarse_quantizer)
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
end
