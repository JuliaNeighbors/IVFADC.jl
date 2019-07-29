struct InvertedList{U}
    idxs::Vector{Int}
    codes::Vector{Vector{U}}
end


const InvertedIndex{U} = Dict{Int, InvertedList{U}}


struct CoarseQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}
    vectors::Matrix{T}
    distance::D
end


struct IVFADCIndex{U<:Unsigned,
                   D1<:Distances.PreMetric,
                   D2<:Distances.PreMetric,
                   T<:AbstractFloat}
    coarse_quantizer::CoarseQuantizer{D1,T}
    residual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,D2,T,2}
    inverse_index::InvertedIndex{U}
end


Base.length(ivfadc::IVFADCIndex) =
    mapreduce(ivlist->length(ivlist.codes), +, values(ivfadc.inverse_index))

Base.size(ivfadc::IVFADCIndex) = (size(ivfadc.coarse_quantizer.vectors, 1),
                                  length(ivfadc))

Base.show(io::IO, ivfadc::IVFADCIndex{U,D1,D2,T}) where {U,D1,D2,T} = begin
    nvars, nvectors = size(ivfadc)
    nc = size(ivfadc.coarse_quantizer.vectors, 2)
    print(io, "IVFADC Index, $nc coarse vectors, total of $nvectors-element $T vectors, $U codes")
end


function build_index(data::Matrix{T};
                     kc::Int=DEFAULT_COARSE_K,
                     k::Int=DEFAULT_QUANTIZATION_K,
                     m::Int=DEFAULT_QUANTIZATION_M,
                     coarse_distance::Distances.PreMetric=DEFAULT_COARSE_DISTANCE,
                     quantization_distance::Distances.PreMetric=DEFAULT_QUANTIZATION_DISTANCE,
                     quantization_method::Symbol=DEFAULT_QUANTIZATION_METHOD,
                     coarse_maxiter::Int=DEFAULT_COARSE_MAXITER,
                     quantization_maxiter::Int=DEFAULT_QUANTIZATION_MAXITER
                    ) where {T<:AbstractFloat}
    # Checks
    # TODO

    # Run kmeans, build coarse quantizer
    @debug "clustering..."
    cmodel = kmeans(data,
                    kc,
                    maxiter=coarse_maxiter,
                    distance=coarse_distance,
                    init=:kmpp,
                    display=:none)

    # Calculate residuals
    @debug "residual calculation..."
    residuals = _build_residuals(cmodel, data)

    # Build residuals quantizer
    @debug "building quantizer..."
    rq = build_quantizer(residuals,
                         k=k,
                         m=m,
                         method=quantization_method,
                         distance=quantization_distance,
                         maxiter=quantization_maxiter)

    # Quantize residuals for each cluster
    @debug "building inverted index..."
    ii = _build_inverted_index(rq, cmodel, residuals)

    # Return
    @debug "building coarse quantizer and returning"
    cq = CoarseQuantizer(cmodel.centers, coarse_distance)
    return IVFADCIndex(cq, rq, ii)
end


function _build_residuals(km::KmeansResult, data::Matrix{T}) where {T}
    residuals = similar(data)
    for cluster in 1:nclusters(km)
        bidxs = km.assignments .== cluster
        @. residuals[:, bidxs] = data[:, bidxs] - km.centers[:, cluster]
    end
    return residuals
end


function _build_inverted_index(rq::QuantizedArrays.OrthogonalQuantizer{U,D,T,2},
                               km::KmeansResult,
                               data::Matrix{T}) where {U,D,T}
    invindex = InvertedIndex{U}()
    for cluster in 1:nclusters(km)
        idxs = findall(x->isequal(x, cluster), km.assignments)
        qdata = QuantizedArrays.quantize_data(rq, data[:, idxs])
        ivlist = InvertedList(idxs, [qdata[:, j] for j in 1:length(idxs)])
        push!(invindex, cluster => ivlist)
    end
    return invindex
end


function add_to_index!(ivfadc::IVFADCIndex{U,D1,D2,T},
                       point::Vector{T}
                      ) where{U,D1,D2,T}
    # Checks and initializations
    # TODO(Corneliu): Checks
    cq_distance = ivfadc.coarse_quantizer.distance
    cq_clcenters = ivfadc.coarse_quantizer.vectors

    # Find belonging cluster
    coarse_distances = colwise(cq_distance, cq_clcenters, point)
    _, mincluster = findmin(coarse_distances)

    # Quantize residual
    residual = ivfadc.coarse_quantizer.vectors[:, mincluster] - point
    qv = vec(QuantizedArrays.quantize_data(
                ivfadc.residual_quantizer,
                reshape(residual, length(residual),1)
               )
            )

    # Insert in the inverted list corresponding to the cluster
    newidx = length(ivfadc) + 1
    push!(ivfadc.inverse_index[mincluster].idxs, newidx)
    push!(ivfadc.inverse_index[mincluster].codes, qv)
    return nothing
end


function delete_from_index!(ivfadc::IVFADCIndex{U,D1,D2,T},
                            points::Vector{Int}
                           ) where{U,D1,D2,T}
    for point in sort(unique(points), rev=true)
        for (cl, ivlist) in ivfadc.inverse_index
            if point in ivlist.idxs
                pidx = findfirst(x->x==point, ivlist.idxs)
                deleteat!(ivlist.idxs, pidx)
                deleteat!(ivlist.codes, pidx)
                _shift_inverse_index!(ivfadc.inverse_index, point)
                break
            end
        end
    end
end


function _shift_inverse_index!(inverse_index::InvertedIndex{U},
                               point::Int
                              ) where{U}
    for (cl, ivlist) in inverse_index
        ivlist.idxs[ivlist.idxs .> point] .-= 1
    end
end


function knn_search(ivfadc::IVFADCIndex{U,D1,D2,T},
                    point::Vector{T},
                    k::Int;
                    w::Int=1
                   ) where {U,D1,D2,T}
    # Checks and initializations
    @assert k >= 1 "Number of neighbors must be k >= 1"
    @assert w >= 1 "Number of clusters to search in must be w >= 1"

    cq_distance = ivfadc.coarse_quantizer.distance
    cq_clcenters = ivfadc.coarse_quantizer.vectors
    nclusters = size(cq_clcenters, 2)
    w = min(w, nclusters)

    # Find the 'w' closest coarse vectors
    coarse_distances = colwise(cq_distance, cq_clcenters, point)
    closest_clusters = sortperm(coarse_distances)[1:w]

    # Calculate all residual distances
    # (between the vector and the codebooks of the residual quantizer)
    rq_cbooks = ivfadc.residual_quantizer.codebooks
    m, n = length(rq_cbooks), length(point)
    difftables = Vector{Dict{U,T}}(undef, m)
    for i in 1:m
        rr = QuantizedArrays.rowrange(n, m, i)
        diffs = colwise(cq_distance, rq_cbooks[i].vectors, point[rr])
        difftables[i] = Dict{U,T}(zip(rq_cbooks[i].codes, diffs))
    end

    # Calculate distances between point and the vectors
    # from the clusters using the residual distances.
    ids = Vector{Int}()
    distances = Vector{T}()
    neighbors = SortedMultiDict{T,Int}()
    maxdist = zero(T)
    @inbounds for cl in closest_clusters
        d = coarse_distances[cl]
        ivlist = ivfadc.inverse_index[cl]
        for (id, code) in zip(ivlist.idxs, ivlist.codes)
            for (i, code_el) in enumerate(code)
                d += difftables[i][code_el]
            end
            if length(neighbors) < k
                push!(neighbors, d=>id)
                maxdist, _ = last(neighbors)
            elseif maxdist > d
                delete!((neighbors, lastindex(neighbors)))  # delete the max key
                push!(neighbors, d=>id)
                maxdist, _ = last(neighbors)
            end
        end
    end
    return collect(values(neighbors)), collect(keys(neighbors))
end
