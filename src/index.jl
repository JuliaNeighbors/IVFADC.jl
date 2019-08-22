"""
    InvertedList{I<:Unsigned, U<:Unsigned}

Basic structure which corresponds to the points found within
a Voronoi cell. The fields `idxs` contains the indices of the
points while `codes` contains quantized vector data.
"""
struct InvertedList{I<:Unsigned, U<:Unsigned}
    idxs::Vector{I}
    codes::Vector{Vector{U}}
end


Base.show(io::IO, ivlist::InvertedList{I,U}) where {I,U} = begin
    n = length(ivlist.idxs)
    print(io, "InvertedList{$I,$U}, $n vectors")
end


"""
Simple alias for `Vector{InvertedList{I<:Unsigned, U<:Unsigned}}`.
"""
const InvertedIndex{I,U} = Vector{InvertedList{I,U}}


"""
    IVFADCIndex{U<:Unsigned, I<:Unsigned, Dc<:Distances.PreMetric, Dr<:Distances.PreMetric, T<:AbstractFloat, Q<:AbstractCoarseQuantizer{Dc,T}}

The inverse file system object. It allows for approximate nearest
neighbor search into the contained vectors.

# Fields
  * `coarse_quantizer::AbstractCoarseQuantizer{Dc,T}` contains the coarse vectors
  * `residual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,Dr,T,2}`
is employed to quantize vectors when adding to the index
  * `inverse_index::InvertedIndex{I,U}` is the actual inverse index employed
to perform the search.
"""
struct IVFADCIndex{U<:Unsigned,
                   I<:Unsigned,
                   Dc<:Distances.PreMetric,
                   Dr<:Distances.PreMetric,
                   T<:AbstractFloat,
                   Q<:AbstractCoarseQuantizer{Dc,T}}
    coarse_quantizer::Q
    residual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,Dr,T,2}
    inverse_index::InvertedIndex{I,U}
end


"""
    length(ivfadc::IVFADCIndex)

Returns the number of vectors indexed by `ivfadc`.
"""
Base.length(ivfadc::IVFADCIndex) =
    mapreduce(ivlist->length(ivlist.idxs), +, ivfadc.inverse_index)


"""
    size(ivfadc::IVFADCIndex)

Returns a tuple with the dimensionality and number of the vectors indexed by `ivfadc`.
"""
Base.size(ivfadc::IVFADCIndex) = (size(ivfadc.coarse_quantizer, 1), length(ivfadc))
Base.size(ivfadc::IVFADCIndex, i::Int) = size(ivfadc)[i]


Base.show(io::IO, ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q}) where {U,I,Dc,Dr,T,Q} = begin
    nvars, nvectors = size(ivfadc)
    idxsize = sizeof(I)
    m = length(ivfadc.residual_quantizer.codebooks)
    compsize = sizeof(U)
    codesize = m * compsize + idxsize
    cqstr = ifelse(Q <: HNSWQuantizer, "HNSW", "naive")
    print(io, "IVFADCIndex, $cqstr coarse quantizer, $codesize-byte encoding ($idxsize + $compsize×$m), $nvectors $T vectors")
end


"""
    IVFADCIndex(data [;kwargs])

Main constructor for building an inverse file system for billion-scale ANN search.

# Arguments
  * `Matrix{T<:AbstractFloat}` input data

# Keyword arguments
  * `kc::Int=DEFAULT_COARSE_K` number of clusters (Voronoi cells) to employ
in the coarse quantization step
  * `k::Int=DEFAULT_QUANTIZATION_K` number of residual quantization levels to use
  * `m::Int=DEFAULT_QUANTIZATION_M` number of residual quantizers to use
  * `coarse_quantizer::Symbol=DEFAULT_COARSE_QUANTIZER` coarse quantizer
  * `coarse_distance=DEFAULT_COARSE_DISTANCE` coarse quantization distance
  * `quantization_distance=DEFAULT_QUANTIZATION_DISTANCE` residual quantization distance
  * `quantization_method=DEFAULT_QUANTIZATION_METHOD` residual quantization method
  * `coarse_maxiter=DEFAULT_COARSE_MAXITER` number of clustering iterations for obtaining
the coarse vectors
  * `quantization_maxiter=DEFAULT_QUANTIZATION_MAXITER` number of clustering iterations for
residual quantization
  * `index_type=UInt32` type for the indexes of the vectors in the inverted list
"""
function IVFADCIndex(data::Matrix{T};
                     kc::Int=DEFAULT_COARSE_K,
                     k::Int=DEFAULT_QUANTIZATION_K,
                     m::Int=DEFAULT_QUANTIZATION_M,
                     coarse_quantizer::Symbol=DEFAULT_COARSE_QUANTIZER,
                     coarse_distance::Distances.PreMetric=DEFAULT_COARSE_DISTANCE,
                     quantization_distance::Distances.PreMetric=DEFAULT_QUANTIZATION_DISTANCE,
                     quantization_method::Symbol=DEFAULT_QUANTIZATION_METHOD,
                     coarse_maxiter::Int=DEFAULT_COARSE_MAXITER,
                     quantization_maxiter::Int=DEFAULT_QUANTIZATION_MAXITER,
                     index_type::Type{I}=UInt32
                    ) where {I<:Unsigned, T<:AbstractFloat}
    # Checks
    nrows, nvectors = size(data)
    bits_required = ceil(Int, log2(nvectors))
    @assert kc >= 2 "Number of coarse clusters has to be >= 2"
    @assert k <= nvectors "Number of quantization levels  has to be <= $nvectors"
    @assert m >= 1 && m <= nrows "Number of codebooks has to be between 1 and $nrows"
    @assert coarse_quantizer in [:naive, :hnsw] "Coarse quantizer can be :naive or :hnsw only"
    @assert coarse_maxiter > 0 "Number of clustering iterations has to be > 0"
    @assert quantization_maxiter > 0 "Number of clustering iterations has to be > 0"
    @assert QuantizedArrays.TYPE_TO_BITS[index_type] >=
        bits_required "$nvectors vectors require at least $bits_required index bits"

    # Run kmeans, build coarse quantizer
    @debug "Clustering..."
    cmodel = kmeans(data,
                    kc,
                    maxiter=coarse_maxiter,
                    distance=coarse_distance,
                    init=:kmpp,
                    display=:none)

    # Calculate residuals
    @debug "Residual calculation..."
    residuals = _build_residuals(cmodel, data)

    # Build residuals quantizer
    @debug "Building quantizer..."
    rq = build_quantizer(residuals,
                         k=k,
                         m=m,
                         method=quantization_method,
                         distance=quantization_distance,
                         maxiter=quantization_maxiter)

    # Quantize residuals for each cluster
    @debug "Building inverted index..."
    ii = _build_inverted_index(rq, cmodel, residuals, index_type=index_type)

    # Return
    @debug "Building coarse quantizer..."
    if coarse_quantizer == :naive
        cq = NaiveQuantizer{typeof(coarse_distance),
                eltype(cmodel.centers)}(cmodel.centers)
    else
        hnsw = HierarchicalNSW([cmodel.centers[:,i] for i in 1:kc],
                                metric=coarse_distance)
        add_to_graph!(hnsw)
        cq = HNSWQuantizer(hnsw)
    end
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
                               data::Matrix{T};
                               index_type::Type{I}=UInt32
                              ) where {U,I<:Unsigned,D,T}
    n = nclusters(km)
    invindex = InvertedIndex{I,U}(undef, n)
    for cluster in 1:n
        idxs = findall(isequal(cluster), km.assignments)
        qdata = QuantizedArrays.quantize_data(rq, data[:, idxs])
        ivlist = InvertedList{I,U}(
                    idxs .- one(I),
                    [qdata[:, j] for j in 1:length(idxs)])
        invindex[cluster] = ivlist
    end
    return invindex
end


"""
    knn_search(ivfadc, point, k[; w=1])

Searches at most `k` closest neighbors of `point` in the index `ivfadc`;
the neighbors will be searched for in the points contained in the closest
`w` clusters.
"""
function knn_search(ivfadc::IVFADCIndex{U,I,Dc,Dr,T},
                    point::Vector{T},
                    k::Int;
                    w::Int=1
                   ) where {U,I,Dc,Dr,T}
    # Checks and initializations
    @assert k >= 1 "Number of neighbors must be k >= 1"
    @assert w >= 1 "Number of clusters to search in must be w >= 1"
    cq = ivfadc.coarse_quantizer
    nclusters = size(cq, 2)
    rq_cbooks = ivfadc.residual_quantizer.codebooks
    m, n = length(rq_cbooks), length(point)
    w = min(w, nclusters)

    # Find the 'w' closest coarse vectors and calculate residuals
    closest_clusters, coarse_distances = coarse_search(cq, point, w)
    residuals = _closest_cluster_residuals(cq, point, closest_clusters)

    # Calculate distances between point and the vectors
    # from the clusters using the residual distances.
    distances = Vector{T}()
    neighbors = SortedMultiDict{T,I}()
    maxdist = zero(T)
    difftables = Vector{LittleDict{U,T}}(undef, m)
    @inbounds for j = eachindex(closest_clusters)
        dc = coarse_distances[j]
        # Calculate all residual distances
        # (between the vector and the codebooks of the residual quantizer)
        for i = eachindex(difftables)
            rr = QuantizedArrays.rowrange(n, m, i)
            diffs = colwise(Dc(), rq_cbooks[i].vectors, residuals[rr, j])
            difftables[i] = LittleDict{U,T}(rq_cbooks[i].codes, diffs)
        end
        # Loop through the iverted list and calculate
        # the actual (quantized) distances through lookup;
        # use SortedMultiDict as a maxheap for `k` neighbors
        ivlist = ivfadc.inverse_index[closest_clusters[j]]
        for i = eachindex(ivlist.idxs)
            d = dc
            @simd for ii = eachindex(ivlist.codes[i])
                # add difference vector for the ii'th code of the i'th subquantizer
                d += difftables[ii][ivlist.codes[i][ii]]
            end
            if length(neighbors) < k
                push!(neighbors, d => ivlist.idxs[i])
                maxdist, _ = last(neighbors)
            elseif maxdist > d
                delete!((neighbors, lastindex(neighbors)))  # delete the max key
                push!(neighbors, d => ivlist.idxs[i])
                maxdist, _ = last(neighbors)
            end
        end
    end
    return collect(values(neighbors)), collect(keys(neighbors))
end


function knn_search(ivfadc::IVFADCIndex{U,I,Dc,Dr,T},
                    points::Vector{Vector{T}},
                    k::Int;
                    w::Int=1
                   ) where {U,I,Dc,Dr,T}
    n = length(points)
    idxs = Vector{Vector{I}}(undef, n)
    dists = Vector{Vector{T}}(undef, n)
    for i = eachindex(points)  # Threads.@threads for i = ... for multi-threading
        idxs[i], dists[i] = knn_search(ivfadc, points[i], k, w=w)
    end
    return idxs, dists
end
