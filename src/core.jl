"""
    InvertedList{U<:Unsigned}

Basic structure which corresponds to the points found within
a Voronoi cell. The fields `idxs` contains the indices of the
points while `codes` contains quantized vector data.
"""
struct InvertedList{U<:Unsigned}
    idxs::Vector{Int}
    codes::Vector{Vector{U}}
end


"""
Simple alias for `Dict{Int, InvertedList{U<:Unsigned}}`.
"""
const InvertedIndex{U} = Dict{Int, InvertedList{U}}


"""
    CoarseQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}

Coarse quantization structure. The `vector` fields contains the
coarse vectors while `distance` contains the distance that is
used to calculate the distance from a point to the coarse vectors.
"""
struct CoarseQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}
    vectors::Matrix{T}
    distance::D
end


"""
    IVFADCIndex{U<:Unsigned, D1<:Distances.PreMetric, D2<:Distances.PreMetric, T<:AbstractFloat}

The inverse file system object. It allows for approximate nearest
neighbor search into the contained vectors.

# Fields
  * `coarse_quantizer::CoarseQuantizer{D1,T}` contains the coarse vectors
  * `residual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,D2,T,2}`
is employed to quantize vectors when adding to the index
  * `inverse_index::InvertedIndex{U}` is the actual inverse index employed
to perform the search.
"""
struct IVFADCIndex{U<:Unsigned,
                   D1<:Distances.PreMetric,
                   D2<:Distances.PreMetric,
                   T<:AbstractFloat}
    coarse_quantizer::CoarseQuantizer{D1,T}
    residual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,D2,T,2}
    inverse_index::InvertedIndex{U}
end


"""
    length(ivfadc::IVFADCIndex)

Returns the number of vectors indexed by `ivfadc`.
"""
Base.length(ivfadc::IVFADCIndex) =
    mapreduce(ivlist->length(ivlist.codes), +, values(ivfadc.inverse_index))


"""
    size(ivfadc::IVFADCIndex)

Returns a tuple with the dimensionality and number of the vectors indexed by `ivfadc`.
"""
Base.size(ivfadc::IVFADCIndex) = (size(ivfadc.coarse_quantizer.vectors, 1), length(ivfadc))


Base.show(io::IO, ivfadc::IVFADCIndex{U,D1,D2,T}) where {U,D1,D2,T} = begin
    nvars, nvectors = size(ivfadc)
    nc = size(ivfadc.coarse_quantizer.vectors, 2)
    print(io, "IVFADC Index, $nc coarse vectors, total of $nvectors-element $T vectors, $U codes")
end


"""
    build_index(data [;kwargs])

Builds an inverse file system for billion-scale ANN search.

# Arguments
  * `Matrix{T<:AbstractFloat}` input data

# Keyword arguments
  * `kc::Int=DEFAULT_COARSE_K` number of clusters (Voronoi cells) to employ
in the coarse quantization step
  * `k::Int=DEFAULT_QUANTIZATION_K` number of residual quantization levels to use
  * `m::Int=DEFAULT_QUANTIZATION_M` number of residual quantizers to use
  * `coarse_distance=DEFAULT_COARSE_DISTANCE` coarse quantization distance
  * `quantization_distance=DEFAULT_QUANTIZATION_DISTANCE` residual quantization distance
  * `quantization_method=DEFAULT_QUANTIZATION_METHOD` residual quantization method
  * `coarse_maxiter=DEFAULT_COARSE_MAXITER` number of clustering iterations for obtaining
the coarse vectors
  * `quantization_maxiter=DEFAULT_QUANTIZATION_MAXITER` number of clustering iterations for
residual quantization
"""
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
    nrows, nvectors = size(data)
    @assert kc >= 2 "Number of coarse clusters has to be >= 2"
    @assert k <= nvectors "Number of quantization levels  has to be <= $nvectors"
    @assert m >= 1 && m <= nrows "Number of codebooks has to be between 1 and $nrows"
    @assert coarse_maxiter > 0 "Number of clustering iterations has to be > 0"
    @assert quantization_maxiter > 0 "Number of clustering iterations has to be > 0"

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
    ii = _build_inverted_index(rq, cmodel, residuals)

    # Return
    @debug "Finalizing..."
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


"""
    add_to_index!(ivfadc, point)

Adds `point` to the index `ivfadc`; the point is assigned to a cluster
and its quantized code added to the inverted list corresponding to the
cluster.
"""
function add_to_index!(ivfadc::IVFADCIndex{U,D1,D2,T},
                       point::Vector{T}
                      ) where{U,D1,D2,T}
    # Checks and initializations
    nrows, nvectors = size(ivfadc)
    @assert nrows == length(point) "Adding to index requires $nrows-element vectors"

    cq_distance = ivfadc.coarse_quantizer.distance
    cq_clcenters = ivfadc.coarse_quantizer.vectors

    # Find belonging cluster
    coarse_distances = colwise(cq_distance, cq_clcenters, point)
    _, mincluster = findmin(coarse_distances)

    # Quantize residual
    residual = point - ivfadc.coarse_quantizer.vectors[:, mincluster]
    qv = vec(QuantizedArrays.quantize_data(
                ivfadc.residual_quantizer,
                reshape(residual, nrows, 1)
               )
            )

    # Insert in the inverted list corresponding to the cluster
    newidx = nvectors + 1
    push!(ivfadc.inverse_index[mincluster].idxs, newidx)
    push!(ivfadc.inverse_index[mincluster].codes, qv)
    return nothing
end


"""
    delete_from_index!(ivfadc, points)

Deletes the points with indices contained in `points` from
the index `ivfadc`.
"""
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


"""
    knn_search(ivfadc, point, k[; w=1])

Searches at most `k` closest neighbors of `point` in the index `ivfadc`;
the neighbors will be searched for in the points contained in the closest
`w` clusters.
"""
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
    rq_cbooks = ivfadc.residual_quantizer.codebooks
    m, n = length(rq_cbooks), length(point)
    w = min(w, nclusters)

    # Find the 'w' closest coarse vectors and calculate residuals
    coarse_distances = colwise(cq_distance, cq_clcenters, point)
    closest_clusters = sortperm(coarse_distances)[1:w]
    residuals = point .- cq_clcenters[:, closest_clusters]

    # Calculate distances between point and the vectors
    # from the clusters using the residual distances.
    ids = Vector{Int}()
    distances = Vector{T}()
    neighbors = SortedMultiDict{T,Int}()
    maxdist = zero(T)
    difftables = Vector{Dict{U,T}}(undef, m)
    @inbounds for (j, cl) in enumerate(closest_clusters)
        dc = coarse_distances[cl]
        # Calculate all residual distances
        # (between the vector and the codebooks of the residual quantizer)
        for i in 1:m
            rr = QuantizedArrays.rowrange(n, m, i)
            diffs = colwise(cq_distance, rq_cbooks[i].vectors, residuals[rr, j])
            difftables[i] = Dict{U,T}(zip(rq_cbooks[i].codes, diffs))
        end
        ivlist = ivfadc.inverse_index[cl]
        for (id, code) in zip(ivlist.idxs, ivlist.codes)
            d = dc
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