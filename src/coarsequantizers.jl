"""
Abstract coarse quantizer type. The coarse quantizer is ment to
assign a point to a given number of Voronoi cells in which its
neighbors will be searched.
"""
abstract type AbstractCoarseQuantizer{D,T} end


# Naive coarse quantizer
"""
    NaiveQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}

Coarse quantization structure based on brute force search.
The `vector` fields contains the coarse vectors while `distance`
contains the distance that is used to calculate the distance
from a point to the coarse vectors.
"""
struct NaiveQuantizer{D<:Distances.PreMetric, T<:AbstractFloat} <: AbstractCoarseQuantizer{D,T}
    vectors::Matrix{T}
end


Base.show(io::IO, cq::NaiveQuantizer{D,T}) where {D,T} = begin
    nrows, nclusters = size(cq.vectors)
    print(io, "NaiveQuantizer{$D,$T}, $nrows×$nclusters cluster centres")
end


Base.size(cq::NaiveQuantizer{D,T}) where {D,T} = size(cq.vectors)
Base.size(cq::NaiveQuantizer{D,T}, i::Int) where {D,T} = size(cq.vectors, i)


function coarse_search(cq::NaiveQuantizer{D,T}, point::Vector{T}, w::Int) where {D,T}
    coarse_distances = colwise(D(), cq.vectors, point)
    closest_clusters = sortperm(coarse_distances)[1:w]
    return closest_clusters, coarse_distances[closest_clusters]
end


_closest_cluster_residuals(cq::NaiveQuantizer{D,T},
                           point::Vector{T},
                           closest_clusters::Vector{Int}
                          ) where {D,T} = begin
    return point .- cq.vectors[:, closest_clusters]
end


_get_quantizer_vector(cq::NaiveQuantizer, idx::Int) = cq.vectors[:, idx]


# HNSW coarse quantizer
"""
    HNSWQuantizer{U<:Unsigned, V<:Vector{Vector{T}}, D<:Distances.PreMetric, T<:AbstractFloat}

Coarse quantization structure based on HNSW search structure.
The `hnsw` field contains the coarse vectors.
"""
struct HNSWQuantizer{U, V, D<:Distances.PreMetric, T<:AbstractFloat} <: AbstractCoarseQuantizer{D,T}
    hnsw::HierarchicalNSW{U,T,V,D}
end


Base.show(io::IO, cq::HNSWQuantizer{U,V,D,T}) where {U,V,D,T} = begin
    nrows, nclusters = size(cq)
    print(io, "HNSWQuantizer{$U,$V,$D,$T}, $nrows×$nclusters cluster centres")
end


Base.size(cq::HNSWQuantizer{D,T}) where {D,T} = length(cq.hnsw.data[1]), length(cq.hnsw.data)
Base.size(cq::HNSWQuantizer{D,T}, i::Int) where {D,T} = size(cq)[i]


function coarse_search(cq::HNSWQuantizer{U,V,D,T}, point::Vector{T}, w::Int) where {U,V,D,T}
    closest_clusters, coarse_distances = HNSW.knn_search(cq.hnsw, point, w)
    return Int.(closest_clusters), coarse_distances
end


_closest_cluster_residuals(cq::HNSWQuantizer{U,V,D,T},
                           point::Vector{T},
                           closest_clusters::Vector{Int}
                          ) where {U,V,D,T} = begin
    nclusters = length(closest_clusters)
    residuals = Matrix{T}(undef, size(point,1), nclusters)
    @inbounds @simd for i in eachindex(closest_clusters)
        residuals[:, i] = point - cq.hnsw.data[closest_clusters[i]]
    end
    return residuals
end


_get_quantizer_vector(cq::HNSWQuantizer, idx::Int) = cq.hnsw.data[idx]
