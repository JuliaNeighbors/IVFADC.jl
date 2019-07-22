struct InvertedList{U}
    idxs::Vector{Int}
    codes::Matrix{U}
end


struct CoarseQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}
    coarse_vectors::Matrix{T}
    coarse_distance::D
end


struct IVFADCIndex{U<:Unsigned,
                   D1<:Distances.PreMetric,
                   D2<:Distances.PreMetric,
                   T<:AbstractFloat}
    coarse_quantizer::CoarseQuantizer{D1,T}
    residue_quantizer::QuantizedArrays.OrthogonalQuantizer{U,D2,T,2}
    inverse_index::Dict{Int, InvertedList{U}}
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

    # Calculate residues
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
    invindex = Dict{Int, InvertedList{U}}()
    for cluster in 1:nclusters(km)
        idxs = findall(x->isequal(x, cluster), km.assignments)
        qdata = QuantizedArrays.quantize_data(rq, data[:, idxs])
        push!(invindex, cluster => InvertedList(idxs, qdata))
    end
    return invindex
end


function knn_search(ivfadc::IVFADCIndex{U,D1,D2,T}, query::Vector{T}, k::Int
                   ) where {U,D1,D2,T}

    # TODO(Corneliu) Implement this

end
