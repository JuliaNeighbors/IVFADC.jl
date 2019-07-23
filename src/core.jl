struct InvertedList{U}
    idxs::Vector{Int}
    codes::Vector{Vector{U}}
end


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
    inverse_index::Dict{Int, InvertedList{U}}
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
    invindex = Dict{Int, InvertedList{U}}()
    for cluster in 1:nclusters(km)
        idxs = findall(x->isequal(x, cluster), km.assignments)
        qdata = QuantizedArrays.quantize_data(rq, data[:, idxs])
        ivlist = InvertedList(idxs, [qdata[:, j] for j in 1:length(idxs)])
        push!(invindex, cluster => ivlist)
    end
    return invindex
end


function add_to_index!(ivfadc::IVFADCIndex{U,D1,D2,T}, point::Vector{T}
                      ) where{U,D1,D2,T}
    # Checks
    # TODO(Corneliu)

    # Find belonging cluster
    mpoint = reshape(point, length(point), 1)
    dists = vec(pairwise(ivfadc.coarse_quantizer.distance,
                         ivfadc.coarse_quantizer.vectors,
                         mpoint,
                         dims=2))
    _, cluster = findmin(dists)

    # Quantize residual
    residual = ivfadc.coarse_quantizer.vectors[:, cluster] - mpoint
    qv = vec(QuantizedArrays.quantize_data(ivfadc.residual_quantizer, residual))

    # Insert in the inverted list corresponding to the cluster
    newidx = length(ivfadc) + 1
    push!(ivfadc.inverse_index[cluster].idxs, newidx)
    push!(ivfadc.inverse_index[cluster].codes, qv)
    return nothing
end


function knn_search(ivfadc::IVFADCIndex{U,D1,D2,T}, query::Vector{T}, k::Int
                   ) where {U,D1,D2,T}

    # TODO(Corneliu) Implement this

end
