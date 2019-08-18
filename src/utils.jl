# Functions that shift the indexes of the points in an inverted index
function _shift_up_inverse_index!(inverse_index::InvertedIndex{I,U}, by::I=one(I)) where{I,U}
    @inbounds @simd for i = eachindex(inverse_index)
        inverse_index[i].idxs .+= by
    end
end


function _shift_down_inverse_index!(inverse_index::InvertedIndex{I,U}, by::I=one(I)) where{I,U}
    @inbounds @simd for i = eachindex(inverse_index)
        inverse_index[i].idxs .-= by
    end
end


function _shift_inverse_index!(inverse_index::InvertedIndex{I,U}, point::I, by::I=one(I)) where{I,U}
    @inbounds @simd for i = eachindex(inverse_index)
        inverse_index[i].idxs[inverse_index[i].idxs .> point] .-= by
    end
end


"""
    pop!(ivfadc)

Pops from the index `ivfadc` the point with the highest index and returns it updating
the index as well.
"""
pop!(ivfadc) = _pop!(ivfadc, :last)


"""
    popfirst!(ivfadc)

Pops from the index `ivfadc` the first point and returns it updating the index as well.
"""
popfirst!(ivfadc) = _pop!(ivfadc, :first)


# Utility function for poping
function _pop!(ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q},
               position::Symbol=:last
              ) where{U,I,Dc,Dr,T,Q}
    @assert length(ivfadc) > 0 "Cannot pop element from empty index"
    cluster = 0  # cluster with max index
    idx = 0      # index in inverted list
    (vecid, shift) = I.(ifelse(position==:last, (length(ivfadc)-1, 0), (0, 1)))
    local point_codes
    @inbounds for i = eachindex(ivfadc.inverse_index)
        if vecid in ivfadc.inverse_index[i].idxs
            cluster = i
            idx = findfirst(isequal(vecid), ivfadc.inverse_index[i].idxs)
            point_codes = ivfadc.inverse_index[i].codes[idx]
        end
    end

    # Get point
    reconstructed = _get_quantizer_vector(ivfadc.coarse_quantizer, cluster) +
                        _decode_point(ivfadc.residual_quantizer, point_codes)

    # Delete index data
    deleteat!(ivfadc.inverse_index[cluster].idxs, idx)
    deleteat!(ivfadc.inverse_index[cluster].codes, idx)

    # Shift index
    _shift_down_inverse_index!(ivfadc.inverse_index, shift)
    return reconstructed
end


function _decode_point(quantizer::QuantizedArrays.OrthogonalQuantizer{U,Dr,T,2},
                       codes::Vector{U}
                      ) where{U,Dr,T}
    m, n = length(quantizer.codebooks), quantizer.dims[1]
    residual = Vector{T}(undef, n)
    @inbounds for i in 1:m
        rr = QuantizedArrays.rowrange(n, m, i)
        residual[rr] .= quantizer.codebooks[i][codes[i]]
    end
    return residual
end


"""
    delete_from_index!(ivfadc, points)

Deletes the points with indices contained in `points` from
the index `ivfadc`.
"""
function delete_from_index!(ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q},
                            points::Vector{<:Integer}
                           ) where{U,I,Dc,Dr,T,Q}
    shifted_points = I.(points .- 1)  # shift points
    for point in sort(unique(shifted_points), rev=true)
        @inbounds for i = eachindex(ivfadc.inverse_index)
            if point in ivfadc.inverse_index[i].idxs
                pidx = findfirst(isequal(point), ivfadc.inverse_index[i].idxs)
                deleteat!(ivfadc.inverse_index[i].idxs, pidx)
                deleteat!(ivfadc.inverse_index[i].codes, pidx)
                _shift_inverse_index!(ivfadc.inverse_index, point)
                break
            end
        end
    end
end


"""
    push!(ivfadc, point)

Pushes `point` to the end of index `ivfadc`; the point is assigned to a cluster
and its quantized code added to the inverted list corresponding to the cluster.
"""
push!(ivfadc, point) = _push!(ivfadc, point, :last)


"""
    pushfirst!(ivfadc, point)

Pushes `point` to the beginning of index `ivfadc`; the point is assigned to a cluster
and its quantized code added to the inverted list corresponding to the cluster.
"""
pushfirst!(ivfadc, point) = _push!(ivfadc, point, :first)


# Utility function for pushing
function _push!(ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q},
                point::Vector{T},
                position::Symbol
               ) where{U,I,Dc,Dr,T,Q}
    # Checks and initializations
    nrows, nvectors = size(ivfadc)
    @assert nrows == length(point) "Adding to index requires $nrows-element vectors"
    @assert QuantizedArrays.TYPE_TO_BITS[I] >=
        log2(nvectors+1) "Cannot index, exceeding index capacity of $(Int(typemax(I)+1)) points"

    qpoint, mincluster = _encode_point(ivfadc, point)

    # Insert in the inverted list corresponding to the cluster
    (vecid, shift) = ifelse(position == :first, (0, one(I)), (nvectors, zero(I)))
    _shift_up_inverse_index!(ivfadc.inverse_index, shift)
    push!(ivfadc.inverse_index[mincluster].idxs, vecid)
    push!(ivfadc.inverse_index[mincluster].codes, qpoint)
    return nothing
end


function _encode_point(ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q},
                       point::Vector{T}
                      ) where{U,I,Dc,Dr,T,Q}
    nrows, nvectors = size(ivfadc)

    # Find belonging cluster (first [1] for (idxs,dist), second for idxs array)
    mincluster = coarse_search(ivfadc.coarse_quantizer, point, 1)[1][1]

    # Quantize residual
    residual = point - _get_quantizer_vector(ivfadc.coarse_quantizer, mincluster)
    quantized_point = vec(QuantizedArrays.quantize_data(ivfadc.residual_quantizer,
                            reshape(residual, nrows, 1)))
    return quantized_point, mincluster
end
