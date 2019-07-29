# IVFADC.jl - inverted file system with asymmetric distance computation for
#             billion-scale approximate nearest enighbor search
#             written at 0x0α Research by Corneliu Cofaru, 2019

module IVFADC

using DataStructures
using Distances
using Clustering
using QuantizedArrays

export IVFADCIndex,
       build_index,
       add_to_index!,
       delete_from_index!,
       knn_search

include("core.jl")
include("defaults.jl")

end # module
