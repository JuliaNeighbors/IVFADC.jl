var documenterSearchIndex = {"docs":
[{"location":"api/#","page":"API Reference","title":"API Reference","text":"","category":"page"},{"location":"api/#","page":"API Reference","title":"API Reference","text":"Modules = [IVFADC]","category":"page"},{"location":"api/#IVFADC.IVFADCIndex","page":"API Reference","title":"IVFADC.IVFADCIndex","text":"IVFADCIndex{U<:Unsigned, I<:Unsigned, Dc<:Distances.PreMetric, Dr<:Distances.PreMetric, T<:AbstractFloat}\n\nThe inverse file system object. It allows for approximate nearest neighbor search into the contained vectors.\n\nFields\n\ncoarse_quantizer::CoarseQuantizer{Dc,T} contains the coarse vectors\nresidual_quantizer::QuantizedArrays.OrthogonalQuantizer{U,Dr,T,2}\n\nis employed to quantize vectors when adding to the index\n\ninverse_index::InvertedIndex{I,U} is the actual inverse index employed\n\nto perform the search.\n\n\n\n\n\n","category":"type"},{"location":"api/#IVFADC.add_to_index!-Union{Tuple{T}, Tuple{Dr}, Tuple{Dc}, Tuple{I}, Tuple{U}, Tuple{IVFADCIndex{U,I,Dc,Dr,T},Array{T,1}}} where T where Dr where Dc where I where U","page":"API Reference","title":"IVFADC.add_to_index!","text":"add_to_index!(ivfadc, point)\n\nAdds point to the index ivfadc; the point is assigned to a cluster and its quantized code added to the inverted list corresponding to the cluster.\n\n\n\n\n\n","category":"method"},{"location":"api/#IVFADC.build_index-Union{Tuple{Array{T,2}}, Tuple{T}, Tuple{I}} where T<:AbstractFloat where I<:Unsigned","page":"API Reference","title":"IVFADC.build_index","text":"build_index(data [;kwargs])\n\nBuilds an inverse file system for billion-scale ANN search.\n\nArguments\n\nMatrix{T<:AbstractFloat} input data\n\nKeyword arguments\n\nkc::Int=DEFAULT_COARSE_K number of clusters (Voronoi cells) to employ\n\nin the coarse quantization step\n\nk::Int=DEFAULT_QUANTIZATION_K number of residual quantization levels to use\nm::Int=DEFAULT_QUANTIZATION_M number of residual quantizers to use\ncoarse_distance=DEFAULT_COARSE_DISTANCE coarse quantization distance\nquantization_distance=DEFAULT_QUANTIZATION_DISTANCE residual quantization distance\nquantization_method=DEFAULT_QUANTIZATION_METHOD residual quantization method\ncoarse_maxiter=DEFAULT_COARSE_MAXITER number of clustering iterations for obtaining\n\nthe coarse vectors\n\nquantization_maxiter=DEFAULT_QUANTIZATION_MAXITER number of clustering iterations for\n\nresidual quantization\n\nindex_type=UInt32 type for the indexes of the vectors in the inverted list\n\n\n\n\n\n","category":"method"},{"location":"api/#IVFADC.delete_from_index!-Union{Tuple{T}, Tuple{Dr}, Tuple{Dc}, Tuple{I}, Tuple{U}, Tuple{IVFADCIndex{U,I,Dc,Dr,T},Array{#s74,1} where #s74<:Integer}} where T where Dr where Dc where I where U","page":"API Reference","title":"IVFADC.delete_from_index!","text":"delete_from_index!(ivfadc, points)\n\nDeletes the points with indices contained in points from the index ivfadc.\n\n\n\n\n\n","category":"method"},{"location":"api/#IVFADC.knn_search-Union{Tuple{T}, Tuple{Dr}, Tuple{Dc}, Tuple{I}, Tuple{U}, Tuple{IVFADCIndex{U,I,Dc,Dr,T},Array{T,1},Int64}} where T where Dr where Dc where I where U","page":"API Reference","title":"IVFADC.knn_search","text":"knn_search(ivfadc, point, k[; w=1])\n\nSearches at most k closest neighbors of point in the index ivfadc; the neighbors will be searched for in the points contained in the closest w clusters.\n\n\n\n\n\n","category":"method"},{"location":"api/#IVFADC.CoarseQuantizer","page":"API Reference","title":"IVFADC.CoarseQuantizer","text":"CoarseQuantizer{D<:Distances.PreMetric, T<:AbstractFloat}\n\nCoarse quantization structure. The vector fields contains the coarse vectors while distance contains the distance that is used to calculate the distance from a point to the coarse vectors.\n\n\n\n\n\n","category":"type"},{"location":"api/#IVFADC.InvertedIndex","page":"API Reference","title":"IVFADC.InvertedIndex","text":"Simple alias for Vector{InvertedList{I<:Unsigned, U<:Unsigned}}.\n\n\n\n\n\n","category":"type"},{"location":"api/#IVFADC.InvertedList","page":"API Reference","title":"IVFADC.InvertedList","text":"InvertedList{I<:Unsigned, U<:Unsigned}\n\nBasic structure which corresponds to the points found within a Voronoi cell. The fields idxs contains the indices of the points while codes contains quantized vector data.\n\n\n\n\n\n","category":"type"},{"location":"api/#Base.length-Tuple{IVFADCIndex}","page":"API Reference","title":"Base.length","text":"length(ivfadc::IVFADCIndex)\n\nReturns the number of vectors indexed by ivfadc.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.size-Tuple{IVFADCIndex}","page":"API Reference","title":"Base.size","text":"size(ivfadc::IVFADCIndex)\n\nReturns a tuple with the dimensionality and number of the vectors indexed by ivfadc.\n\n\n\n\n\n","category":"method"},{"location":"examples/#Usage-examples-1","page":"Usage examples","title":"Usage examples","text":"","category":"section"},{"location":"examples/#Building-an-IVFADC-index-1","page":"Usage examples","title":"Building an IVFADC index","text":"","category":"section"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Building an index can be performed with build_index","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"using IVFADC\ndata = rand(10, 100);  # 100 points, 10 dimensions\n\nivfadc = build_index(data, kc=20, k=32, m=2, index_type=UInt8)","category":"page"},{"location":"examples/#Searching-the-index-1","page":"Usage examples","title":"Searching the index","text":"","category":"section"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Searching into the index is done with knn_search","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"point = data[:, 55];\nidxs, dists = knn_search(ivfadc, point, 5)","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Internally, the IVFADC index uses 0-based indexing; to retrieve the actual 1-based neighbor indexes that correspond to indexes in data, a simple transform has to be performed:","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"int_idxs = Int.(idxs) .+ 1","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Results may vary depending on how many clusters are being used to search into","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"knn_search(ivfadc, point, 5, w=10)  # search into 10 clusters","category":"page"},{"location":"examples/#Updating-the-index-1","page":"Usage examples","title":"Updating the index","text":"","category":"section"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Adding and removing points to and from the index is done with the add_to_index! and delete_from_index! functions","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"for i in 1:5\n    add_to_index!(ivfadc, rand(10))\nend\nivfadc","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"and","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"delete_from_index!(ivfadc, [1,2,3]);\nivfadc","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"respectively.","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"note: Note\nWhen adding a new point, its index will always be the number of already existing points. When deleting points, the indexes of all points are updated so that they are consecutive.","category":"page"},{"location":"examples/#Limits-and-advanced-usage-of-the-index-1","page":"Usage examples","title":"Limits and advanced usage of the index","text":"","category":"section"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"It is not possible to add more points than the maximum value of the indexing type","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"ivfadc = build_index(rand(2,256), kc=2, k=16, m=1, index_type=UInt8)","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"Adding a new point to an index that already has 256 points (which is the maximum for the UInt8) throws an error","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"add_to_index!(ivfadc, rand(2))","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"It is however possible to add the point after deleting another one","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"delete_from_index!(ivfadc, [1])\nadd_to_index!(ivfadc, rand(2))","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"In the example above, the index is being used as a FIFO buffer where the first point is removed and the last one appended to the buffer.","category":"page"},{"location":"examples/#","page":"Usage examples","title":"Usage examples","text":"note: Note\nDeleting from the index is a slow operation as the all indexes of the points contained need to be properly updated depending on the positions of the points that are being deleted.","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"CurrentModule=IVFADC","category":"page"},{"location":"#Introduction-1","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"IVFADC implements an inverted file system with asymmetric distance computation for fast approximate nearest neighbor search in large i.e. billion-scale, high dimensional datasets.","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"warning: Warning\nThis package is under heavy development and should not be used at this point in production systems.","category":"page"},{"location":"#Implemented-features-1","page":"Introduction","title":"Implemented features","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"building the index\nadding files to the index\nremoving files from the index\nsearching into the index","category":"page"},{"location":"#Installation-1","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"Installation can be performed from either inside or outside Julia.","category":"page"},{"location":"#Git-cloning-1","page":"Introduction","title":"Git cloning","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"$ git clone https://github.com/zgornel/IVFADC.jl","category":"page"},{"location":"#Julia-REPL-1","page":"Introduction","title":"Julia REPL","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"The package can be installed from inside Julia with:","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(\"IVFADC\")","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"or","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"Pkg.add(PackageSpec(url=\"https://github.com/zgornel/IVFADC.jl\", rev=\"master\"))","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"for the latest master branch.","category":"page"}]
}
