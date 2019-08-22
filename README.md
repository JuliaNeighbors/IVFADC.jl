![Alt text](https://github.com/zgornel/IVFADC.jl/blob/master/docs/src/assets/logo.png)

Inverted file system with asymmetric distance computation for billion-scale approximate nearest neighbor search.

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/zgornel/IVFADC.jl.svg?branch=master)](https://travis-ci.org/zgornel/IVFADC.jl)
[![Coverage Status](https://coveralls.io/repos/github/zgornel/IVFADC.jl/badge.svg?branch=master)](https://coveralls.io/github/zgornel/IVFADC.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://zgornel.github.io/IVFADC.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://zgornel.github.io/IVFADC.jl/dev)


## Installation
```julia
using Pkg
Pkg.add("IVFADC")
```
or
```julia
Pkg.add(PackageSpec(url="https://github.com/zgornel/IVFADC.jl", rev="master"))
```
for the latest `master` branch.


## Examples

### Create an index
```julia
using IVFADC
using Distances

nrows, nvectors = 50, 1_000
data = rand(Float32, nrows, nvectors)

kc = 100  # coarse vectors (i.e. Voronoi cells)
k = 256   # residual quantization levels/codebook
m = 10	  # residual quantizer codebooks

ivfadc = IVFADCIndex(data,
                     kc=kc,
                     k=k,
                     m=m,
                     coarse_quantizer=:naive,
                     coarse_distance=SqEuclidean(),
                     quantization_distance=SqEuclidean(),
                     quantization_method=:pq,
                     index_type=UInt16)
# IVFADCIndex, naive coarse quantizer, 12-byte encoding (2 + 1×10), 1000 Float32 vectors
```

### Add and delete points to the index
Points can be added to the index by using the `push!` and `pushfirst!` methods.
Removing points from the index can be performed using the `pop!`, `popfirst!` and
`delete_from_index!` methods.
```julia
for i in 1:15
    push!(ivfadc, rand(Float32, nrows))
end
length(ivfadc)
# 1015

delete_from_index!(ivfadc, [1000, 1001, 1010, 1015])
length(ivfadc)
# 1011
```

The `pop!` and `popfirst!` methods also return the indexed (and quantized) vectors respectively.
```julia
pop!(ivfadc)
# 50-element Array{Float32,1}:
#   0.30565456
#   0.6903644
#   ⋮
#   0.20116138
#   0.90699536

popfirst!(ivfadc)
# 50-element Array{Float32,1}:
#  0.29412186
#  0.0709379
#  ⋮
#  0.51727176
#  0.69718516

length(ivfadc)
# 09
```

### Search the index
```julia
point = data[:, 123];
idxs, dists = knn_search(ivfadc, point, 3)
# (UInt16[0x007a, 0x0237, 0x0081], Float32[4.303085, 10.026548, 10.06385])

int_idxs = Int.(idxs) .+ 1  # retrieve 1-based integer neighbors
# 3-element Array{Int64,1}:
#  123
#  568
#  130
```


## Features
To keep track with the latest features, please consult [NEWS.md](https://github.com/zgornel/IVFADC.jl/blob/master/NEWS.md) and the [documentation](https://zgornel.github.io/IVFADC.jl/dev).


## License

The code has an MIT license and therefore it is free.


## Reporting Bugs

This is work in progress and bugs may still be present...¯\\_(ツ)_/¯ Do not worry, just [open an issue](https://github.com/zgornel/IVFADC.jl/issues/new) to report a bug or request a feature.


## References

 - [Jègou et al. "Product quantization for nearest neighbor search", IEEE TPAMI, 2011](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
 - [Baranchuk et al. "Revisiting the inverted indices for billion-scale approximate nearest neighbors, ECCV, 2018"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.pdf)
