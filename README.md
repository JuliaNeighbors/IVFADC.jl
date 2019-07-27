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

ivfadc = build_index(data,
                     kc=kc,
                     k=k,
                     m=m,
                     coarse_distance=SqEuclidean(),
                     quantization_distance=SqEuclidean(),
                     quantization_method=:pq)

# IVFADC Index, 100 coarse vectors, total of 1000-element Float32 vectors, UInt8 codes
```

### Add ond delete points to the index
```julia
for i in 1:15
    add_to_index!(ivfadc, rand(Float32, nrows))
end
length(ivfadc)
# 1015

delete_from_index!(ivfadc, [1, 2, 1010, 1015])
length(ivfadc)
# 1011
```

### Search the index
```julia
point = rand(Float32, nrows);
knn_search(ivfadc, point, 3)
# ([21, 263, 284], Float32[25.333912, 49.33256, 67.82121])
```


## Features
To keep track with the latest features, please consult [NEWS.md](https://github.com/zgornel/IVFADC.jl/blob/master/NEWS.md) and the [documentation](https://zgornel.github.io/IVFADC.jl/dev).


## License

The code has an MIT license and therefore it is free.


## Reporting Bugs

This is work in progress and bugs may still be present...¯\\_(ツ)_/¯ Do not worry, just [open an issue](https://github.com/zgornel/IVFADC.jl/issues/new) to report a bug or request a feature.


## References

 - [Jègou et al. "Product quantization for nearest neighbor search", IEEE TPAMI, 2011](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
