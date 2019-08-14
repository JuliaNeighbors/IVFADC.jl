```@meta
CurrentModule=IVFADC
```

# Introduction

`IVFADC` implements an inverted file system with asymmetric distance computation for fast approximate nearest neighbor search in large i.e. billion-scale, high dimensional datasets.

## Implemented features
 - building the index: the outer constructor `IVFADCIndex`
 - adding points to the index: `push!`, `pushfirst!`
 - removing points from the index: `pop!`, `popfirst!`, `delete_from_index!`
 - searching into the index: `knn_search`
 - saving/loading the index to/from disk: `save_ivfadc_index`, `load_ivfadc_index`

## Installation

Installation can be performed from either inside or outside Julia.

### Git cloning
```
$ git clone https://github.com/zgornel/IVFADC.jl
```

### Julia REPL
The package can be installed from inside Julia with:
```
using Pkg
Pkg.add("IVFADC")
```
or
```
Pkg.add(PackageSpec(url="https://github.com/zgornel/IVFADC.jl", rev="master"))
```
for the latest `master` branch.
