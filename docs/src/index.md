```@meta
CurrentModule=IVFADC
```

# Introduction

`IVFADC` implements an inverted file system with asymmetric distance computation for fast approximate nearest neighbor search in large i.e. billion-scale, high dimensional datasets.

!!! warning

    This package is under heavy development and should not be used at this point in production systems.


## Implemented features
 - building the index
 - adding files to the index
 - removing files from the index
 - searching into the index

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
