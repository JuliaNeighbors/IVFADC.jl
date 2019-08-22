# Usage examples

## Building an IVFADC index
Building an index can be performed using an outer constructor
```@repl index
using IVFADC
data = rand(10, 100);  # 100 points, 10 dimensions

ivfadc = IVFADCIndex(data, kc=5, k=8, m=2)
```
The coarse quantizer, used in the first level coarse neighbor search, can be specified using the
`coarse_quantizer` keyword argument
```@repl index
ivfadc = IVFADCIndex(data, kc=5, k=8, m=2, coarse_quantizer=:hnsw)  # fast!
ivfadc = IVFADCIndex(data, kc=5, k=8, m=2, coarse_quantizer=:naive) # simple
```
The HNSW coarse quantizer is recommended is 'many' clusters are being used and coarse search
dominates search time as opposed to in-cluster search. 


## Searching the index
Searching into the index is done with `knn_search` for multiple queries
```@repl index
points = [rand(10) for _ in 1:3];
idxs, dists = knn_search(ivfadc, points, 5)
```
as well as single queries
```@repl index
point = data[:, 55];
idxs, dists = knn_search(ivfadc, point, 5)
```
Internally, the IVFADC index uses 0-based indexing; to retrieve the actual 1-based
neighbor indexes that correspond to indexes in `data`, a simple transform
has to be performed:
```@repl index
int_idxs = Int.(idxs) .+ 1
```

Results may vary depending on how many clusters are being used to search into, option configurable through the keyword argument `w` of `knn_search`
```@repl index
knn_search(ivfadc, point, 5, w=10)  # search into 10 clusters
```

## Updating the index
Adding and removing points to and from the index is done with
`push!`, `pop!`, `pushfirst!` and `popfirst!` methods. As they imply,
point can be added (and quantized) or removed (and reconstructed) at the
beginning or end of the index. In practice, this implies updating the point
indexes in the index, if the case.
```@repl index
for i in 1:5
    push!(ivfadc, rand(10))
end
ivfadc
pop!(ivfadc)
pushfirst!(ivfadc, 0.1*collect(1:10))
popfirst!(ivfadc)
```
!!! note

    When adding a new point, its index will always be the number of already existing points.
    When deleting points, the indexes of all points are updated so that they are consecutive.

The function `delete_from_index!` removes the points indicated in the vector of indexes.
```@repl index
delete_from_index!(ivfadc, [10, 21, 32]);
ivfadc
```

!!! note

    Deleting from the index is a slow operation as the all indexes of the points contained
    need to be properly updated depending on the positions of the points that are being deleted.

## Limits and advanced usage of the index
It is not possible to add more points than the maximum value of the indexing type
```@repl index
ivfadc = IVFADCIndex(rand(2,256), kc=2, k=16, m=1, index_type=UInt8)
```
Adding a new point to an index that already has 256 points (which is the maximum for the `UInt8`)
throws an error
```@repl index
push!(ivfadc, rand(2))
```
It is however possible to add the point after deleting another one
```@repl index
popfirst!(ivfadc)
push!(ivfadc, rand(2))
ivfadc
```
In the example above, the index is being used as a FIFO buffer where the first point is removed
and a new one is appended to the buffer.
