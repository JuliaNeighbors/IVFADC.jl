# Usage examples

## Building an IVFADC index
Building an index can be performed with `build_index`
```@repl index
using IVFADC
data = rand(10, 100);  # 100 points, 10 dimensions

ivfadc = build_index(data, kc=20, k=32, m=2, index_type=UInt8)
```

## Searching the index
Searching into the index is done with `knn_search`
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

Results may vary depending on how many clusters are being used to search into
```@repl index
knn_search(ivfadc, point, 5, w=10)  # search into 10 clusters
```

## Updating the index
Adding and removing points to and from the index is done with the
`add_to_index!` and `delete_from_index!` functions
```@repl index
for i in 1:5
    add_to_index!(ivfadc, rand(10))
end
ivfadc
```
and
```@repl index
delete_from_index!(ivfadc, [1,2,3]);
ivfadc
```
respectively.

!!! note

    When adding a new point, its index will always be the number of already existing points.
    When deleting points, the indexes of all points are updated so that they are consecutive.

## Limits and advanced usage of the index
It is not possible to add more points than the maximum value of the indexing type
```@repl index
ivfadc = build_index(rand(2,256), kc=2, k=16, m=1, index_type=UInt8)
```
Adding a new point to an index that already has 256 points (which is the maximum for the `UInt8`)
throws an error
```
add_to_index!(ivfadc, rand(2))
```
It is however possible to add the point after deleting another one
```@repl index
delete_from_index!(ivfadc, [1])
add_to_index!(ivfadc, rand(2))
```
In the example above, the index is being used as a FIFO buffer where the first point is removed
and the last one appended to the buffer.

!!! note

    Deleting from the index is a slow operation as the all indexes of the points contained
    need to be properly updated depending on the positions of the points that are being deleted.
