# Usage examples

Building an index can be performed with `build_index`
```@repl index
using IVFADC
data = rand(10, 10_000);

ivfadc = build_index(data, kc=500, k=256, m=5)
```

Searching into the index is done with `knn_search`
```@repl index
point = data[:, 1234];
knn_search(ivfadc, point, 5)
```

Results may vary depending on how many clusters are being used to search into
```@repl index
knn_search(ivfadc, point, 5, w=10)
```

Adding and removing points is done with
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
