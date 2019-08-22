function save_ivfadc_index(filename::AbstractString,
                           ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q}
                          ) where {U,I,Dc,Dr,T,Q<:NaiveQuantizer}
    open(filename, "w") do fid
        # Initialize all variables needed to write
        nrows, nclusters = size(ivfadc.coarse_quantizer.vectors)
        n = length(ivfadc)
        m = length(ivfadc.residual_quantizer.codebooks)
        k = ivfadc.residual_quantizer.k
        d = size(ivfadc.residual_quantizer.codebooks[1].vectors, 1)

        # Get types
        coarsequanttype_str = "NaiveQuantizer"
        quanttype_str = string(typeof(ivfadc.residual_quantizer.quantization))
        quanteltype_str = string(U)
        indextype_str = string(I)
        disttypecoarse_str = string(Dc)
        disttyperesidual_str = string(Dr)
        origeltype_str = string(T)

        # Start writing text information
        println(fid, "$nrows $nclusters")
        println(fid, "$n $m $k $d")
        println(fid, "$coarsequanttype_str")
        println(fid, "$quanttype_str")
        println(fid, "$quanteltype_str")
        println(fid, "$indextype_str")
        println(fid, "$disttypecoarse_str")
        println(fid, "$disttyperesidual_str")
        println(fid, "$origeltype_str")

        # Write coarse quantizer
        _write_naive_coarse_quantizer(fid, ivfadc.coarse_quantizer)

        # Write residual quantizer
        _write_residual_quantizer(fid, ivfadc.residual_quantizer, (nrows, nclusters))

        # Write inverse index
        _write_inverse_index(fid, ivfadc.inverse_index, (nrows, nclusters))
    end
end


_write_naive_coarse_quantizer(fid::IO, quantizer) = begin
    nclusters = size(quantizer.vectors, 2)
    for i in 1:nclusters
        write(fid, quantizer.vectors[:,i])
    end
end


_write_residual_quantizer(fid::IO, quantizer, dims) = begin
    nrows, nclusters = dims
    d = size(quantizer.codebooks[1].vectors, 1)
    m = length(quantizer.codebooks)
    for i in 1:m  # codebooks
        write(fid, quantizer.codebooks[i].codes)
        for j in 1:d
            write(fid, quantizer.codebooks[i].vectors[j,:])
        end
    end
    for i in 1:nrows  # rotation matrix
        write(fid, quantizer.rot[:,i])
    end
end


_write_inverse_index(fid::IO, inverse_index, dims) = begin
    nrows, nclusters = dims
    for i in 1:nclusters
        clsize = length(inverse_index[i].idxs)
        write(fid, clsize)
        write(fid, inverse_index[i].idxs)
        for j in 1:clsize
            write(fid, inverse_index[i].codes[j])
        end
    end
end


# Generate a WordVectors object from binary file
function load_ivfadc_index(filename::AbstractString)
    open(filename, "r") do fid
        nrows, nclusters = map(x -> parse(Int, x), split(readline(fid), ' '))
        n, m, k, d = map(x -> parse(Int, x), split(readline(fid), ' '))
        CQ = _read_type_from_line(readline(fid))
        Q = _read_type_from_line(readline(fid))
        U = eval(Symbol(readline(fid)))
        I = eval(Symbol(readline(fid)))
        Dc = _read_type_from_line(readline(fid))
        Dr = _read_type_from_line(readline(fid))
        T = eval(Symbol(readline(fid)))

        # Read coarse quantizer
        coarse_quantizer = _load_coarse_quantizer(CQ, fid;
                                dims=(nrows, nclusters),
                                disttype=Dc,
                                vectype=T)

        # Read residual quantizer
        cbooks = Vector{CodeBook{U,T}}(undef, m)
        codes_length = sizeof(U) * k
        codevecs_length = sizeof(T) * k
        for i = 1:m
            codes = collect(reinterpret(U, read(fid, codes_length)))
            vectors = Matrix{T}(undef, d, k)
            for j in 1:d
                vectors[j,:] = collect(reinterpret(T, read(fid, codevecs_length)))
            end
            cbooks[i] = CodeBook(codes, vectors)
        end
        binary_length = sizeof(T) * nrows
        rotmat = Matrix{T}(undef, nrows, nrows)
        for i in 1:nrows
            rotmat[:,i] = collect(reinterpret(T, read(fid, binary_length)))
        end
        residual_quantizer = ArrayQuantizer(Q(), (nrows, n), cbooks, k, Dr(), rotmat)

        # Read inverse index
        inverse_index = IVFADC.InvertedIndex{I,U}(undef, nclusters)
        encoded_length = sizeof(U) * m
        for i in 1:nclusters
            clsize = reinterpret(Int, read(fid, sizeof(Int)))[1]
            idxs_length = sizeof(I) * clsize
            idxs = collect(reinterpret(I, read(fid, idxs_length)))
            codes = Vector{Vector{U}}(undef, clsize)
            for j in 1:clsize
                codes[j] = collect(reinterpret(U, read(fid, encoded_length)))
            end
            inverse_index[i] = InvertedList(idxs, codes)
        end
        return IVFADCIndex(coarse_quantizer, residual_quantizer, inverse_index)
    end
end


_read_type_from_line(line) = begin
    if occursin('.', line)
        _module, _type = String.(split(line, "."))
    else
        _module, _type = string(@__MODULE__), line
    end
    return eval(Expr(:., Symbol(_module), QuoteNode(Symbol(_type))))
end


_load_coarse_quantizer(::Type{CQ},
                       fid::IO;
                       dims=(0,0),
                       disttype::Type{Dc}=typeof(DEFAULT_COARSE_DISTANCE),
                       vectype::Type{T}=AbstractFloat
                      ) where {CQ<:NaiveQuantizer,Dc,T} = begin
    nrows, nclusters = dims
    data = Matrix{T}(undef, nrows, nclusters)
    binary_length = sizeof(T) * nrows
    for i in 1:nclusters
        data[:,i] = collect(reinterpret(T, read(fid, binary_length)))
    end
    coarse_quantizer = IVFADC.NaiveQuantizer{Dc,T}(data)
end


function save_ivfadc_index(filename::AbstractString,
                           ivfadc::IVFADCIndex{U,I,Dc,Dr,T,Q}
                          ) where {U,I,Dc,Dr,T,Q<:HNSWQuantizer}
    open(filename, "w") do fid
        # Initialize all variables needed to write
        nrows = length(ivfadc.coarse_quantizer.hnsw.data[1])
        nclusters = length(ivfadc.coarse_quantizer.hnsw.data)
        n = length(ivfadc)
        m = length(ivfadc.residual_quantizer.codebooks)
        k = ivfadc.residual_quantizer.k
        d = size(ivfadc.residual_quantizer.codebooks[1].vectors, 1)

        # Get types
        coarsequanttype_str = "HNSWQuantizer"
        quanttype_str = string(typeof(ivfadc.residual_quantizer.quantization))
        quanteltype_str = string(U)
        indextype_str = string(I)
        disttypecoarse_str = string(Dc)
        disttyperesidual_str = string(Dr)
        origeltype_str = string(T)

        # Start writing text information
        println(fid, "$nrows $nclusters")
        println(fid, "$n $m $k $d")
        println(fid, "$coarsequanttype_str")
        println(fid, "$quanttype_str")
        println(fid, "$quanteltype_str")
        println(fid, "$indextype_str")
        println(fid, "$disttypecoarse_str")
        println(fid, "$disttyperesidual_str")
        println(fid, "$origeltype_str")

        # Write coarse quantizer
        _write_hnsw_coarse_quantizer(fid, ivfadc.coarse_quantizer.hnsw)

        # Write residual quantizer
        _write_residual_quantizer(fid, ivfadc.residual_quantizer, (nrows, nclusters))

        # Write inverse index
        _write_inverse_index(fid, ivfadc.inverse_index, (nrows, nclusters))
    end
end


_write_hnsw_coarse_quantizer(fid::IO,
                             hnsw::HierarchicalNSW{T,F,V,M}
                            ) where {T,F,V,M} = begin
    println(fid, string(T))
    nclusters= length(hnsw.data)

    # Start writing
    write(fid, hnsw.lgraph.M0)          # .lgraph
    write(fid, hnsw.lgraph.M)
    write(fid, hnsw.lgraph.m_L)
    nlinkedlists = length(hnsw.lgraph.linklist)
    write(fid, nlinkedlists)
    for i in 1:nlinkedlists
        listlength = length(hnsw.lgraph.linklist[i])
        write(fid, listlength)
        write(fid, hnsw.lgraph.linklist[i])
    end
    write(fid, hnsw.added)              # .added
    for i in 1:nclusters                # .data
        write(fid, hnsw.data[i])
    end
    write(fid, hnsw.ep)                 # .ep
    write(fid, hnsw.entry_level)        # .entry_level
    write(fid, hnsw.vlp.num_elements)   # .vlp
    nvlps = length(hnsw.vlp.pool)
    write(fid, nvlps)
    for i in 1:nvlps
        write(fid, hnsw.vlp.pool[i].visited_value)  # UInt8
        write(fid, length(hnsw.vlp.pool[i].list))
        write(fid, hnsw.vlp.pool[i].list)
    end
    # metric can be skipped             # .metric
    write(fid, hnsw.efConstruction)     # .efConstruction
    write(fid, hnsw.ef)                 # .ef
end


_load_coarse_quantizer(::Type{CQ},
                       fid::IO;
                       dims=(0,0),
                       disttype::Type{Dc}=typeof(DEFAULT_COARSE_DISTANCE),
                       vectype::Type{F}=AbstractFloat
                      ) where {CQ<:HNSWQuantizer,Dc,F} = begin
    # We already have F (float type) and M (metric type = Dc)
    T = eval(Symbol(readline(fid)))
    V = Vector{Vector{F}}
    intsize = sizeof(Int)
    nrows, nclusters = dims

    # .lgraph
    M0 = reinterpret(Int, read(fid, intsize))[1]
    M = reinterpret(Int, read(fid, intsize))[1]
    m_L = reinterpret(Float64, read(fid, 8))[1]  # a Float64
    nlinkedlists = reinterpret(Int, read(fid, intsize))[1]
    linklist = Vector{Vector{T}}(undef, nlinkedlists)
    for i in 1:nlinkedlists
        listlength = reinterpret(Int, read(fid, intsize))[1]
        linklist[i] = collect(reinterpret(T, read(fid, sizeof(T) * listlength)))
    end
    lgraph = HNSW.LayeredGraph{T}(linklist, M0, M, m_L)

    # .added
    added = collect(reinterpret(Bool, read(fid, nclusters * sizeof(Bool))))

    # .data
    data = V(undef, nclusters)
    for i in 1:nclusters
        data[i] = collect(reinterpret(F, read(fid, nrows * sizeof(F))))
    end

    # .ep
    ep = reinterpret(T, read(fid, sizeof(T)))[1]

    # .entry_level
    entry_level = reinterpret(Int, read(fid, intsize))[1]

    # .vlp
    num_elements = reinterpret(Int, read(fid, intsize))[1]
    nvlps = reinterpret(Int, read(fid, intsize))[1]
    pool = Vector{HNSW.VisitedList}(undef, nvlps)
    for i in 1:nvlps
        visited_value = read(fid, 1)[1]
        len = reinterpret(Int, read(fid, intsize))[1]
        list = read(fid, len)
        pool[i] = HNSW.VisitedList(list, visited_value)
    end
    vlp = HNSW.VisitedListPool(pool, num_elements)

    # .metric need not be written

    # .efConstruction
    efConstruction = reinterpret(Int, read(fid, intsize))[1]

    # .ef
    ef = reinterpret(Int, read(fid, intsize))[1]
    hnsw = HNSW.HierarchicalNSW{T,F,V,Dc}(lgraph, added, data,
                ep, entry_level, vlp, Dc(), efConstruction, ef)
    return HNSWQuantizer{T,V,Dc,F}(hnsw)
end
