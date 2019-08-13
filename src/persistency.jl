function save_ivfadc_index(filename::AbstractString, ivfadc::IVFADCIndex{U,I,Dc,Dr,T}
                          ) where {U,I,Dc,Dr,T}
    open(filename, "w") do fid
        # Initialize all variables needed to write
        nrows, nclusters = size(ivfadc.coarse_quantizer.vectors)
        n = length(ivfadc)
        m = length(ivfadc.residual_quantizer.codebooks)
        k = ivfadc.residual_quantizer.k
        d = size(ivfadc.residual_quantizer.codebooks[1].vectors, 1)

        # Get types
        quanttype_str = string(typeof(ivfadc.residual_quantizer.quantization))
        quanteltype_str = string(U)
        indextype_str = string(I)
        disttypecoarse_str = string(Dc)
        disttyperesidual_str = string(Dr)
        origeltype_str = string(T)

        # Start writing text information
        println(fid, "$nrows $nclusters")
        println(fid, "$n $m $k $d")
        println(fid, "$quanttype_str")
        println(fid, "$quanteltype_str")
        println(fid, "$indextype_str")
        println(fid, "$disttypecoarse_str")
        println(fid, "$disttyperesidual_str")
        println(fid, "$origeltype_str")

        # Write coarse quantizer
        for i in 1:nclusters
            write(fid, ivfadc.coarse_quantizer.vectors[:,i])
        end

        # Write residual quantizer
        for i in 1:m  # codebooks
            write(fid, ivfadc.residual_quantizer.codebooks[i].codes)
            for j in 1:d
                write(fid, ivfadc.residual_quantizer.codebooks[i].vectors[j,:])
            end
        end
        for i in 1:nrows  # rotation matrix
            write(fid, ivfadc.residual_quantizer.rot[:,i])
        end

        # Write inverse index
        for i in 1:nclusters
            clsize = length(ivfadc.inverse_index[i].idxs)
            write(fid, clsize)
            write(fid, ivfadc.inverse_index[i].idxs)
            for j in 1:clsize
                write(fid, ivfadc.inverse_index[i].codes[j])
            end
        end
    end
end


# Generate a WordVectors object from binary file
function load_ivfadc_index(filename::AbstractString)
    open(filename, "r") do fid
        nrows, nclusters = map(x -> parse(Int, x), split(readline(fid), ' '))
        n, m, k, d = map(x -> parse(Int, x), split(readline(fid), ' '))
        _module, _val = split(readline(fid), ".")
        Q = eval(Expr(:., Symbol(_module), QuoteNode(Symbol(_val))))
        U = eval(Symbol(readline(fid)))
        I = eval(Symbol(readline(fid)))
        _module, _val = split(readline(fid), ".")
        Dc = eval(Expr(:., Symbol(_module), QuoteNode(Symbol(_val))))
        _module, _val = split(readline(fid), ".")
        Dr = eval(Expr(:., Symbol(_module), QuoteNode(Symbol(_val))))
        T = eval(Symbol(readline(fid)))

        # Read coarse quantizer
        data = Matrix{T}(undef, nrows, nclusters)
        binary_length = sizeof(T) * nrows
        for i in 1:nclusters
            data[:,i] = collect(reinterpret(T, read(fid, binary_length)))
        end
        coarse_quantizer = IVFADC.CoarseQuantizer(data, Dc())

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
