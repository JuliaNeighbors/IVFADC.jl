using Pkg
Pkg.add("Documenter")
using Documenter, IVFADC

# Make src directory available
push!(LOAD_PATH,"../src/")

# Make documentation
makedocs(
    modules = [IVFADC],
    format = Documenter.HTML(),
    sitename = "  ",
    authors = "Corneliu Cofaru, 0x0α Research",
    clean = true,
    debug = true,
    pages = [
        "Introduction" => "index.md",
        "Usage examples" => "examples.md",
        "API Reference" => "api.md",
    ]
)

# Deploy documentation
deploydocs(
    repo = "github.com/zgornel/IVFADC.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
