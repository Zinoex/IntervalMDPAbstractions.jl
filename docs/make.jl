using IntervalMDPAbstractions
using Documenter

DocMeta.setdocmeta!(
    IntervalMDPAbstractions,
    :DocTestSetup,
    :(using IntervalMDPAbstractions);
    recursive = true,
)

makedocs(;
    modules = [IntervalMDPAbstractions],
    authors = "Frederik Baymler Mathiesen",
    sitename = "IntervalMDPAbstractions.jl",
    format = Documenter.HTML(;
        canonical = "https://Zinoex.github.io/IntervalMDPAbstractions.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/Zinoex/IntervalMDPAbstractions.jl", devbranch = "main")
