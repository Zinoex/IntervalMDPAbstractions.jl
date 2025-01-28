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
    authors = "Frederik Baymler Mathiesen <frederik@baymler.com> and contributors",
    sitename = "IntervalMDPAbstractions.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://www.baymler.com/IntervalMDPAbstractions.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md"
        "Reference" => Any[
            "Dynamics"=>"reference/dynamics.md",
            "Specifications"=>"reference/specifications.md",
            "Abstractions"=>"reference/abstractions.md",
        ]
    ],
    doctest = true,
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/Zinoex/IntervalMDPAbstractions.jl", devbranch = "main")
