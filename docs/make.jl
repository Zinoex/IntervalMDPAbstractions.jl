using IntervalSySCoRe
using Documenter

DocMeta.setdocmeta!(IntervalSySCoRe, :DocTestSetup, :(using IntervalSySCoRe); recursive=true)

makedocs(;
    modules=[IntervalSySCoRe],
    authors="Frederik Baymler Mathiesen",
    sitename="IntervalSySCoRe.jl",
    format=Documenter.HTML(;
        canonical="https://Zinoex.github.io/IntervalSySCoRe.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Zinoex/IntervalSySCoRe.jl",
    devbranch="main",
)
