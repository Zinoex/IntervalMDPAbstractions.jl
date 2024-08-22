
test_files = ["direct.jl", "decoupled.jl", "compare_abstractions.jl"]
for f in test_files
    @testset "dynamics/$f" include(f)
end