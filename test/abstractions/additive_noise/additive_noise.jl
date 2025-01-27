
test_files = ["direct.jl", "decoupled.jl", "compare_abstractions.jl"]
for f in test_files
    @testset "abstractions/additive_noise/$f" include(f)
end
