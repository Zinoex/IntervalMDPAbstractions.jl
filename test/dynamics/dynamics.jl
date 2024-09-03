
test_files = ["linear_additive_gaussian.jl"]
for f in test_files
    @testset "dynamics/$f" include(f)
end