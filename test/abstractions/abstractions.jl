
test_files = ["additive_noise/additive_noise.jl", "mixture.jl"]
for f in test_files
    @testset "abstractions/$f" include(f)
end
