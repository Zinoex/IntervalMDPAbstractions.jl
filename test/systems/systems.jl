
test_files = ["affine_additive_gaussian.jl"]
for f in test_files
    @testset "systems/$f" include(f)
end