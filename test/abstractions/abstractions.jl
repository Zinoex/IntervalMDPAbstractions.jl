
test_files = ["direct.jl"]
for f in test_files
    @testset "dynamics/$f" include(f)
end