
test_files = ["additive_noise.jl", "affine_additive.jl", "nonlinear.jl", "uncertain_pwa.jl", "gaussian_process.jl", "stochastical_switched.jl"]
for f in test_files
    @testset "dynamics/$f" include(f)
end
