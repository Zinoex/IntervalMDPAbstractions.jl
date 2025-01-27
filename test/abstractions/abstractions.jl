
@testset verbose=true "abstractions/additive_noise" include("additive_noise/additive_noise.jl")
@testset "abstractions/gaussian_process.jl" include("gaussian_process.jl")
@testset "abstractions/mixture.jl" include("mixture.jl")