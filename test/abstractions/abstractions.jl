
@testset verbose=true "abstractions/additive_noise" include("additive_noise/additive_noise.jl")
@testset verbose=true "abstractions/gaussian_process.jl" include("gaussian_process.jl")
@testset verbose=true "abstractions/mixture.jl" include("mixture.jl")