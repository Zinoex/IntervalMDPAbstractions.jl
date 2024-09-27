using MAT, MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5
const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

function load_dynamics(partitions::MatlabFile)
    # Extract hypercube data
    state_partitions = read(partitions, "partitions")

    # Extract Neural Network Bounds [CROWN]
    M_upper = read(partitions, "M_h")
    M_lower = read(partitions, "M_l")
    b_upper = read(partitions, "B_h")
    b_lower = read(partitions, "B_l")

    n = size(state_partitions, 1)

    Xs = [
        UncertainAffineRegion(
            Hyperrectangle(low=state_partitions[ii, 1, :], high=state_partitions[ii, 2, :]),
            convert(Matrix{Float64}, transpose(M_lower[ii, :, :])), b_lower[ii, :], 
            convert(Matrix{Float64}, transpose(M_upper[ii, :, :])), b_upper[ii, :]
        ) for ii in 1:n
    ]

    return Xs
end

function load_system(system_name::String, number_hypercubes::Int)
    filename = joinpath(@__DIR__, "nndm_data/$(system_name)_partition_data_$number_hypercubes.mat")
    file = matopen(joinpath(@__DIR__, filename))

    dynamics = load_dynamics(file)

    close(file)

    return dynamics
end

function cartpole_sys()
    pwa_dyn = load_system("cartpole", 3840)
    w = AdditiveDiagonalGaussianNoise([0.01, 0.01, 0.01, 0.01])
    dyn = UncertainPWAAdditiveNoiseDynamics(4, 0, pwa_dyn, w)

    initial = EmptySet(4)
    reach = EmptySet(4)
    avoid = EmptySet(4)

    sys = System(dyn, initial, reach, avoid)

    return sys
end


function cartpole_decoupled(; sparse=false)
    sys = cartpole_sys()

    X = Hyperrectangle(; low=[-1.0, -0.5, deg2rad(-12.0), -0.5], high=[1.0, 0.5, deg2rad(12.0), 0.5])
    state_split = (10, 4, 24, 4)
    state_abs = StateUniformGridSplit(X, state_split)

    input_abs = InputDiscrete([Universe(0)])

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end