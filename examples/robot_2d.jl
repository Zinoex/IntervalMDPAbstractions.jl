using LinearAlgebra, LazySets
using IntervalMDP, IntervalSySCoRe

"""
    InputRobot

Input abstraction ONLY for the 2D robot example.
"""
struct InputRobot <: IntervalSySCoRe.InputAbstraction
    input_space::Hyperrectangle
    ranges::NTuple{2, Real}
end
IntervalSySCoRe.numinputs(input::InputRobot) = prod(input.ranges)
function IntervalSySCoRe.inputs(input::InputRobot)
    l = low(input.input_space)
    h = high(input.input_space)
    ranges = [LinRange(l, h, num_steps) for (l, h, num_steps) in zip(l, h, input.ranges)]

    regions = [
        Singleton([x[1] * cos(x[2]), x[1] * sin(x[2])])
        for x in Iterators.product(ranges...)
    ]
    
    return regions
end

function robot_2d_sys(;spec=:reachavoid)
    # Dynamics
    # x₁[k + 1] = x₁[k] + u₁[k] * cos(u₂[k]) + w₁[k]
    # x₂[k + 1] = x₂[k] + u₁[k] * sin(u₂[k]) + w₂[k]

    A = zeros(Float64, 2, 2)
    A[1, 1] = 1.0
    A[2, 2] = 1.0

    # Transform inputs from u₁, u₂ to v₁ = u₁ * cos(u₂), v₂ = u₁ * sin(u₂)
    B = zeros(Float64, 2, 2)
    B[1, 1] = 10.0
    B[2, 2] = 10.0

    w_variance = [0.75, 0.75]
    w_stddev = sqrt.(w_variance)

    dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    reach_region = Hyperrectangle(; low=[5.0, 5.0], high=[7.0, 7.0])
    if spec == :reachavoid
        avoid_region = Hyperrectangle(; low=[-2.0, -2.0], high=[2.0, 2.0])
    elseif spec == :reachability
        avoid_region = EmptySet(2)
    else
        throw(ArgumentError("Invalid spec argument"))
    end

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function robot_2d_decoupled(; spec=:reachavoid, state_split=(40, 40), input_split=(20, 20))
    sys = robot_2d_sys(;spec=spec)

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    state_abs = StateGridSplit(X, state_split)

    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])
    input_abs = InputRobot(U, input_split)

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function main()
    mdp_reachavoid, reach_reachavoid, avoid_reachavoid = robot_2d_decoupled(;spec=:reachavoid, state_split=(40, 40), input_split=(20, 20))

    prop_reachavoid = FiniteTimeReachAvoid(reach_reachavoid, avoid_reachavoid, 10)
    spec_reachavoid = Specification(prop_reachavoid, Pessimistic, Maximize)
    prob_reachavoid = Problem(mdp_reachavoid, spec_reachavoid)

    V_reachavoid, k_reachavoid, res_reachavoid = value_iteration(prob_reachavoid)

    mdp_reachability, reach_reachability, avoid_reachability = robot_2d_decoupled(;spec=:reachability, state_split=(20, 20), input_split=(10, 10))

    prop_reachability = FiniteTimeReachAvoid(reach_reachability, avoid_reachability, 10)
    spec_reachability = Specification(prop_reachability, Pessimistic, Maximize)
    prob_reachability = Problem(mdp_reachability, spec_reachability)

    V_reachability, k_reachability, res_reachability = value_iteration(prob_reachability)
end