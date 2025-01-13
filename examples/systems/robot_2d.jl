using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

"""
    InputRobot

Input abstraction ONLY for the 2D robot example.
"""
struct InputRobot <: IntervalMDPAbstractions.InputAbstraction
    input_space::Hyperrectangle
    ranges::NTuple{2,Real}
end
IntervalMDPAbstractions.numinputs(input::InputRobot) = prod(input.ranges)
IntervalMDPAbstractions.issetbased(input::InputRobot) = false
function IntervalMDPAbstractions.inputs(input::InputRobot)
    l = low(input.input_space)
    h = high(input.input_space)
    ranges = [LinRange(l, h, num_steps) for (l, h, num_steps) in zip(l, h, input.ranges)]

    regions = [
        Singleton([x[1] * cos(x[2]), x[1] * sin(x[2])]) for
        x in Iterators.product(ranges...)
    ]

    return regions
end

function robot_2d_sys(time_horizon; spec = :reachavoid)
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
    sys = System(dyn, initial_region)

    reach_region = Hyperrectangle(; low = [5.0, 5.0], high = [7.0, 7.0])
    if spec == :reachavoid
        avoid_region = Hyperrectangle(; low = [-2.0, -2.0], high = [2.0, 2.0])
        prop = FiniteTimeRegionReachAvoid(reach_region, avoid_region, time_horizon)
    elseif spec == :reachability
        prop = FiniteTimeRegionReachability(reach_region, time_horizon)
    else
        throw(ArgumentError("Invalid spec argument"))
    end
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function robot_2d_decoupled(
    time_horizon = 10;
    sparse = true,
    spec = :reachavoid,
    state_split = (40, 40),
    input_split = (21, 21),
)
    sys, spec = robot_2d_sys(time_horizon; spec = spec)

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputRobot(U, input_split)

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalMDPAbstractions.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function robot_2d_direct(
    time_horizon = 10;
    sparse = true,
    spec = :reachavoid,
    state_split = (40, 40),
    input_split = (21, 21),
)
    sys, spec = robot_2d_sys(time_horizon; spec = spec)

    X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
    input_abs = InputRobot(U, input_split)

    if sparse
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalMDPAbstractions.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function main()
    @time "abstraction reach-avoid" mdp_reachavoid, spec_reachavoid, _ =
        robot_2d_decoupled(;
            spec = :reachavoid,
            state_split = (40, 40),
            input_split = (21, 21),
        )
    prob_reachavoid = Problem(mdp_reachavoid, spec_reachavoid)

    @time "value-iteration reach-avoid" V_reachavoid, k_reachavoid, res_reachavoid =
        value_iteration(prob_reachavoid)
    V_reachavoid = V_reachavoid[2:end, 2:end]

    @time "abstraction reachability" mdp_reachability, spec_reachability, _ =
        robot_2d_decoupled(;
            spec = :reachability,
            state_split = (20, 20),
            input_split = (11, 11),
        )
    prob_reachability = Problem(mdp_reachability, spec_reachability)

    @time "value-iteration reachability" V_reachability, k_reachability, res_reachability =
        value_iteration(prob_reachability)
    V_reachability = V_reachability[2:end, 2:end]

    return V_reachavoid, V_reachability
end
