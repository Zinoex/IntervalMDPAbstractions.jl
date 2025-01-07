using LinearAlgebra, LazySets, Plots
using IntervalMDP, IntervalSySCoRe


function van_der_pol_sys(time_horizon; sampling_time = 0.1)
    f(x, u) = [
        x[1] + x[2] * sampling_time,
        x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * sampling_time + u[1],
    ]

    w_variance = [0.2, 0.2]
    w_stddev = sqrt.(w_variance)

    dyn = NonlinearAdditiveNoiseDynamics(f, 2, 1, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    sys = System(dyn, initial_region)

    reach_region = Hyperrectangle(; low = [-1.4, -2.9], high = [-0.7, -2.0])
    prop = FiniteTimeRegionReachability(reach_region, time_horizon)
    spec = Specification(prop, Pessimistic, Maximize)

    return sys, spec
end

function van_der_pol_decoupled(
    time_horizon = 10;
    sparse = false,
    state_split = (50, 50),
    input_split = 10,
)
    sys, spec = van_der_pol_sys(time_horizon)

    X = Hyperrectangle(; low = [-4.0, -4.0], high = [4.0, 4.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [-1.0], high = [1.0])
    input_abs = InputLinRange(U, input_split)

    if sparse
        target_model = SparseOrthogonalIMDPTarget()
    else
        target_model = OrthogonalIMDPTarget()
    end

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function van_der_pol_direct(time_horizon = 10; state_split = (50, 50), input_split = 10)
    sys, spec = van_der_pol_sys(time_horizon)

    X = Hyperrectangle(; low = [-4.0, -4.0], high = [4.0, 4.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low = [-1.0], high = [1.0])
    input_abs = InputLinRange(U, input_split)

    target_model = SparseIMDPTarget()

    prob = AbstractionProblem(sys, spec)
    mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

    upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
    upper_bound_spec =
        IntervalSySCoRe.convert_specification(upper_bound_spec, state_abs, target_model)

    return mdp, abstract_spec, upper_bound_spec
end

function van_der_pol_plot_nominal()
    sys, spec = van_der_pol_sys(1)

    X = Hyperrectangle(low = [-4.0, -4.0], high = [4.0, 4.0])
    state_abs = StateUniformGridSplit(X, (50, 50))

    U = Hyperrectangle(low = [-1.0], high = [1.0])
    input_abs = InputLinRange(U, 10)

    R = IntervalSySCoRe.regions(state_abs)[837]
    u = IntervalSySCoRe.inputs(input_abs)[3]

    Y = nominal(dynamics(sys), R, u)

    xs = sample(R, 100)
    ys = [nominal(dynamics(sys), x, element(u)) for x in xs]

    p = plot(R, color = :blue, label = "X")
    scatter!(
        p,
        Plots.unzip(Tuple.(xs)),
        color = :blue,
        label = "Xhat",
        markershape = :xcross,
    )
    plot!(p, Y, color = :red, label = "Y")
    scatter!(
        p,
        Plots.unzip(Tuple.(ys)),
        color = :red,
        label = "Yhat",
        markershape = :xcross,
    )

    display(p)
end

function main()
    @time "abstraction" mdp, spec, upper_bound_spec =
        van_der_pol_decoupled(; state_split = (100, 100), input_split = 3)
    prob = Problem(mdp, spec)

    @time "value iteration" V, k, res = value_iteration(prob)
    return V[2:end, 2:end]
end
