using Revise

using LinearAlgebra, LazySets, Plots
using IntervalMDP, IntervalSySCoRe


function van_der_pol_sys(; sampling_time=0.1)
    f(x, u) = [x[1] + x[2] * sampling_time, x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * sampling_time + u[1]]

    w_variance = [0.2, 0.2]
    w_stddev = sqrt.(w_variance)

    dyn = NonlinearAdditiveNoiseDynamics(f, 2, 1, AdditiveDiagonalGaussianNoise(w_stddev))

    initial_region = EmptySet(2)
    reach_region = Hyperrectangle(; low=[-1.4, -2.9], high=[-0.7, -2.0])
    avoid_region = EmptySet(2)

    sys = System(dyn, initial_region, reach_region, avoid_region)

    return sys
end

function van_der_pol_decoupled(; state_split=(50, 50), input_split=10)
    sys = van_der_pol_sys()

    X = Hyperrectangle(; low=[-4.0, -4.0], high=[4.0, 4.0])
    state_abs = StateUniformGridSplit(X, state_split)

    U = Hyperrectangle(; low=[-1.0], high=[1.0])
    input_abs = InputLinRange(U, input_split)

    target_model = DecoupledIMDP()

    mdp, reach, avoid = abstraction(sys, state_abs, input_abs, target_model)

    return mdp, reach, avoid
end

function van_der_pol_plot_nominal()
    sys = van_der_pol_sys()

    X = Hyperrectangle(low=[-4.0, -4.0], high=[4.0, 4.0])
    state_abs = StateUniformGridSplit(X, (50, 50))

    U = Hyperrectangle(low=[-1.0], high=[1.0])
    input_abs = InputLinRange(U, 10)

    R = IntervalSySCoRe.regions(state_abs)[837]
    u = IntervalSySCoRe.inputs(input_abs)[3]

    Y = nominal(dynamics(sys), R, u)

    xs = sample(R, 10)
    ys = [nominal(dynamics(sys), x, element(u)) for x in xs]

    p = plot(R, color=:blue, label="X")
    scatter!(p, Plots.unzip(Tuple.(xs)), color=:blue, label="Xhat", markershape=:xcross)
    plot!(p, Y, color=:red, label="Y")
    scatter!(p, Plots.unzip(Tuple.(ys)), color=:red, label="Yhat", markershape=:xcross)

    display(p)
end

function main()
    @time "abstraction" mdp, reach, avoid = van_der_pol_decoupled(; state_split=(200, 200), input_split=21)

    prop = FiniteTimeReachAvoid(reach, avoid, 10)
    spec = Specification(prop, Optimistic, Minimize)
    prob = Problem(mdp, spec)

    @time "value iteration" V, k, res = value_iteration(prob)
    V[2:end, 2:end]
end