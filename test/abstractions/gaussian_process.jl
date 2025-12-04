using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

# System definition

# Action 1
gp_region1_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[-0.5, -0.5], high=[0.0, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region2_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[0.0, -0.5], high=[0.5, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region3_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[-0.5, 0.0], high=[0.0, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region4_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[0.0, 0.0], high=[0.5, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)

gp_action1 = [gp_region1_action1, gp_region2_action1, gp_region3_action1, gp_region4_action1]

# Action 2
gp_region1_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[-0.5, -0.5], high=[0.0, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region2_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[0.0, -0.5], high=[0.5, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region3_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[-0.5, 0.0], high=[0.0, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)
gp_region4_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(; low=[0.0, 0.0], high=[0.5, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4]
)

gp_action2 = [gp_region1_action2, gp_region2_action2, gp_region3_action2, gp_region4_action2]

# Noise
w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = AbstractedGaussianProcess([gp_action1, gp_action2])
initial_region = Hyperrectangle(; low=[-0.1, -0.1], high=[0.1, 0.1])
sys = System(dyn, initial_region)

horizon = 10
avoid_region = EmptySet(2)
prop = FiniteTimeRegionSafety(avoid_region, horizon)
spec = Specification(prop, Pessimistic, Maximize)

prob = AbstractionProblem(sys, spec)

X = Hyperrectangle(; low=[-0.5, -0.5], high=[0.5, 0.5])
state_abs = StateUniformGridSplit(X, (2, 2))
input_abs = InputDiscrete([1, 2])

@testset "direct vs decoupled" begin
    # Decoupled
    target_model = OrthogonalIMDPTarget()
    mdp_decoupled, abstract_spec_decoupled = abstraction(prob, state_abs, input_abs, target_model)

    @test state_values(mdp_decoupled) == (3, 3)
    @test action_values(mdp_decoupled) == (2,)  # 2 control actions

    prob_decoupled = VerificationProblem(mdp_decoupled, abstract_spec_decoupled)

    V_decoupled, k, res = solve(prob_decoupled)
    @test k == 10

    # Direct
    target_model = IMDPTarget()
    mdp_direct, abstract_spec_direct = abstraction(prob, state_abs, input_abs, target_model)

    @test state_values(mdp_direct) == (2 * 2 + 1,)
    @test action_values(mdp_direct) == (2,)  # 2 control actions

    prob_direct = VerificationProblem(mdp_direct, abstract_spec_direct)

    V_direct, k, res = solve(prob_direct)
    @test k == 10

    @test all(V_decoupled[1:(end - 1), 1:(end - 1)] .≥ reshape(V_direct[1:(end - 1)], 2, 2))
end
