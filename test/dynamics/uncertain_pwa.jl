using Revise, Test
using IntervalMDPAbstractions, LazySets

# System definition

# Action 1
upwa_region1_action1 = UncertainAffineRegion(
    Hyperrectangle(low = [-0.5, -0.5], high = [0.0, 0.0]),
    [1.0 0.1; -0.1 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.1 1.1],
    [0.0, 0.5],
)
upwa_region2_action1 = UncertainAffineRegion(
    Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0]),
    [1.0 0.1; -0.2 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.2 1.1],
    [0.0, 0.5],
)
upwa_region3_action1 = UncertainAffineRegion(
    Hyperrectangle(low = [-0.5, 0.0], high = [0.0, 0.5]),
    [1.0 0.1; -0.3 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.3 1.1],
    [0.0, 0.5],
)
upwa_region4_action1 = UncertainAffineRegion(
    Hyperrectangle(low = [0.0, 0.0], high = [0.5, 0.5]),
    [1.0 0.1; -0.4 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.4 1.1],
    [0.0, 0.5],
)

upwa_action1 = [upwa_region1_action1, upwa_region2_action1, upwa_region3_action1, upwa_region4_action1]

# Action 2
upwa_region1_action2 = UncertainAffineRegion(
    Hyperrectangle(low = [-0.5, -0.5], high = [0.0, 0.0]),
    [1.0 0.1; -0.1 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.0 1.1],
    [0.0, 0.5],
)
upwa_region2_action2 = UncertainAffineRegion(
    Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0]),
    [1.0 0.1; -0.2 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.0 1.1],
    [0.0, 0.5],
)
upwa_region3_action2 = UncertainAffineRegion(
    Hyperrectangle(low = [-0.5, 0.0], high = [0.0, 0.5]),
    [1.0 0.1; -0.3 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.0 1.1],
    [0.0, 0.5],
)
upwa_region4_action2 = UncertainAffineRegion(
    Hyperrectangle(low = [0.0, 0.0], high = [0.5, 0.5]),
    [1.0 0.1; -0.4 1.1],
    [0.0, 0.5],
    [1.0 0.1; 0.0 1.1],
    [0.0, 0.5],
)

upwa_action2 = [upwa_region1_action2, upwa_region2_action2, upwa_region3_action2, upwa_region4_action2]

# Noise
w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = UncertainPWAAdditiveNoiseDynamics(2, [upwa_action1, upwa_action2], w)

X = Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0])
a = 2

Y = concretize(nominal(dyn, X, a))
Y_expected = concretize(ConvexHull(
    AffineMap([1.0 0.1; -0.2 1.1], X, [0.0, 0.5]),
    AffineMap([1.0 0.1; 0.0 1.1], X, [0.0, 0.5])
))
@test isequivalent(
    Y,
    Y_expected,
)