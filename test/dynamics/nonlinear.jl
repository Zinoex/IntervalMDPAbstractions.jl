using Revise, Test
using IntervalMDPAbstractions, LazySets

# System definition
sampling_time = 0.1
f(x, u) = [
    x[1] + x[2] * sampling_time,
    x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * sampling_time + u[1],
]

w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = NonlinearAdditiveNoiseDynamics(f, 2, 1, w)
initial_region = Hyperrectangle(low = [-1.0, -1.0], high = [1.0, 1.0])
sys = System(dyn, initial_region)

@test noise(dyn) == w
@test dimstate(sys) == 2
@test diminput(sys) == 1

# Hyperrectangular control
X = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
U = Hyperrectangle(low = [0.0], high = [1.0])
U_abs = InputDiscrete([U])

# 1st-order Taylor expansion at [center(X); center(U)] = [0.5; 0.5; 0.5]:
# y₁ = (x₁ - 0.5) + 0.1 * (x₂ - 0.5)
# y₂ = -0.1 * (x₁ - 0.5) + 1.1 * (x₂ - 0.5) + (u₁ - 0.5) + 0.5 ± 0.0625

IntervalMDPAbstractions.prepare_nominal(dyn, U_abs)
Y = concretize(nominal(dyn, X, U))
AXD = AffineMap([1.0 0.1; -0.1 1.1], Translation(X, [-0.5, -0.5]), [0.0, 0.5])
BU = LinearMap([0.0, 1.0][:], Translation(U, [-0.5]))
AXBUD = MinkowskiSum(AXD, BU)
Y_expected = concretize(
    MinkowskiSum(AXBUD, Hyperrectangle(low = [0.0, -0.0625], high = [0.0, 0.0625])),
)
@test isequivalent(Y, Y_expected)

# Singleton control
X = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
U = Singleton([2.0])
U_abs = InputDiscrete([U])

# 1st-order Taylor expansion at [center(X)] = [0.5; 0.5]:
# y₁ = (x₁ - 0.5) + 0.1 * (x₂ - 0.5) + 0.5
# y₂ = -0.15 * (x₁ - 0.5) + 1.025 * (x₂ - 0.5) + 0.5 ± 0.0625

IntervalMDPAbstractions.prepare_nominal(dyn, U_abs)
Y = concretize(nominal(dyn, X, U))
AXD = AffineMap([1.0 0.1; -0.15 1.025], Translation(X, [-0.5, -0.5]), [0.55, 2.4625])
Y_expected =
    concretize(MinkowskiSum(AXD, Hyperrectangle(low = [0.0, -0.025], high = [0.0, 0.05])))
@test isequivalent(Y, Y_expected)

# Vector inputs
X = [0.5, 0.5]
U = [2.0]

Y = nominal(dyn, X, U)
Y_expected = [
    X[1] + X[2] * sampling_time,
    X[2] + (-X[1] + (1 - X[1])^2 * X[2]) * sampling_time + U[1],
]
@test Y ≈ Y_expected
