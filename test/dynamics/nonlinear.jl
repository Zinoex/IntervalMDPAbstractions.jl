using Revise, Test
using IntervalMDPAbstractions, LazySets

# System definition
sampling_time = 0.1
f_nonlinear(x, u) = [
    x[1] + x[2] * sampling_time,
    x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * sampling_time + u[1],
]

w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = NonlinearAdditiveNoiseDynamics(f_nonlinear, 2, 1, w)
initial_region = Hyperrectangle(low = [-1.0, -1.0], high = [1.0, 1.0])
sys = System(dyn, initial_region)

@test noise(dyn) == w
@test dimstate(sys) == 2
@test diminput(sys) == 1

# Hyperrectangular control
X = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
U = Hyperrectangle(low = [0.0], high = [1.0])
U_abs = InputDiscrete([U])

# 1st-order Taylor expansion at [center(X); center(U)] = [0.5; 0.5; 0.5]: y = f(c) + J(c) * (x - c)
# y₁ = 0.55 + (x₁ - 0.5) + 0.1 * (x₂ - 0.5)
# y₂ = 0.9625 - 1.5 * τ * (x₁ - 0.5) + 1.025 * (x₂ - 0.5) + (u₁ - 0.5) + [-0.025, 0.05]

IntervalMDPAbstractions.prepare_nominal(dyn, U_abs)
Y = concretize(nominal(dyn, X, U))
AXD = AffineMap([1.0 0.1; -0.15 1.025], Translation(X, [-0.5, -0.5]), [0.55, 0.9625])
BU = LinearMap([0.0, 1.0][:], Translation(U, [-0.5]))
AXBUD = MinkowskiSum(AXD, BU)
Y_expected =
    concretize(MinkowskiSum(AXBUD, Hyperrectangle(low = [0.0, -0.025], high = [0.0, 0.05])))
@test isequivalent(Y, Y_expected)

# Singleton control
X = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
U = Singleton([2.0])
U_abs = InputDiscrete([U])

# 1st-order Taylor expansion at [center(X)] = [0.5; 0.5]:
# y₁ = 0.55 + (x₁ - 0.5) + 0.1 * (x₂ - 0.5)
# y₂ = 2.4625 - 1.5 * τ * (x₁ - 0.5) + 1.025 * (x₂ - 0.5) + [-0.025, 0.05]

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


# Test transform
w = AdditiveGaussianNoise([
    0.3 0.1;
    0.1 0.2
])
dyn = NonlinearAdditiveNoiseDynamics(f_nonlinear, 2, 1, w)
sys = System(dyn, initial_region)

Tx, sys = IntervalMDPAbstractions.decouple(sys)

@test Tx.T ≈ [
    0.5257311121191336 -0.8506508083520399
    -0.8506508083520399 -0.5257311121191336
]

@test initial(sys) == concretize(Tx.T * initial_region)

@test dimstate(sys) == 2
@test diminput(sys) == 1

# Hyperrectangular regions
Z = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
X = Tx.Tinv * Z
U = Hyperrectangle(low = [0.0], high = [1.0])
U_abs = InputDiscrete([U])


# 1st-order Taylor expansion at [center(X); center(U)] = [0.5; 0.5; 0.5]: y = f(c) + J(c) * (x - c)
# y₁ = 0.1037818307842547 + 0.4746930233034408 * (z₁ - 0.5) - 0.9130272380832306 * (z₂ - 0.5) - 0.8506508083520399 * (u - 0.5) + [-0.07040294042680408, 0.09967345025805252]
# y₂ = 0.33602540390190677 - 0.8821940819609738 * (z₁ - 0.5) - 0.6818389162483736 * (z₂ - 0.5) - 0.5257311121191336 * (u₁ - 0.5) + [-0.04351141009169895, 0.06160158003544844]

IntervalMDPAbstractions.prepare_nominal(dynamics(sys), U_abs)
Y = concretize(nominal(dynamics(sys), Z, U))
AXD = AffineMap(
    [1.0262282491794423 0.07620882128041234; 0.11620994945829699 1.1089030406688507],
    Translation(Z, [-0.5, -0.5]),
    [0.1037818307842547, 0.33602540390190677],
)
BU = LinearMap([-0.8506508083520399, -0.5257311121191336][:], Translation(U, [-0.5]))
AXBUD = MinkowskiSum(AXD, BU)
Y_expected = concretize(
    MinkowskiSum(
        AXBUD,
        Hyperrectangle(
            low = [-0.07040294042680408, -0.04351141009169895],
            high = [0.09967345025805252, 0.06160158003544844],
        ),
    ),
)
@test isequivalent(Y, Y_expected)
