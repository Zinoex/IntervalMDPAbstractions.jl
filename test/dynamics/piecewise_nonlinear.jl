using Revise, Test
using IntervalMDPAbstractions, LazySets

τ = 0.1

region1 = Hyperrectangle(low = [-1.0, -1.0], high = [0.0, 1.0])
f(x, u) = [x[1] + x[2] * τ, x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * τ]
dyn_reg1 = NonlinearDynamicsRegion(f, region1)

region2 = Hyperrectangle(low = [0.0, -1.0], high = [1.0, 1.0])
g(x, u) = [x[1] + x[2] * τ, x[2] + (-x[2] + (1 - x[2])^2 * x[1]) * τ]
dyn_reg2 = NonlinearDynamicsRegion(g, region2)

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = PiecewiseNonlinearAdditiveNoiseDynamics([dyn_reg1, dyn_reg2], 2, 0, w)

prepare_nominal(dyn, InputDiscrete([0.0]))

X = Hyperrectangle(low = [-1.0, 0.0], high = [1.0, 1.0])
U = Singleton([0.0])

Y = concretize(nominal(dyn, X, U))

# First piecewise region
X1 = Hyperrectangle(low = [-1.0, 0.0], high = [0.0, 1.0])

# 1st-order Taylor expansion at [center(X1)] = [-0.5; 0.5]:  
# z = x - center(X1)
# y₁ = z₁ + 0.1 * z₂ - 0.45
# y₂ = -0.25 * z₁ + 1.2245 * z₂ + 0.6625 + [-0.075, 0.1]
X1centered = Translation(X1, -[-0.5, 0.5])
AXD1 = AffineMap([1.0 0.1; -0.25 1.225], X1centered, [-0.45, 0.6625])
Y1_expected = MinkowskiSum(AXD1, Hyperrectangle(low = [0.0, -0.075], high = [0.0, 0.1]))

# First piecewise region
X2 = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])

# 1st-order Taylor expansion at [center(X2)] = [0.5; 0.5]:
# z = x - center(X2)
# y₁ = z₁ + 0.1 * z₂ + 0.55
# y₂ = 0.025 * z₁ + 0.85 * z₂ + 0.4625 + [-0.025, 0.05]
X2centered = Translation(X2, -[0.5, 0.5])
AXD2 = AffineMap([1.0 0.1; 0.025 0.85], X2centered, [0.55, 0.4625])
Y2_expected = MinkowskiSum(AXD2, Hyperrectangle(low = [0.0, -0.025], high = [0.0, 0.05]))

Y_expected = concretize(ConvexHull(Y1_expected, Y2_expected))
@test isequivalent(Y, Y_expected)


#### Region-based actions
region2 = Hyperrectangle(low = [0.0, -1.0], high = [1.0, 1.0])
h(x, u) = [x[1] + x[2] * τ, x[2] + (-x[2] + (1 - x[2])^2 * x[1] + u[1]) * τ]
dyn_reg2 = NonlinearDynamicsRegion(h, region2)

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = PiecewiseNonlinearAdditiveNoiseDynamics([dyn_reg1, dyn_reg2], 2, 1, w)

input_abs = InputGridSplit(Hyperrectangle(low = [-1.0], high = [1.0]), (3,))
prepare_nominal(dyn, input_abs)

X = Hyperrectangle(low = [-1.0, 0.0], high = [1.0, 1.0])
U = Hyperrectangle(low = [-1.0], high = [-1 / 3])

Y = concretize(nominal(dyn, X, U))

# First piecewise region
X1 = Hyperrectangle(low = [-1.0, 0.0], high = [0.0, 1.0])

# 1st-order Taylor expansion at [center(X1)] = [-0.5; 0.5]:  
# z = x - center(X1)
# y₁ = z₁ + 0.1 * z₂
# y₂ = -0.1 * z₁ + 1.1 * z₂ + 0.6625 + [-0.075, 0.1]
X1centered = Translation(X1, -[-0.5, 0.5])
AXD1 = LinearMap([1.0 0.1; -0.1 1.1], X1centered)
Y1_expected =
    MinkowskiSum(AXD1, Hyperrectangle(; low = [0.0, -0.0625], high = [0.0, 0.0625]))

# First piecewise region
X2 = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])

# 1st-order Taylor expansion at [center(X2)] = [0.5; 0.5]:
# z = x - center(X2)
# y₁ = z₁ + 0.1 * z₂
# y₂ = 0.1 * z₁ + 0.9 * z₂ + 0.1 * u₁ - 1/15 + [-0.0625, 0.0625]
X2centered = Translation(X2, -[0.5, 0.5])
AXD2 = AffineMap([1.0 0.1; 0.1 0.9], X2centered, [0.0, -1 / 15])
U = Hyperrectangle(low = [-1.0], high = [-1 / 3])
Ucentered = Translation(U, -center(U))
AXDBU2 = MinkowskiSum(AXD2, LinearMap([0.0; 0.1], Ucentered))
Y2_expected =
    MinkowskiSum(AXDBU2, Hyperrectangle(low = [0.0, -0.0625], high = [0.0, 0.0625]))

Y_expected = concretize(ConvexHull(Y1_expected, Y2_expected))
@test isequivalent(Y, Y_expected)

# Vector inputs
x = [0.5, 0.5]
u = [0.0]

y = nominal(dyn, x, u)
y_expected = g(x, u)
@test y ≈ y_expected
