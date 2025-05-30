using Revise, Test
using IntervalMDPAbstractions, LazySets

τ = 0.1

region1 = Hyperrectangle(; low=[-1.0, -1.0], high=[0.0, 1.0])
f_piecewise(x, u) = [x[1] + x[2] * τ, x[2] + (-x[1] + (1 - x[1])^2 * x[2]) * τ]
dyn_reg1 = NonlinearDynamicsRegion(f_piecewise, region1)

region2 = Hyperrectangle(; low=[0.0, -1.0], high=[1.0, 1.0])
g_piecewise(x, u) = [x[1] + x[2] * τ, x[2] + (-x[2] + (1 - x[2])^2 * x[1]) * τ]
dyn_reg2 = NonlinearDynamicsRegion(g_piecewise, region2)

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = PiecewiseNonlinearAdditiveNoiseDynamics([dyn_reg1, dyn_reg2], 2, 0, w)

prepare_nominal(dyn, InputDiscrete([0.0]))

X = Hyperrectangle(; low=[-1.0, 0.0], high=[1.0, 1.0])
U = Singleton([0.0])

Y = nominal(dyn, X, U)
Y1 = concretize(Y.X)
Y2 = concretize(Y.Y)

# First piecewise region
X1 = Hyperrectangle(; low=[-1.0, 0.0], high=[0.0, 1.0])

# 1st-order Taylor expansion at [center(X1)] = [-0.5, 0.5]:  
# z = x - center(X1)
# y₁ = z₁ + 0.1 * z₂ - 0.45
# y₂ = -0.25 * z₁ + 1.2245 * z₂ + 0.6625 + [-0.075, 0.1]
X1centered = Translation(X1, -[-0.5, 0.5])
AXD1 = AffineMap([1.0 0.1; -0.25 1.225], X1centered, [-0.45, 0.6625])
Y1_expected = concretize(MinkowskiSum(AXD1, Hyperrectangle(; low=[0.0, -0.075], high=[0.0, 0.1])))

# Second piecewise region
X2 = Hyperrectangle(; low=[0.0, 0.0], high=[1.0, 1.0])

# 1st-order Taylor expansion at [center(X2)] = [0.5, 0.5]:
# z = x - center(X2)
# y₁ = z₁ + 0.1 * z₂ + 0.55
# y₂ = 0.025 * z₁ + 0.85 * z₂ + 0.4625 + [-0.025, 0.05]
X2centered = Translation(X2, -[0.5, 0.5])
AXD2 = AffineMap([1.0 0.1; 0.025 0.85], X2centered, [0.55, 0.4625])
Y2_expected = concretize(MinkowskiSum(AXD2, Hyperrectangle(; low=[0.0, -0.025], high=[0.0, 0.05])))

Y1_equiv = isequivalent(Y1, Y1_expected)
if Y1_equiv
    @test Y1_equiv
    @test isequivalent(Y2, Y2_expected)
else
    @test isequivalent(Y2, Y1_expected)
    @test isequivalent(Y1, Y2_expected)
end

#### Region-based actions
region2 = Hyperrectangle(; low=[0.0, -1.0], high=[1.0, 1.0])
h_piecewise(x, u) = [x[1] + x[2] * τ, x[2] + (-x[2] + (1 - x[2])^2 * x[1] + u[1]) * τ]
dyn_reg2 = NonlinearDynamicsRegion(h_piecewise, region2)

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = PiecewiseNonlinearAdditiveNoiseDynamics([dyn_reg1, dyn_reg2], 2, 1, w)

input_abs = InputGridSplit(Hyperrectangle(; low=[-1.0], high=[1.0]), (3,))
prepare_nominal(dyn, input_abs)

X = Hyperrectangle(; low=[-1.0, 0.0], high=[1.0, 1.0])
U = Hyperrectangle(; low=[-1.0], high=[-1 / 3])

Y = nominal(dyn, X, U)
Y1 = concretize(Y.X)
Y2 = concretize(Y.Y)

# First piecewise region
X1 = Hyperrectangle(; low=[-1.0, 0.0], high=[0.0, 1.0])

# 1st-order Taylor expansion at [center(X1); center(U)] = [-0.5, 0.5, -2/3]:  
# z = [x; u] - [center(X1); center(U)]
# y₁ = z₁ + 0.1 * z₂ - 0.45
# y₂ = -0.25 * z₁ + 1.225 * z₂ + 0.6625 + [-0.075, 0.1]
X1centered = Translation(X1, -[-0.5, 0.5])
AXD1 = AffineMap([1.0 0.1; -0.25 1.225], X1centered, [-0.45, 0.6625])
Y1_expected = concretize(MinkowskiSum(AXD1, Hyperrectangle(; low=[0.0, -0.075], high=[0.0, 0.1])))

# Second piecewise region
X2 = Hyperrectangle(; low=[0.0, 0.0], high=[1.0, 1.0])

# 1st-order Taylor expansion at [center(X2)] = [0.5, 0.5, -2/3]:
# z = [x; u] - [center(X2); center(U)]
# y₁ = z₁ + 0.1 * z₂ + 0.55
# y₂ = 0.025 * z₁ + 0.85 * z₂ + 0.1 * u₁ + (1/3 + 0.0625) + [-0.025, 0.05]
Z2centered = Translation(CartesianProduct(X2, U), -[0.5, 0.5, -2 / 3])
AXBUD2 = AffineMap([1.0 0.1 0.0; 0.025 0.85 0.1], Z2centered, [0.55, 1 / 3 + 0.0625])
Y2_expected = concretize(
    MinkowskiSum(AXBUD2, Hyperrectangle(; low=[0.0, -0.025], high=[0.0, 0.05])),
)

Y1_equiv = isequivalent(Y1, Y1_expected)
if Y1_equiv
    @test Y1_equiv
    @test isequivalent(Y2, Y2_expected)
else
    @test isequivalent(Y2, Y1_expected)
    @test isequivalent(Y1, Y2_expected)
end

# TODO: Test transform

# Vector inputs
x = [0.5, 0.5]
u = [0.0]
X = Singleton(x)
U = Singleton(u)

y = nominal(dyn, X, U)
y_expected = g_piecewise(x, u)
@test element(y) ≈ y_expected
