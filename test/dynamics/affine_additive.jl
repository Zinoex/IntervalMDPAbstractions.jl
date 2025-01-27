using Revise, Test
using IntervalMDPAbstractions, LazySets

# System definition
A = [1.0 0.5; 0.0 1.0]
B = reshape([0.0; 1.0], (2, 1))

w_stddev = [0.1, 0.1]
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = AffineAdditiveNoiseDynamics(A, B, w)
initial_region = Hyperrectangle(low = [-1.0, -1.0], high = [1.0, 1.0])
sys = System(dyn, initial_region)

@test noise(dyn) == w
@test dimstate(sys) == 2
@test diminput(sys) == 1

# Hyperrectangular regions
X = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
U = Hyperrectangle(low = [0.0], high = [1.0])

Y = concretize(nominal(dyn, X, U))
@test isequivalent(
    Y,
    VPolytope([[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [1.5, 2.0], [0.5, 2.0], [0.0, 1.0]]),
)

# Singleton regions
X = Singleton([1.0, 1.0])
U = Singleton([2.0])

Y = concretize(nominal(dyn, X, U))
@test isequivalent(Y, Singleton([1.5, 3.0]))
