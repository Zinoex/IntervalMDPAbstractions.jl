A1 = [
    0.1 0.9
    0.8 0.2
]
B1 = [
    0.0
    0.0
][:, :]
w1_stddev = [0.3, 0.2]
mode1 = AffineAdditiveNoiseDynamics(A1, B1, AdditiveDiagonalGaussianNoise(w1_stddev))

A2 = [
    0.8 0.2
    0.1 0.9
]
B2 = [
    0.0
    0.0
][:, :]
w2_stddev = [0.2, 0.1]
mode2 = AffineAdditiveNoiseDynamics(A2, B2, AdditiveDiagonalGaussianNoise(w2_stddev))

dyn = StochasticSwitchedDynamics([mode1, mode2], [0.7, 0.3])
initial_region = EmptySet(2)
sys = System(dyn, initial_region)

@test dimstate(sys) == 2
@test diminput(sys) == 1