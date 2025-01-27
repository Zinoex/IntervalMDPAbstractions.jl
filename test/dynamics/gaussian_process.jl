using Revise, Test
using IntervalMDPAbstractions, LazySets

# System definition

# Action 1
gp_region1_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [-0.5, -0.5], high = [0.0, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region2_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region3_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [-0.5, 0.0], high = [0.0, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region4_action1 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [0.0, 0.0], high = [0.5, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)

gp_action1 = [gp_region1_action1, gp_region2_action1, gp_region3_action1, gp_region4_action1]

# Action 2
gp_region1_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [-0.5, -0.5], high = [0.0, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region2_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region3_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [-0.5, 0.0], high = [0.0, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)
gp_region4_action2 = AbstractedGaussianProcessRegion(
    Hyperrectangle(low = [0.0, 0.0], high = [0.5, 0.5]),
    [-0.5, 0.5],
    [0.0, 0.6],
    [0.1, 0.3],
    [0.2, 0.4],
)

gp_action2 = [gp_region1_action2, gp_region2_action2, gp_region3_action2, gp_region4_action2]

# Noise
w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = AbstractedGaussianProcess([gp_action1, gp_action2])
@test dimstate(sys) == 2
@test diminput(sys) == 1


# Find mean and variance bounds
X = Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0])
a = 1

bounds = gp_bounds(dyn, X, a)
@test bounds == gp_region2_action1

@testset "transition_prob_bounds" begin
    Z = Hyperrectangle(low = [-0.5, 0.0], high = [0.0, 0.4])

    pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(bounds, Z, 1)
    @test pl ≈ 0.49379033467422386483302189
    @test pu ≈ 0.98758066934844772966604379

    pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(bounds, Z, 2)
    @test pl ≈ 0.22974240559874370586377918
    @test pu ≈ 0.32165098790894893041383367

    pl, pu = IntervalMDPAbstractions.transition_prob_bounds(bounds, Z)
    @test pl ≈ 0.11344457934946493711769394
    @test pu ≈ 0.31765629793570925226664165
end