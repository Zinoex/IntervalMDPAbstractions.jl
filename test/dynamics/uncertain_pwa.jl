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

upwa_action1 =
    [upwa_region1_action1, upwa_region2_action1, upwa_region3_action1, upwa_region4_action1]

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

upwa_action2 =
    [upwa_region1_action2, upwa_region2_action2, upwa_region3_action2, upwa_region4_action2]

# Noise
w_variance = [0.2, 0.2]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = UncertainPWAAdditiveNoiseDynamics(2, [upwa_action1, upwa_action2], w)

@test noise(dyn) == w
@test dimstate(dyn) == 2
@test diminput(dyn) == 1

# Nominal dynamics
@testset "nominal_dynamics" begin
    X = Hyperrectangle(low = [0.0, -0.5], high = [0.5, 0.0])
    a = 2

    Y = concretize(nominal(dyn, X, a))
    Y_expected = concretize(
        ConvexHull(
            AffineMap([1.0 0.1; -0.2 1.1], X, [0.0, 0.5]),
            AffineMap([1.0 0.1; 0.0 1.1], X, [0.0, 0.5]),
        ),
    )
    @test isequivalent(Y, Y_expected)
end

# TODO: Test transform
@testset "linear_transformation" begin
    Tx = IntervalMDPAbstractions.LinearTransformation(
        [
            0.5 2.0;
            0.0 2.0
        ],
        [
            1.0 0.8;
            0.0 0.6
        ],
    )

    dyn_transformed = IntervalMDPAbstractions.transform(dyn, Tx, w)
    X = Hyperrectangle(low = [-0.5, -0.5], high = [0.5, 0.0])
    a = 2
    Y = nominal(dyn_transformed, X, a)
    Y1 = concretize(Y.X)
    Y2 = concretize(Y.Y)

    display(vertices_list(Y1))
    display(vertices_list(Y2))

    region_1 = Zonotope(
        [-0.625, -0.5],
        [0.125 0.5; 0.0 0.5],
    )
    Y1_expected = concretize(ConvexHull(
        AffineMap(Tx.T * [1.0  0.1; -0.1  1.1] * Tx.Tinv, Intersection(region_1, X), Tx.T * [0.0, 0.5]),
        AffineMap(Tx.T * [1.0 0.1; 0.0 1.1] * Tx.Tinv, Intersection(region_1, X), Tx.T * [0.0, 0.5]),
    ))
    region_2 = Zonotope(
        [-0.375, -0.5],
        [0.125 0.5; 0.0 0.5],
    )
    Y2_expected = concretize(ConvexHull(
        AffineMap(Tx.T * [1.0 0.1; -0.2 1.1] * Tx.Tinv, Intersection(region_2, X), Tx.T * [0.0, 0.5]),
        AffineMap(Tx.T * [1.0 0.1; 0.0 1.1] * Tx.Tinv, Intersection(region_2, X), Tx.T * [0.0, 0.5]),
    ))

    display(vertices_list(Y1_expected))
    display(vertices_list(Y2_expected))

    Y1_equiv = isequivalent(Y1, Y1_expected)
    if Y1_equiv
        @test Y1_equiv
        @test isequivalent(Y2, Y2_expected)
    else    
        @test isequivalent(Y2, Y1_expected)
        @test isequivalent(Y1, Y2_expected)
    end
end

# Vector states
@testset "vector_input" begin
    X = [-0.25, 0.25]
    a = 1

    Y = concretize(nominal(dyn, X, a))
    Y_expected = VPolytope([
        [1.0 0.1; -0.3 1.1] * X + [0.0, 0.5],
        [1.0 0.1; 0.3 1.1] * X + [0.0, 0.5],
    ])
    @test isequivalent(Y, Y_expected)
end
