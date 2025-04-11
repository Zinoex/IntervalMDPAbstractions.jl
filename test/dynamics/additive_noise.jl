using Revise, Test
using IntervalMDPAbstractions, LazySets, LinearAlgebra

@testset "diagonally gaussian" begin
    w_stddev = [-0.1, 0.1]
    @test_throws ArgumentError AdditiveDiagonalGaussianNoise(w_stddev)

    w_stddev = [0.5, 0.7]
    w = AdditiveDiagonalGaussianNoise(w_stddev)

    @test IntervalMDPAbstractions.dim(w) == 2
    @test IntervalMDPAbstractions.stddev(w) == w_stddev
    @test IntervalMDPAbstractions.decouplingmode(w) isa IntervalMDPAbstractions.DirectDecoupling

    @testset "transition_prob_bounds" begin
        Y = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
        Z = Hyperrectangle(low = [1.0, 0.0], high = [2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 1)
        @test pl ≈ 0.02271846070634608727902886
        @test pu ≈ 0.47724986805182079279971736

        pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 2)
        @test pl ≈ 0.42343627449016523494358104
        @test pu ≈ 0.52494947594604699843798632

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.00961982036364639498159342
        @test pu ≈ 0.25053206812912340314964549
    end

    @testset "transition_prob_bounds not hyperrectangular" begin
        Y = VPolytope([[0.5, 0.0], [1.0, 0.7], [0.0, 1.0]])
        Z = Hyperrectangle(low = [1.0, 0.0], high = [2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.00961982036364639498159342
        @test pu ≈ 0.25053206812912340314964549
    end
end

@testset "degenerate gaussian" begin
    w_stddev = [0.0]
    w = AdditiveDiagonalGaussianNoise(w_stddev)
    Z = Hyperrectangle(low = [0.0], high = [1.0])

    Y = Hyperrectangle(low = [0.1], high = [0.5])
    pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 1)
    @test pl ≈ 1.0
    @test pu ≈ 1.0

    Y = Hyperrectangle(low = [-0.5], high = [-0.1])
    pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 1)
    @test pl ≈ 0.0
    @test pu ≈ 0.0

    Y = Hyperrectangle(low = [-0.5], high = [0.5])
    pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 1)
    @test pl ≈ 0.0
    @test pu ≈ 1.0
end

@testset "non-diagonal gaussian" begin
    Bw = [
        0.0                0.0                1.93809268727258e-05 0.0                  0.00194098374232017 0.0
        0.0                0.0                0.0                  5.77690735355876e-07 0.0                 0.00149272446097720
        0.0458702172680378 0.0                0.0                  0.0                  0.0                 0.0
        0.0424838359439877 0.0                0.0                  0.0                  0.0                 0.0
        0.0                0.0397464228657784 0.0                  0.0                  0.0                 0.0
        0.0                0.0377149901840099 0.0                  0.0                  0.0                 0.0
        0.0                0.0                0.0                  0.0                  0.0                 0.0
    ]
    Σ = Diagonal([1.0, 1.0, 100.0, 100.0, 5.0, 5.0])
    full_Σ = Bw * Σ * Bw'

    w = AdditiveGaussianNoise(full_Σ)

    @test IntervalMDPAbstractions.dim(w) == 7
    @test IntervalMDPAbstractions.decouplingmode(w) isa IntervalMDPAbstractions.LinearTransformationRequired
    transformation, decoupled_w = IntervalMDPAbstractions.decouple(w)
    @test decoupled_w isa IntervalMDPAbstractions.AdditiveDiagonalGaussianNoise
    @test IntervalMDPAbstractions.dim(decoupled_w) == 7
    @test IntervalMDPAbstractions.stddev(decoupled_w) ≈ sqrt.([
        0.0,
        1.7347234759768075e-18,
        3.469446951953614e-18,
        1.1141164954656932e-5,
        1.8874651472400656e-5,
        0.003002198615205235,
        0.003908953148732654
    ])

    @test transformation isa IntervalMDPAbstractions.LinearTransformation
    @test transformation.T == transformation.Tinv'
    @test transformation.Tinv ≈ [
        0.0   0.0                  0.0                  0.0  1.0  0.0                 0.0
        0.0   0.0                  0.0                  1.0  0.0  0.0                 0.0
        0.0   0.0                  0.6795063045187845   0.0  0.0  0.0                 0.7336696682562424
        0.0   0.0                  -0.7336696682562424  0.0  0.0  0.0                 0.6795063045187845
        0.0   -0.6883261814564766  0.0                  0.0  0.0  0.7254013150812078  0.0
        0.0   0.7254013150812078   0.0                  0.0  0.0  0.6883261814564765  0.0
        1.0   0.0                  0.0                  0.0  0.0  0.0                 0.0
    ]
end

@testset "centrally uniform" begin
    r = [-0.1, 0.1]
    @test_throws ArgumentError AdditiveCentralUniformNoise(r)

    r = [0.5, 0.7]
    w = AdditiveCentralUniformNoise(r)

    @test IntervalMDPAbstractions.dim(w) == 2
    @test IntervalMDPAbstractions.decouplingmode(w) isa IntervalMDPAbstractions.DirectDecoupling

    @testset "transition_prob_bounds" begin
        Y = Hyperrectangle(low = [0.0, 0.0], high = [1.0, 1.0])
        Z = Hyperrectangle(low = [1.0, 0.0], high = [2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 1)
        @test pl ≈ 0.0
        @test pu ≈ 0.5

        pl, pu = IntervalMDPAbstractions.axis_transition_prob_bounds(Y, Z, w, 2)
        @test pl ≈ 0.5
        @test pu ≈ 1.0 / 1.4

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.0
        @test pu ≈ 0.5 / 1.4
    end

    @testset "transition_prob_bounds not hyperrectangular" begin
        Y = VPolytope([[0.5, 0.0], [1.0, 0.7], [0.0, 1.0]])
        Z = Hyperrectangle(low = [1.0, 0.0], high = [2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.0
        @test pu ≈ 0.5 / 1.4
    end
end
