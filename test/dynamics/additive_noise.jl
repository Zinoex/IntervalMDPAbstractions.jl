using Revise, Test
using IntervalMDPAbstractions, LazySets

@testset "gaussian" begin
    w_stddev = [-0.1, 0.1]
    @test_throws ArgumentError AdditiveDiagonalGaussianNoise(w_stddev)

    w_stddev = [0.5, 0.7]
    w = AdditiveDiagonalGaussianNoise(w_stddev)

    @test IntervalMDPAbstractions.dim(w) == 2
    @test IntervalMDPAbstractions.stddev(w) == w_stddev
    @test IntervalMDPAbstractions.candecouple(w) == true

    @testset "transition_prob_bounds" begin
        Y = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])
        Z = Hyperrectangle(low=[1.0, 0.0], high=[2.0, 1.0])

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
        Z = Hyperrectangle(low=[1.0, 0.0], high=[2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.00961982036364639498159342
        @test pu ≈ 0.25053206812912340314964549
    end
end

@testset "centrally uniform" begin
    r = [-0.1, 0.1]
    @test_throws ArgumentError AdditiveCentralUniformNoise(r)

    r = [0.5, 0.7]
    w = AdditiveCentralUniformNoise(r)

    @test IntervalMDPAbstractions.dim(w) == 2
    @test IntervalMDPAbstractions.candecouple(w) == true

    @testset "transition_prob_bounds" begin
        Y = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])
        Z = Hyperrectangle(low=[1.0, 0.0], high=[2.0, 1.0])

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
        Z = Hyperrectangle(low=[1.0, 0.0], high=[2.0, 1.0])

        pl, pu = IntervalMDPAbstractions.transition_prob_bounds(Y, Z, w)
        @test pl ≈ 0.0
        @test pu ≈ 0.5 / 1.4
    end
end