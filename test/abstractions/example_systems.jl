using IntervalSySCoRe, LazySets

if !@isdefined example_systems_included
    example_systems_included = true

    function simple_1d_sys()

        A = [0.95][:, :]
        B = [0.0][:, :]
        w_stddev = [0.05]

        dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

        initial_region = EmptySet(1)
        reach_region = Hyperrectangle(; low = [-0.5], high = [0.5])
        avoid_region = EmptySet(1)

        sys = System(dyn, initial_region, reach_region, avoid_region)

        return sys
    end

    function modified_running_example_sys()
        A = 0.9I(2)
        B = 0.7I(2)
        w_stddev = [1.0, 1.0]

        dyn = AffineAdditiveNoiseDynamics(A, B, AdditiveDiagonalGaussianNoise(w_stddev))

        initial_region = EmptySet(2)
        reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])
        avoid_region = EmptySet(2)

        sys = System(dyn, initial_region, reach_region, avoid_region)

        return sys
    end
end
