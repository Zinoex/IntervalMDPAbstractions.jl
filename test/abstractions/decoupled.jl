using Revise, Test
using LinearAlgebra, LazySets
using IntervalMDP, IntervalMDPAbstractions

include("example_systems.jl")

@testset "1d dense vs sparse" begin
    function simple_1d_decoupled(; sparse = false)
        sys, spec = simple_1d_sys()
    
        X = Hyperrectangle(; low = [-2.5], high = [2.5])
        state_abs = StateUniformGridSplit(X, (10,))
        input_abs = InputDiscrete([Singleton([0.0])])
    
        if sparse
            target_model = SparseOrthogonalIMDPTarget()
        else
            target_model = OrthogonalIMDPTarget()
        end
    
        prob = AbstractionProblem(sys, spec)
        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)
    
        return mdp, abstract_spec
    end

    # Dense
    mdp_dense, spec_dense = simple_1d_decoupled()
    @test num_states(mdp_dense) == 11
    @test stateptr(mdp_dense)[end] == 11

    prob_dense = Problem(mdp_dense, spec_dense)

    V_dense, k, res = value_iteration(prob_dense)
    @test k == 10

    # Sparse
    mdp_sparse, spec_sparse = simple_1d_decoupled(; sparse = true)
    @test num_states(mdp_sparse) == 11
    @test stateptr(mdp_sparse)[end] == 11

    prob_sparse = Problem(mdp_sparse, spec_sparse)

    V_sparse, k, res = value_iteration(prob_sparse)
    @test k == 10
    @test all(V_dense .≥ V_sparse)
    
    @test satisfaction_mode(spec_dense) == satisfaction_mode(spec_sparse)
    @test strategy_mode(spec_dense) == strategy_mode(spec_sparse)
    
    prop_dense = system_property(spec_dense)
    prop_sparse = system_property(spec_sparse)
    @test all(IntervalMDP.reach(prop_dense) .== IntervalMDP.reach(prop_sparse))
    @test all(IntervalMDP.avoid(prop_dense) .== IntervalMDP.avoid(prop_sparse))
end

@testset "2d" begin
    function modified_running_example_decoupled(; sparse = false, range_vs_grid = :grid)
        sys, spec = modified_running_example_sys()

        X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
        state_abs = StateUniformGridSplit(X, (10, 10))

        U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])
        if range_vs_grid == :range
            input_abs = InputLinRange(U, [3, 3])
        elseif range_vs_grid == :grid
            input_abs = InputGridSplit(U, [3, 3])
        else
            throw(ArgumentError("Invalid range_vs_grid argument"))
        end

        if sparse
            target_model = SparseOrthogonalIMDPTarget()
        else
            target_model = OrthogonalIMDPTarget()
        end

        prob = AbstractionProblem(sys, spec)
        mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

        return mdp, abstract_spec
    end

    @testset "dense vs sparse" begin
        # Dense, input grid
        mdp_dense, spec_dense = modified_running_example_decoupled()
        @test num_states(mdp_dense) == 121  # 11 * 11 total states
        @test stateptr(mdp_dense)[end] == 10 * 10 * 9 + 1  # 10 * 10 non-sink states, 9 actions

        prob_dense = Problem(mdp_dense, spec_dense)

        V_dense, k, res = value_iteration(prob_dense)
        @test k == 10

        # Sparse, input grid
        mdp_sparse, spec_sparse = modified_running_example_decoupled(; sparse = true)
        @test num_states(mdp_sparse) == 121  # 11 * 11 total states
        @test stateptr(mdp_sparse)[end] == 10 * 10 * 9 + 1  # 10 * 10 non-sink states, 9 actions

        prob_sparse = Problem(mdp_sparse, spec_sparse)

        V_sparse, k, res = value_iteration(prob_sparse)
        @test k == 10
        @test all(V_dense .≥ V_sparse)
    
        @test satisfaction_mode(spec_dense) == satisfaction_mode(spec_sparse)
        @test strategy_mode(spec_dense) == strategy_mode(spec_sparse)
        
        prop_dense = system_property(spec_dense)
        prop_sparse = system_property(spec_sparse)
        @test all(IntervalMDP.reach(prop_dense) .== IntervalMDP.reach(prop_sparse))
        @test all(IntervalMDP.avoid(prop_dense) .== IntervalMDP.avoid(prop_sparse))
    end

    @testset "dense grid vs range" begin
        # Dense, input grid
        mdp_grid, spec_grid = modified_running_example_decoupled(; range_vs_grid = :grid)

        prob_grid = Problem(mdp_grid, spec_grid)
        V_grid, k, res = value_iteration(prob_grid)

        # Dense, input range
        mdp_range, spec_range = modified_running_example_decoupled(; range_vs_grid = :range)
        @test num_states(mdp_range) == 121  # 11 * 11 total states
        @test stateptr(mdp_range)[end] == 10 * 10 * 9 + 1  # 10 * 10 non-sink states, 9 actions

        prob_range = Problem(mdp_range, spec_range)
        V_range, k, res = value_iteration(prob_range)

        @test k == 10
        @test all(V_range .≥ V_grid)
    
        @test satisfaction_mode(spec_grid) == satisfaction_mode(spec_range)
        @test strategy_mode(spec_grid) == strategy_mode(spec_range)
        
        prop_grid = system_property(spec_grid)
        prop_range = system_property(spec_range)
        @test all(IntervalMDP.reach(prop_grid) .== IntervalMDP.reach(prop_range))
        @test all(IntervalMDP.avoid(prop_grid) .== IntervalMDP.avoid(prop_range))
    end
end