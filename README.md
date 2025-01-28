# IntervalMDPAbstractions.jl - Abstraction-based verification and synthesis of stochastic systems via IMDPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Zinoex.github.io/IntervalMDPAbstractions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Zinoex.github.io/IntervalMDPAbstractions.jl/dev/)
[![Build Status](https://github.com/Zinoex/IntervalMDPAbstractions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Zinoex/IntervalMDPAbstractions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Zinoex/IntervalMDPAbstractions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Zinoex/IntervalMDPAbstractions.jl)

[IntervalMDPAbstractions.jl](https://github.com/Zinoex/IntervalMDPAbstractions.jl) is [Julia](https://julialang.org/) package for verifying stochastic systems and synthesizing correct-by-construction controllers via abstraction-based techniques with IMDP-inspired target models. It provides a set of abstraction methods for different types of stochastic systems (linear, non-linear, uncertain piece-wise affine, Gaussian processes, and stochastically switched systems) to Interval Markov Decision Processes (IMDPs), orthogonally decoupled IMDPs (odIMDPs), and mixtures of odIMDPs. We refer to [1, 2] for more details on the target models and abstraction process.

The package is built on top of [IntervalMDP.jl](https://github.com/Zinoex/IntervalMDP.jl), which provides a set of structs to model the IMDP, odIMDP, etc. along with methods for verifying and synthesizing controllers using value iteration.

## Installation

This package requires Julia v1.10 or later. Refer to the [official documentation](https://julialang.org/downloads/) on how to install it for your system.

To install IntervalMDPAbstractions.jl, use the following command inside Julia's REPL:

```julia
julia> import Pkg; Pkg.add(url="https://github.com/Zinoex/IntervalMDPAbstractions.jl")
```

## Usage

We will here go through the example of a 4D building automation system (for more examples see the `examples/systems` directory). First, we define the system and specification:

```julia
A = [
    0.6682 0.0 0.02632 0.0
    0.0 0.683 0.0 0.02096
    1.0005 0.0 -0.000499 0.0
    0.0 0.8004 0.0 0.1996
]

B = [
    0.1320
    0.1402
    0.0
    0.0
][:, :]

C = [3.4378, 2.9272, 13.0207, 10.4166]

w_variance = [1 / 12.9199, 1 / 12.9199, 1 / 2.5826, 1 / 3.2276]
w_stddev = sqrt.(w_variance)
w = AdditiveDiagonalGaussianNoise(w_stddev)

dyn = AffineAdditiveNoiseDynamics(A, B, C, w)

initial_region = EmptySet(4)

sys = System(dyn, initial_region)

time_horizon = 10
avoid_region = EmptySet(4)
prop = FiniteTimeRegionSafety(avoid_region, time_horizon)
spec = Specification(prop, Pessimistic, Maximize)

abstraction_problem = AbstractionProblem(sys, spec)
```
Note that the avoid region is defined as an empty set, which means that the resulting specification is the probability of staying within the region of interest.

Next, we define the gridding of the region of interest, the input space, and the target model:
```julia
X = Hyperrectangle(;  # X is the region of interest
    low = [18.75, 18.75, 29.5, 29.5],
    high = [21.25, 21.25, 36.5, 36.5],
)
state_abs = StateUniformGridSplit(X, state_split)

U = Hyperrectangle(; low = [17.0], high = [20.0])
input_abs = InputLinRange(U, input_split)

target_model = OrthogonalIMDPTarget()  # In this case, we use odIMDPs as the target model
```

Finally, we can build the abstraction:
```julia
mdp, abstract_spec = abstraction(prob, state_abs, input_abs, target_model)

upper_bound_spec = Specification(system_property(spec), !satisfaction_mode(spec))
upper_bound_spec = IntervalMDPAbstractions.convert_specification(upper_bound_spec, state_abs, target_model)
```
Note that we separately generate the specification to verify the upper bound of the satisfaction probability. This is due to the theoretical quirk that, if reachability or avoid regions do not align with the gridding, then using the lower bound specification (reach and avoid states in the abstract model) to compute an upper bound may yield an unsound upper bound. See [2, Section 7] for more information on the theoretical background.

We can then use this model to verify the satisfaction probability of the specification:
```julia
problem = Problem(mdp, abstract_spec)
strategy, value_function_lower, iterations, residual = control_synthesis(problem)

problem = Problem(mdp, upper_bound_spec, strategy)
value_function_upper, iterations, residual = value_iteration(problem)
```

## Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the program “IntervalMDPAbstraction.jl” (Abstraction-based verification and synthesis of stochastic systems via IMDPs) written by the Frederik Baymler Mathiesen. Fred van Keulen, Dean of Mechanical Engineering.

© 2025, Frederik Baymler Mathiesen, Delft Center for Systems and Control, TU Delft

[1] Mathiesen, Frederik Baymler, Sofie Haesaert, and Luca Laurenti. "Scalable control synthesis for stochastic systems via structural IMDP abstractions." arXiv preprint arXiv:2411.11803 (2024).

[2] Cauchi, Nathalie, et al. "Efficiency through uncertainty: Scalable formal synthesis for stochastic hybrid systems." Proceedings of the 22nd ACM international conference on hybrid systems: computation and control. 2019.