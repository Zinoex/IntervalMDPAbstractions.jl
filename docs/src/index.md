```@meta
CurrentModule = IntervalMDPAbstractions
```

# IntervalMDPAbstractions

[IntervalMDPAbstractions.jl](https://github.com/Zinoex/IntervalMDPAbstractions.jl) is [Julia](https://julialang.org/) package for verifying stochastic systems and synthesizing correct-by-construction controllers via abstraction-based techniques with IMDP-inspired target models. It provides a set of abstraction methods for different types of stochastic systems (linear, non-linear, uncertain piece-wise affine, Gaussian processes, and stochastically switched systems) to Interval Markov Decision Processes (IMDPs), orthogonally decoupled IMDPs (odIMDPs), and mixtures of odIMDPs. We refer to [1, 2] for more details on the target models and abstraction process.

The package is built on top of [IntervalMDP.jl](https://github.com/Zinoex/IntervalMDP.jl), which provides a set of structs to model the IMDP, odIMDP, etc. along with methods for verifying and synthesizing controllers using value iteration.

This package is work-in-progress and the API is subject to change, as we discover what the most intuitive and fast interface is. The package is not yet registered in the Julia package registry, but can be installed directly from the GitHub repository. 

## Installation
This package requires Julia v1.10 or later. Refer to the [official documentation](https://julialang.org/downloads/) on how to install it for your system.

To install IntervalMDPAbstractions.jl, use the following command inside Julia's REPL:

```julia
julia> import Pkg; Pkg.add(url="https://github.com/Zinoex/IntervalMDPAbstractions.jl")
```

[1] Mathiesen, Frederik Baymler, Sofie Haesaert, and Luca Laurenti. "Scalable control synthesis for stochastic systems via structural IMDP abstractions." arXiv preprint arXiv:2411.11803 (2024).

[2] Cauchi, Nathalie, et al. "Efficiency through uncertainty: Scalable formal synthesis for stochastic hybrid systems." Proceedings of the 22nd ACM international conference on hybrid systems: computation and control. 2019.