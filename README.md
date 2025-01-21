# IntervalMDPAbstractions - Abstraction-based verification and synthesis of stochastic systems via IMDPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Zinoex.github.io/IntervalMDPAbstractions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Zinoex.github.io/IntervalMDPAbstractions.jl/dev/)
[![Build Status](https://github.com/Zinoex/IntervalMDPAbstractions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Zinoex/IntervalMDPAbstractions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Zinoex/IntervalMDPAbstractions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Zinoex/IntervalMDPAbstractions.jl)

IntervalMDPAbstractions.jl is a Julia package for verifying stochastic systems and synthesizing correct-by-construction controllers via abstraction-based techniques. It provides a set of abstraction methods for different types of stochastic systems (linear, non-linear uncertain piece-wise affine) to Interval Markov Decision Processes (IMDPs), orthogonally decoupled IMDPs (odIMDPs), and mixtures of odIMDPs.

The package is built on top of IntervalMDP.jl, which provides a set of structs to model the IMDP, odIMDP, etc. along with methods for verifying and synthesizing controllers using value iteration.

## Installation

This package requires Julia v1.10 or later. Refer to the official documentation on how to install it for your system.

To install IntervalMDPAbstractions.jl, use the following command inside Julia's REPL:

```julia
julia> import Pkg; Pkg.add("https://github.com/Zinoex/IntervalMDP.jl")
```

## Usage

TODO

## Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the program “IntervalMDPAbstraction.jl” (Abstraction-based verification and synthesis of stochastic systems via IMDPs) written by the Frederik Baymler Mathiesen. Fred van Keulen, Dean of Mechanical Engineering.

© 2025, Frederik Baymler Mathiesen, Delft Center for Systems and Control, TU Delft