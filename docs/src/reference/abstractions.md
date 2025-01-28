# Abstractions
To build an abstraction, in addition to the definition of the system dynamics and the specification, 
we also need to define select inputs and partition the state space/region of interest.
Since there are multiple options for both, we wrap them as input and state abstractions, such that
the [`abstraction`](@ref) method to build the finite-state abstractions can ignore the details of
the choice of input and state partitioning.

## Input abstractions
```@docs
InputAbstraction
InputGridSplit
InputLinRange
InputDiscrete
inputs
numinputs
```

To define a new input abstraction, introduce a new struct type that inherits from `InputAbstraction` and
implements `inputs` and `numinputs` methods. The `inputs` method should return the set of inputs (set-based, singletons, or discrete)
and the `numinputs` should return the number of inputs in the set.

## State abstractions

```@docs
StateAbstraction
StateUniformGridSplit
regions
numregions
statespace
```

Right now, we only support state abstractions that are based on a uniform grid split of the state space. However, one
can easily imagine a non-uniform grid or a refinement-based partitioning for heterogenous abstractions. To implement such a state abstraction,
define a new struct type that inherits from `StateAbstraction` and implements `regions`, `numregions`, and `statespace` methods.

## Target models
```@docs
IMDPTarget
SparseIMDPTarget
OrthogonalIMDPTarget
SparseOrthogonalIMDPTarget
MixtureIMDPTarget
SparseMixtureIMDPTarget
```

## Constructing finite-state abstractions
```@docs
abstraction
```