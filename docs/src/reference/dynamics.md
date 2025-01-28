# Dynamics

## General
```@docs
DiscreteTimeStochasticDynamics
dimstate
diminput
System
dynamics
initial
```

## Additive noise systems
```@docs
AdditiveNoiseDynamics
nominal
prepare_nominal
noise
AffineAdditiveNoiseDynamics
NonlinearAdditiveNoiseDynamics
UncertainPWAAdditiveNoiseDynamics
UncertainAffineRegion
```

### Additive noise structures
```@docs
AdditiveNoiseStructure
AdditiveDiagonalGaussianNoise
AdditiveCentralUniformNoise
```

## Abstracted Gaussian processes
```@docs
AbstractedGaussianProcess
AbstractedGaussianProcessRegion
gp_bounds
region
mean_lower
mean_upper
stddev_lower
stddev_upper
```

## Stochastic switched systems
```@docs
StochasticSwitchedDynamics
```