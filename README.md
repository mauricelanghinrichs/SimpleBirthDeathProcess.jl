# Simple (linear) birth and death process
[![Build Status](https://travis-ci.com/albertopessia/SimpleBirthDeathProcess.jl.svg?branch=master)](https://travis-ci.com/albertopessia/SimpleBirthDeathProcess.jl) [![Coverage](https://codecov.io/gh/albertopessia/SimpleBirthDeathProcess.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/albertopessia/SimpleBirthDeathProcess.jl)

## About
This [Julia](http://julialang.org/) package provides functions for fitting a simple birth and death process (BDP) without migration, also known as the [Kendall process](https://doi.org/10.1111/j.2517-6161.1949.tb00032.x) [\[1\]](#references).

## Installation
_SimpleBirthDeathProcess_ can be easily installed from within Julia:

- Enter the Pkg REPL-mode by pressing  `]` in the Julia REPL
- Issue the command `add https://github.com/albertopessia/SimpleBirthDeathProcess.jl/`
- Press the _Backspace_ key to return to the Julia REPL

## Usage
In what follows we will use the conventions

- `i`: initial population size at time `0`
- `j`: final population size at the end of the observation.
- `t`: total amount of time the stochastic process is observed.
- `λ`: birth rate
- `μ`: death rate
- `η`: vector of parameters `[λ, μ]`

It is assumed that `i` and `j` are both integer numbers, with `i > 0` and `j ≧ 0`.
Parameters `t`, `λ`, and `μ` are non-negative real numbers.

### Transition probability
To evaluate the log-transition probability use

```julia
trans_prob(i, j, t, η)
```

By default, the function returns the logarithm of the transition probability.
To obtain the actual probability set to `false` the keyword `log_value`:

```julia
trans_prob(i, j, t, η, log_value=false)
```

### Simulations
#### Continuous case
To simulate one simple BDP observed continuously over time, use

```julia
rand_continuous(i, t, η)
```

To simulate `n` independent and identically distributed simple BDPs observed continuously over time, use

```julia
rand_continuous(n, i, t, η)
```

#### Discrete case
To simulate one simple BDP observed at `k` time points, equally spaced with same lag `u`, use

```julia
rand_discrete(i, k, u, η)
```

To simulate `n` independent and identically distributed simple BDPs, observed at `k` time points equally spaced by the same lag `u`, use

```julia
rand_discrete(n, i, k, u, η)
```

### Input data
#### Continuous case
If your data was observed continuously, denote with ``s[1], ..., s[h]`` the exact times at which birth or death events occurred.
Denote with ``x[1], ..., x[h]`` the corresponding population sizes observed at ``s[1], ..., s[h]``.
To create a `ObservationContinuousTime` object for data analysis use

```julia
observation_continuous_time(s, x, t)
```

where `x` is the (integer) vector of population sizes of length ``h``, `s` is the vector of event times of length ``h``, and `t` is the total time the process was observed.

#### Discrete case
If your data was observed only at pre-specified fixed points, we need to consider two distinct cases: equally or unequally distributed time points.
When the time points are equidistant define

- `u`: non-negative time lag equally separating each observation
- `x`: vector of length ``k`` (or ``k``-by-``n`` matrix for the case of ``n`` i.i.d. simple BDPs) with the observed population sizes

To create a `ObservationDiscreteTimeEqual` object for data analysis use

```julia
observation_discrete_time_equal(u, x)
```

When the time points are unequally spaced use instead

```julia
observation_discrete_time_unequal(t, x)
```

to create a `ObservationDiscreteTimeUnequal` object, where `x` is the (integer) vector of population sizes of length ``h`` and `t` is the vector of event times of same length ``h``.

### log-likelihood function
To evaluate the log-likelihood function you first need to create one of `ObservationContinuousTime`, `ObservationDiscreteTimeEqual`, `ObservationDiscreteTimeUnequal` as described in the previous sections, either by simulation or by converting pre-existing data.
Call such object `obs_data`.

The value of the log-likelihood function associated with the observed data for a particular combination of parameters `η = [λ, μ]` can be obtained by

```julia
loglik(η, obs_data)
```

### Maximum likelihood estimation
You can compute the maximum likelihood estimator with the function

```julia
mle(obs_data)
```

## License
See [LICENSE.md](LICENSE.md)

## References
\[1\] Kendall, D. G. (1949). Stochastic Processes and Population Growth. _Journal of the Royal Statistical Society: Series B (Methodological)_, 11(2): 230-264. doi: [10.1111/j.2517-6161.1949.tb00032.x](https://doi.org/10.1111/j.2517-6161.1949.tb00032.x)
