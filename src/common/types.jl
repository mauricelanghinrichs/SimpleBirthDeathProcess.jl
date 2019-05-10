"""
    ObservationContinuousTime

Parametric composite type to store a single birth and death process realization,
observed continuously over time.
"""
struct ObservationContinuousTime{
  F <: AbstractFloat
}
  tot_births::Int
  tot_deaths::Int
  integrated_jump::F

  # sum of log population sizes (ancillary statistic)
  sum_log_n::Float64

  len::Int
  waiting_time::Vector{F}
  initial_population_size::Int
  increment::Vector{Int}
end

"""
    observation_continuous_time(t, x, observation_time)

Construct an object of type `ObservationContinuousTime` from a continuously
observed time series. Variable `x` is the vector of observed population sizes
while `t` is the vector of event times. `observation_time` is the total amount
of time used to observe the process, that is `t[end] ≦ observation_time`.
"""
function observation_continuous_time(
  t::Vector{F},
  x::Vector{Int},
  observation_time::F
) where {
  F <: AbstractFloat
}
  if length(x) != length(t)
    error("Arguments 't' and 'x' do not have the same length.")
  end

  if t[1] != 0
    error("The first observation time 't[1]' is not zero.")
  end

  if any(i -> i < 0, x)
    error("Vector 'x' cannot contain negative values.")
  end

  if any(i -> i < 0, t)
    error("Vector 't' cannot contain negative values.")
  end

  if !issorted(t)
    error("Vector 't' is not sorted in ascending order.")
  end

  if observation_time < t[end]
    error("'observation_time' is less than 't[end]'.")
  end

  tot_births = 0
  tot_deaths = 0
  integrated_jump = zero(F)
  len = length(t) - 1
  waiting_time = zeros(F, len)
  increment = zeros(Int, len)

  for s = 2:length(t)
    waiting_time[s - 1] = t[s] - t[s - 1]
    increment[s - 1] = x[s] - x[s - 1]

    if (abs(waiting_time[s - 1]) < floatmin(F)) && (increment[s - 1] != 0)
      error("Error at time point ", s, ". Observation time is zero but ",
            "'x[", s - 1, "]' and 'x[", s, "]' are different.")
    end

    if increment[s - 1] == -1
      tot_deaths += 1
    elseif increment[s - 1] == 1
      tot_births += 1
    else
      error("Error at time point ", s, ". Increment 'x[", s, "] - x[", s - 1,
            "]' is different from ±1.")
    end

    # add current waiting_time times the previous population size
    integrated_jump += x[s - 1] * waiting_time[s - 1]
  end

  # no events between t[end] and observation_time
  integrated_jump += x[s] * (observation_time - t[end])

  sum_log_n = sum(log, x[1:len])

  ObservationContinuousTime{F}(tot_births, tot_deaths, integrated_jump,
                               sum_log_n, len, waiting_time, x[1], increment)
end

"""
    ObservationDiscreteTimeEven

Parametric composite type to store birth and death process realizations,
observed at fixed time points with the same lag.
"""
struct ObservationDiscreteTimeEven{
  F <: AbstractFloat,
  I <: Integer
}
  # total number of samples
  n::Int

  # total number of time points
  k::Int

  # same time lag between each time points
  u::F

  # complete list of population sizes
  # column 'j' is j-th time series while row 'i' is time (i - 1) * u
  state::Matrix{I}
end

"""
    observation_discrete_time_even(u, x)

Construct an object of type `ObservationDiscreteTimeEven` from a discretely
observed time series. Variable `x` is a vector (or matrix if more than 1
observation) of observed population sizes while `u` is the fixed time lag after
which every observation was made.
"""
function observation_discrete_time_even(
  u::Real,
  x::Vector{I}
) where {
  I <: Integer
}
  if u <= 0
    error("Argument 'u' cannot be less than or equal to zero.")
  end

  if any(i -> i < 0, x)
    error("Vector 'x' cannot contain negative values")
  end

  k = length(x)
  F = typeof(float(u))

  ObservationDiscreteTimeEven{F, I}(1, k, F(u), reshape(x, k, 1))
end

function observation_discrete_time_even(
  u::Real,
  x::Matrix{I}
) where {
  I <: Integer
}
  if u <= 0
    error("Argument 'u' cannot be less than or equal to zero.")
  end

  if any(i -> i < 0, x)
    error("Matrix 'x' cannot contain negative values")
  end

  F = typeof(float(u))

  ObservationDiscreteTimeEven{F, I}(size(x, 2), size(x, 1), F(u), x)
end

"""
    ObservationDiscreteTimeUneven

Parametric composite type to store birth and death process realizations,
observed at discrete time points of varying length.
"""
struct ObservationDiscreteTimeUneven{
  F <: AbstractFloat,
  I <: Integer
}
  # complete list of waiting times
  waiting_time::Vector{F}

  # complete list of population sizes
  state::Vector{I}
end

"""
    observation_discrete_time_uneven(x, t)

Construct an object of type `ObservationDiscreteTimeUneven` from a discretely
observed time series. Variable `x` is a vector of observed population sizes
while `t` is the vector of event times.
"""
function observation_discrete_time_uneven(
  t::Vector{F},
  x::Vector{I}
) where {
  F <: AbstractFloat,
  I <: Integer
}
  if length(x) != length(t)
    error("Arguments 't' and 'x' do not have the same length.")
  end

  if t[1] != 0
    error("The first observation time 't[1]' is not zero.")
  end

  if any(i -> i < 0, x)
    error("Vector 'x' cannot contain negative values.")
  end

  if any(i -> i < 0, t)
    error("Vector 't' cannot contain negative values.")
  end

  if !issorted(t)
    error("Vector 't' is not sorted in ascending order.")
  end

  waiting_time = zeros(F, length(t) - 1)

  for s = 2:length(t)
    waiting_time[s - 1] = t[s] - t[s - 1]

    if (abs(waiting_time[s - 1]) < floatmin(F)) && (x[s - 1] != x[s])
      error("Error at time point ", s, ". Observation time is zero but ",
            "'x[", s - 1, "]' and 'x[", s, "]' are different.")
    end
  end

  ObservationDiscreteTimeUneven{F, I}(waiting_time, x)
end
