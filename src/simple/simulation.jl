"""
    simulate_continuous(i, t, η)

Simulate a simple (linear) birth and death process starting at time zero with
`i` units. The stochastic process is observed continuously for `t` time units.
Rates are given by parameter `η = [λ, μ]` where `λ` is the birth rate and `μ` is
the death rate.
"""
function simulate_continuous(
  i::I,
  t::F,
  η::Vector{F}
) where {
  I <: Integer,
  F <: AbstractFloat
}
  ω = η[1] + η[2]
  p = η[1] / ω

  if t < floatmin(F)
    return (zero(Int), zero(Int), zero(F), zero(Float64), zero(Int), [], Int(i),
            [])
  end

  tot_births = 0
  tot_deaths = 0
  integrated_jump = zero(F)
  sum_log_n = zero(Float64)

  # pre-allocate vectors. If needed, we will grow the vector by doubling its
  # size each time.
  len = 100
  waiting_time = zeros(F, len)
  increment = zeros(Int, len)

  popul_size = i
  h = 0
  total_time = zero(F)
  while true
    # sample next event time from a negative exponential distribution with rate
    # n * (λ + μ)
    τ = - log(rand(F)) / (popul_size * ω)
    total_time += τ

    if total_time > t
      integrated_jump += popul_size * (t - total_time + τ)
      break
    end

    if h == len - 1
      # not enough space to store this event
      len *= 2
      resize!(waiting_time, len)
      resize!(increment, len)
    end

    h += 1

    integrated_jump += popul_size * τ
    sum_log_n += log(popul_size)
    waiting_time[h] = τ

    # sample event type with probability (λ / (λ + μ), μ / (λ + μ))
    if rand(F) <= p
      tot_births += 1
      increment[h] = 1
      popul_size += 1
    else
      tot_deaths += 1
      increment[h] = -1
      popul_size -= 1

      if popul_size == 0
        break
      end
    end
  end

  resize!(waiting_time, h)
  resize!(increment, h)

  (tot_births, tot_deaths, integrated_jump, sum_log_n, h, waiting_time, Int(i),
   increment)
end

"""
    simulate_discrete(i, k, u, η)

Simulate a simple (linear) process starting at time zero with `i` units. The
stochastic process is observed every `u` time units until it reaches `u * k`
time. Rates are given by parameter `η = [λ, μ]` where `λ` is the birth rate and
`μ` is the death rate.
"""
function simulate_discrete(
  i::I,
  k::I,
  u::F,
  η::Vector{F}
)::Vector{I} where {
  I <: Integer,
  F <: AbstractFloat
}
  if k == 0
    return [i]
  end

  ω = η[1] + η[2]
  p = η[1] / ω

  event_time = zero(F)
  popul_size = i

  sim_size = zeros(I, k + 1)
  sim_size[1] = popul_size

  counter = 1
  t = u
  while true
    # sample event time from a negative exponential distribution with rate
    # n * (λ + μ)
    event_time -= log(rand(F)) / F(popul_size * ω)

    # new event might be after various null observations
    while (event_time > t) && (counter <= k)
      counter += 1
      t = counter * u
      sim_size[counter] = popul_size
    end

    # sample event type with probability (λ / (λ + μ), μ / (λ + μ))
    popul_size += (rand(F) <= p) ? I(1) : -I(1)

    if (counter > k) || (popul_size == 0)
      break
    end
  end

  sim_size
end

"""
    rand_continuous(n, i, t, η)

Simulate `n` simple (linear) birth and death processes with the same rate
parameter `η = [λ, μ]`, where `λ` is the birth rate and `μ` is the death rate.
Starting at time `0` with `i` units, observe continuously each process until
time `t` is reached.

For simulating only one process, use `rand_continuous(i, t, η)`.
"""
function rand_continuous(
  n::Integer,
  i::Integer,
  t::Real,
  η::Vector{T}
) where {
  T <: Real
}
  if n < 1
    error("Number of simulations 'n' is less than or equal to zero.")
  end

  if i < 1
    error("Initial population size 'i' is less than or equal to zero.")
  end

  if t < 0
    error("Time 't' is negative.")
  end

  if η[1] < 0
    error("Birth rate 'η[1]' is negative.")
  end

  if η[2] < 0
    error("Death rate 'η[2]' is negative.")
  end

  if (η[1] < floatmin(float(η[1]))) && (η[2] < floatmin(float(η[2])))
    error("Birth rate 'η[1]' and death rate 'η[2]' are jointly zero.")
  end

  F = typeof(float((η[1] - η[2]) * t))

  [ObservationContinuousTime{F}(
   simulate_continuous(i, F(t), [F(η[1]), F(η[2])])...) for s = 1:n]
end

"""
    rand_continuous(i, t, η)

Simulate one simple (linear) birth and death process with rate parameter
`η = [λ, μ]`, where `λ` is the birth rate and `μ` is the death rate. Starting at
time `0` with `i` units, observe continuously the process until time `t` is
reached.

For simulating `n` processes, use `rand_continuous(n, i, t, η)`.
"""
function rand_continuous(
  i::Integer,
  t::Real,
  η::Vector{T}
) where {
  T <: Real
}

  if i < 1
    error("Initial population size 'i' is less than or equal to zero.")
  end

  if t < 0
    error("Time 't' is negative.")
  end

  if η[1] < 0
    error("Birth rate 'η[1]' is negative.")
  end

  if η[2] < 0
    error("Death rate 'η[2]' is negative.")
  end

  if (η[1] < floatmin(float(η[1]))) && (η[2] < floatmin(float(η[2])))
    error("Birth rate 'η[1]' and death rate 'η[2]' are jointly zero.")
  end

  F = typeof(float((η[1] - η[2]) * t))

  ObservationContinuousTime{F}(
  simulate_continuous(i, F(t), [F(η[1]), F(η[2])])...)
end

"""
    rand_discrete(n, i, k, u, η)

Simulate `n` simple (linear) birth and death processes with the same rate
parameter `η = [λ, μ]`, where `λ` is the birth rate and `μ` is the death rate.
Starting at time `0` with `i` units, observe each process for a total of `k`
times, each separated by the same lag `u`.

For simulating only one process, use `rand_discrete(i, k, u, η)`.
"""
function rand_discrete(
  n::Integer,
  i::Integer,
  k::Integer,
  u::Real,
  η::Vector{T}
) where {
  T <: Real
}
  if n < 1
    error("Number of simulations 'n' is less than or equal to zero.")
  end

  if i < 1
    error("Initial population size 'i' is less than or equal to zero.")
  end

  if k < 0
    error("Number of observations 'k' is negative.")
  end

  if u < floatmin(float(u))
    error("Time lag 'u' is less than or equal to zero.")
  end

  if η[1] < 0
    error("Birth rate 'η[1]' is negative.")
  end

  if η[2] < 0
    error("Death rate 'η[2]' is negative.")
  end

  if (η[1] < floatmin(float(η[1]))) && (η[2] < floatmin(float(η[2])))
    error("Birth rate 'η[1]' and death rate 'η[2]' are jointly zero.")
  end

  I = typeof(i)
  F = typeof(float((η[1] - η[2]) * u))

  state = zeros(I, k + 1, n)

  for s = 1:n
    state[:, s] = simulate_discrete(i, I(k), F(u), [F(η[1]), F(η[2])])
  end

  ObservationDiscreteTimeEven{F, I}(Int(n), Int(k), F(u), state)
end

"""
    rand_discrete(i, k, u, η)

Simulate one simple (linear) birth and death process with rate parameter
`η = [λ, μ]`, where `λ` is the birth rate and `μ` is the death rate. Starting at
time `0` with `i` units, observe each process for a total of `k` times, each
separated by the same lag `u`.

For simulating `n` processes, use `rand_discrete(n, i, k, u, η)`.
"""
function rand_discrete(
  i::Integer,
  k::Integer,
  u::Real,
  η::Vector{T}
) where {
  T <: Real
}
  rand_discrete(1, i, k, u, η)
end
