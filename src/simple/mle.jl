"""
    mle(x)

Given an observed sample `x`, return the maximum likelihood estimate (MLE) of a
simple birth and death process.
"""
function mle(
  x::ObservationContinuousTime{F}
)::Vector{Float64} where {
  F <: AbstractFloat
}
  [x.tot_births / x.integrated_jump; x.tot_deaths / x.integrated_jump]
end

function mle(
  x::Vector{ObservationContinuousTime{F}}
)::Vector{Float64} where {
  F <: AbstractFloat
}
  B = sum(y -> y.tot_births, x)
  D = sum(y -> y.tot_deaths, x)
  T = sum(y -> y.integrated_jump, x)

  [B / T; D / T]
end

function mle(
  x
)
  θ = approximate_growth_rate(x)
  F = typeof(θ)

  if isinf(θ)
    return (θ < 0) ? [zero(F); F(Inf)] : [F(Inf); zero(F)]
  end

  # anything below this value will be treated as equivalent to zero
  tiny = F(1.0e-20)

  init_value = abs(θ)

  # find the maximum at the border: this is going to be our reference point
  #
  # likelihood function can be infinite at the border
  # as an example, think of an observation such as [1000; 900; 1200] where the
  # know that the death rate cannot be zero (otherwise we cannot move from 1000
  # to 900)
  mle_border = zeros(F, 2)
  ll_border = loglik(mle_border, x)

  # find the maximum along the line μ = 0
  ll_λ = loglik([init_value; zero(F)], x)
  offset_λ = !isinf(ll_λ) ? zero(F) : floatmin(F)
  a, b, c = bracket_optimum(y -> [y; offset_λ], init_value, x)
  λ, ll_λ, converged_λ = brent_method(y -> [y; offset_λ], a, b, c, x)

  if !converged_λ
    @warn string("Something went wrong! ",
                 "I was not able to find an optimum along the border μ ≈ 0!",
                 "Nevertheless, I might be able to find the global optimum...")
  end

  # find the maximum along the line λ = 0
  ll_μ = loglik([zero(F); init_value], x)
  offset_μ = !isinf(ll_μ) ? zero(F) : floatmin(F)
  a, b, c = bracket_optimum(y -> [offset_μ; y], init_value, x)
  μ, ll_μ, converged_μ = brent_method(y -> [offset_μ; y], a, b, c, x)

  if !converged_μ
    @warn string("Something went wrong! ",
                 "I was not able to find an optimum along the border λ ≈ 0!",
                 "Nevertheless, I might be able to find the global optimum...")
  end

  converged_border = true

  if (ll_λ > ll_μ) && (ll_λ > ll_border)
    mle_border[1] = λ
    mle_border[2] = offset_λ
    ll_border = ll_λ
    converged_border = converged_λ
  elseif (ll_μ > ll_λ) && (ll_μ > ll_border)
    mle_border[1] = offset_μ
    mle_border[2] = μ
    ll_border = ll_μ
    converged_border = converged_μ
  end

  # find a good starting point for the general search
  θ = mle_border[1] - mle_border[2]
  η = zeros(F, 2)

  if θ > 0
    a, b, c = bracket_optimum(y -> [θ + y; y], one(F), x)
    m, ll_m, converged = brent_method(y -> [θ + y; y], a, b, c, x)

    if m > 1.0e-20
      η[1] = θ + m
      η[2] = m
    else
      # the border value is actually the maximum
      return mle_border
    end
  elseif θ < 0
    a, b, c = bracket_optimum(y -> [y; y - θ], one(F), x)
    l, ll_l, converged = brent_method(y -> [y; y - θ], a, b, c, x)

    if l > 1.0e-20
      η[1] = l
      η[2] = l - θ
    else
      # the border value is actually the maximum
      return mle_border
    end
  else
    a, b, c = bracket_optimum(y -> [y; y], one(F), x)
    l, ll_l, converged = brent_method(y -> [y; y], a, b, c, x)

    if l > 1.0e-20
      η[1] = l
      η[2] = l
    else
      # the border value is actually the maximum
      return mle_border
    end
  end

  mle_point, mle_loglik, global_converged = multivariate_newton_raphson(η, x)

  if mle_loglik >= ll_border
    if !global_converged
      @warn string("I was not able to find a global optimum within the ",
                   "allowed number of steps! Beware, solution might not be a ",
                   "global optimum.")
    end

    mle_point
  else
    if !converged_border
      @warn string("I was not able to find a global optimum within the ",
                   "allowed number of steps! Beware, solution might not be a ",
                   "global optimum.")
    end

    mle_border
  end
end

"""
    approximate_growth_rate(x)

Find a value that is close to the MLE of the growth rate `θ`.
"""
function approximate_growth_rate(
  x::ObservationDiscreteTimeEqual{F, I}
)::F where {
  F <: AbstractFloat,
  I <: Integer
}
  # this is actually the maximum likelihood estimator of θ = (λ - μ)
  α = F(sum(x.state[2:end, :])) / F(sum(x.state[1:(end - 1), :]))
  log(α) / x.u
end

function approximate_growth_rate(
  x::Vector{ObservationDiscreteTimeEqual{F, I}}
)::F where {
  F <: AbstractFloat,
  I <: Integer
}
  # compute a weighted average of MLEs as an approximate point
  α = zero(F)
  w = zero(F)

  for y = x
    α += log(F(sum(y.state[2:end, :])) / F(sum(y.state[1:(end - 1), :])))
    w += y.u
  end

  α / w
end

"""
   bracket_optimum(η, init, x)

Bracket a maximum of the log-likelihood function by a one-dimensional interval
`0 <= a <= b <= c` such that `loglik(η(b), x) >= loglik(η(a), x)` and
`loglik(η(b), x) >= loglik(η(c), x)`. The choice of the maximizer is implied by
the function `η`.

Algorithm converted from Press et al. (2000) "Numerical recipes: the art of
scientific computing".
"""
function bracket_optimum(
  η::Function,
  init::F,
  x
)::Tuple{F, F, F} where {
  F <: AbstractFloat
}
  # we will search for the minimum of the negative log-likelihood which is
  # equivalent to the maximum of the log-likelihood

  # golden ratio
  ϕ = (1 + sqrt(F(5))) / 2

  # maximum magnification allowed for a parabolic-fit step
  parab_lim = F(100)

  # this tiny number prevents division by zero
  ϵ = F(1.0e-20)

  # initialize
  a = floatmin(F)
  b = init

  lla = -loglik(η(a), x)
  llb = -loglik(η(b), x)

  if llb > lla
    # Switch roles of a and b so that we can go downhill in the direction
    # from a to b
    a, b = b, a
    lla, llb = llb, lla
  end

  # first guess for c
  c = b + ϕ * (b - a)

  # lower bound cannot be negative
  if c <= 0
    c = floatmin(F)
  end

  llc = -loglik(η(c), x)

  while llb > llc
    # compute new candidate point by parabolic extrapolation
    r = (b - a) * (llb - llc)
    q = (b - c) * (llb - lla)

    numer = (b - c) * q - (b - a) * r
    denom = if q > r
      2 * max(q - r, ϵ)
    else
      - 2 * max(r - q, ϵ)
    end

    candidate = b - numer / denom
    candidate_lim = b + parab_lim * (c - b)

    if (b - candidate) * (candidate - c) > 0
      # parabolic fit is between b and c: try it
      candidate_ll = -loglik(η(candidate), x)

      if candidate_ll < llc
        # got a minimum between b and c
        a, b = b, candidate
        lla, llb = llb, candidate_ll
        break
      elseif candidate_ll > llb
        # got a minimum between a and candidate
        c = candidate
        llc = candidate_ll
        break
      else
        # parabolic fit was of no use. Use default magnification.
        candidate = c + ϕ * (c - b)
        candidate_ll = -loglik(η(candidate), x)
      end
    elseif (c - candidate) * (candidate - candidate_lim) > 0
      # parabolic fit is between c and its allowed limit
      candidate_ll = -loglik(η(candidate), x)

      if candidate_ll < llc
        b, c, candidate = c, candidate, candidate + ϕ * (candidate - c)
        llb, llc, candidate_ll = llc, candidate_ll, -loglik(η(candidate), x)
      end
    elseif (candidate - candidate_lim) * (candidate_lim - c) >= 0
      # limit parabolic candidate to maximum allowed value
      candidate = candidate_lim
      candidate_ll = -loglik(η(candidate), x)
    else
      # reject parabolic candidate and use default magnification
      candidate = c + ϕ * (c - b)
      candidate_ll = -loglik(η(candidate), x)
    end

    # eliminate oldest point and continue
    a, b, c = b, c, candidate
    lla, llb, llc = llb, llc, candidate_ll
  end

  if b >= a
    if c >= b
      a, b, c
    else
      a, c, b
    end
  else
    if c >= a
      b, a, c
    else
      b, c, a
    end
  end
end

"""
    brent_method(η, a, b, c, x)

Starting from a bracketing interval `0 <= a <= b <= c` such that
`loglik(η(b), x) >= loglik(η(a), x)` and `loglik(η(b), x) >= loglik(η(c), x)`,
find the maximum likelihood estimate using the Brent-Dekker algorithm.

Algorithm converted from Press et al. (2000) "Numerical recipes: the art of
scientific computing".
"""
function brent_method(
  η::Function,
  lower::F,
  inner::F,
  upper::F,
  x
)::Tuple{F, F, Bool} where {
  F <: AbstractFloat,
  I <: Integer
}
  # 1 / ϕ^2. ϕ is the golden ratio
  inv_ϕ2 = (3 - sqrt(F(5))) / 2

  rel_tol = sqrt(eps(F))
  abs_tol = eps(F)

  # distance moved on the step before last
  old_step = zero(F)
  new_step = zero(F)

  cur_optimum = inner
  cur_minimum = -loglik(η(inner), x)

  # previous step
  old_optimum_1 = cur_optimum
  old_minimum_1 = cur_minimum

  # two steps back
  old_optimum_2 = cur_optimum
  old_minimum_2 = cur_minimum

  converged = false
  max_iter = 1_000

  for iteration = 1:max_iter
    this_iter_tol = rel_tol * cur_optimum + abs_tol
    midpoint = (lower + upper) / 2

    if abs(cur_optimum - midpoint) <= 2 * this_iter_tol - (upper - lower) / 2
      converged = true
      break
    end

    if abs(old_step) > this_iter_tol
      # construct a trial parabolic fit
      r = (cur_optimum - old_optimum_1) * (cur_minimum - old_minimum_2)
      q = (cur_optimum - old_optimum_2) * (cur_minimum - old_minimum_1)
      p = (cur_optimum - old_optimum_2) * q - (cur_optimum - old_optimum_1) * r
      q = 2 * (q - r)

      if q > 0
        p = -p
      else
        q = -q
      end

      if (abs(p) >= abs(q * old_step / 2)) ||
         (p <= q * (lower - cur_optimum)) ||
         (p >= q * (upper - cur_optimum))
        # the above conditions determine the acceptability of the parabolic fit.
        # Here we take the golden section step into the larger of the two
        # segments
        old_step = if cur_optimum >= midpoint
          lower - cur_optimum
        else
          upper - cur_optimum
        end

        new_step = inv_ϕ2 * old_step
      else
        # take the parabolic step
        old_step = new_step
        new_step = p / q
        tmp_optimum = cur_optimum + new_step

        if (tmp_optimum - lower < 2 * this_iter_tol) ||
           (upper - tmp_optimum < 2 * this_iter_tol)
          new_step = if midpoint < cur_optimum
            -this_iter_tol
          else
            this_iter_tol
          end
        end
      end
    else
      old_step = if cur_optimum >= midpoint
        lower - cur_optimum
      else
        upper - cur_optimum
      end

      new_step = inv_ϕ2 * old_step
    end

    candidate = if abs(new_step) >= this_iter_tol
      cur_optimum + new_step
    else
      if new_step > 0
        cur_optimum + this_iter_tol
      else
        cur_optimum - this_iter_tol
      end
    end

    candidate_neg_loglik = -loglik(η(candidate), x)

    if candidate_neg_loglik <= cur_minimum
      if candidate >= cur_optimum
        lower = cur_optimum
      else
        upper = cur_optimum
      end

      old_optimum_2 = old_optimum_1
      old_minimum_2 = old_minimum_1

      old_optimum_1 = cur_optimum
      old_minimum_1 = cur_minimum

      cur_optimum = candidate
      cur_minimum = candidate_neg_loglik
    else
      if candidate < cur_optimum
        lower = candidate
      else
        upper = candidate
      end

      if (candidate_neg_loglik <= old_minimum_1) ||
         (old_optimum_1 == cur_optimum)
        old_optimum_2 = cur_optimum
        old_minimum_2 = cur_minimum

        old_optimum_1 = candidate
        old_minimum_1 = candidate_neg_loglik
      elseif (candidate_neg_loglik <= old_minimum_2) ||
             (old_optimum_2 == cur_optimum) ||
             (old_optimum_2 == old_optimum_1)
        old_optimum_2 = candidate
        old_minimum_2 = candidate_neg_loglik
      end
    end
  end

  (cur_optimum, -cur_minimum, converged)
end

"""
    multivariate_newton_raphson(η, x)

Starting from the point `η` find a new value such that the log-likelihood
function evaluated at `x` is maximized.
"""
function multivariate_newton_raphson(
  η::Vector{F},
  x
)::Tuple{Vector{F}, F, Bool} where {
  F <: AbstractFloat
}
  ϵ = sqrt(eps(F))

  cur_optimum = copy(η)
  cur_maximum = loglik(η, x)

  converged = false
  alt_iter = 100
  max_iter = 1_000

  for iteration = 1:max_iter
    # reset the step size
    γ = one(F)

    ∇, H = gradient_hessian(cur_optimum, x)

    detH = H[1, 1] * H[2, 2] - H[2, 1]^2
    invH = [H[2, 2] -H[2, 1]; -H[2, 1] H[1, 1]] ./ detH
    step_size = invH * ∇
    candidate = cur_optimum .- step_size

    # did we overshoot outside the domain?
    counter = 1
    while ((candidate[1] < 0) || (candidate[2] < 0)) && (counter <= alt_iter)
      γ /= 2
      candidate = cur_optimum .- γ .* step_size
      counter += 1
    end

    if counter > alt_iter
      # algorithm did not converge
      break
    end

    candidate_ll = loglik(candidate, x)

    # by definition log-likelihood must always increase. If this is
    # not the case we took a big step at the previous iteration
    counter = 1
    while (candidate_ll < cur_maximum) && (counter <= alt_iter)
      γ /= 2
      candidate = cur_optimum .- γ .* step_size
      candidate_ll = loglik(candidate, x)
      counter += 1
    end

    if counter > alt_iter
      # algorithm did not converge
      break
    end

    # test the current solution
    if (abs(1 - cur_optimum[1] / candidate[1]) < ϵ) &&
       (abs(1 - cur_optimum[2] / candidate[2]) < ϵ)
      copy!(cur_optimum, candidate)
      cur_maximum = candidate_ll
      converged = true
      break
    end

    copy!(cur_optimum, candidate)
    cur_maximum = candidate_ll
  end

  (cur_optimum, cur_maximum, converged)
end
