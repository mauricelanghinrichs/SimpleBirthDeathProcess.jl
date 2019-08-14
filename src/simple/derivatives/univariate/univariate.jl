"""
    univariate_derivatives(μ, i, j, t, θ)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function univariate_derivatives(
  μ::F,
  i::I,
  j::I,
  t::F,
  θ::F
)::Tuple{F, F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ϵ = floatmin(F)

  if abs(θ) <= ϵ
    # TODO: Improve numerical accuracy of formulas
    if (F === BigFloat) || (μ * t > 0.001)
      univariate_derivatives_equal_rates(μ, F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        m = BigFloat(μ)
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        d1, d2 = univariate_derivatives_equal_rates(m, a, b, s)

        F(d1), F(d2)
      end
    end
  else
    if (F === BigFloat) || (log((θ + μ) / μ) / (θ * t) < 1_000)
      univariate_derivatives_unequal_rates(μ, F(i), F(j), t, θ)
    else
      setprecision(BigFloat, 256) do
        m = BigFloat(μ)
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)
        v = BigFloat(θ)

        d1, d2 = univariate_derivatives_unequal_rates(m, a, b, s, v)

        F(d1), F(d2)
      end
    end
  end
end

"""
    univariate_derivatives(μ, θ, x)

Compute the first and second derivatives of the log-likelihood of the sample
`x` evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function univariate_derivatives(
  μ::F,
  θ::F,
  x::ObservationDiscreteTimeEqual{F, I}
)::Tuple{F, F} where {
  F <: AbstractFloat,
  I <: Integer
}
  d1 = zero(F)
  d2 = zero(F)

  for n = 1:x.n, s = 2:size(x.state, 1)
    r1, r2 = univariate_derivatives(μ, x.state[s - 1, n], x.state[s, n], x.u, θ)
    d1 += r1
    d2 += r2
  end

  d1, d2
end

function univariate_derivatives(
  μ::F,
  θ::F,
  x::Vector{ObservationDiscreteTimeEqual{F, I}}
)::Tuple{F, F} where {
  F <: AbstractFloat,
  I <: Integer
}
  d1 = zero(F)
  d2 = zero(F)

  for m = 1:length(x), n = 1:x[m].n, s = 2:size(x[m].state, 1)
    r1, r2 = univariate_derivatives(μ, x[m].state[s - 1, n], x[m].state[s, n],
                                    x[m].u, θ)
    d1 += r1
    d2 += r2
  end

  d1, d2
end
