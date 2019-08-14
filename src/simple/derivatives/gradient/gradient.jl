"""
    gradient(η, i, j, t)

Compute the gradient of the log-transition probability of a simple birth and
death process evaluated at the point ``η = (λ, μ)^{\\prime}``.
Variable `i` is the initial population size, `j` is the final population size,
and `t` is the elapsed time.
"""
function gradient(
  η::Vector{F},
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ϵ = floatmin(F)

  if t < ϵ
    zeros(F, 2)
  elseif (η[1] < ϵ) && (η[2] < ϵ)
    if i == j
      fill(- i * t, 2)
    else
      fill(F(NaN), 2)
    end
  elseif η[1] < ϵ
    gradient_λ_zero(η[2], i, j, t)
  elseif η[2] < ϵ
    gradient_μ_zero(η[1], i, j, t)
  elseif η[2] ≈ η[1]
    # TODO: Improve numerical accuracy of formulas
    if (F === BigFloat) || (η[1] * t > 0.001)
      gradient_equal_rates(η[1], F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        l = BigFloat(η[1])
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        gradient_equal_rates(l, a, b, s)
      end
    end
  else
    if (F === BigFloat) || (log(η[1] / η[2]) / ((η[1] - η[2]) * t) < 1_000)
      gradient_unequal_rates(η, F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        l = BigFloat(η[1])
        m = BigFloat(η[2])
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        gradient_unequal_rates([l; m], a, b, s)
      end
    end
  end
end

function gradient(
  η::Vector{F},
  x::ObservationDiscreteTimeEqual{F, I}
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = zeros(F, 2)

  for n = 1:x.n, s = 2:size(x.state, 1)
    ∇ .+= gradient(η, x.state[s - 1, n], x.state[s, n], x.u)
  end

  ∇
end

function gradient(
  η::Vector{F},
  x::Vector{ObservationDiscreteTimeEqual{F, I}}
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = zeros(F, 2)

  for m = 1:length(x), n = 1:x[m].n, s = 2:size(x[m].state, 1)
    ∇ .+= gradient(η, x[m].state[s - 1, n], x[m].state[s, n], x[m].u)
  end

  ∇
end
