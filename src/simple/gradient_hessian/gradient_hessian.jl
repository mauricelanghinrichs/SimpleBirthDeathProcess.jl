include("death_rate_zero.jl")
include("birth_rate_zero.jl")
include("equal_rates.jl")
include("unequal_rates.jl")

"""
    gradient_hessian(η, i, j, t)

Compute the gradient and Hessian of the log-transition probability of a simple
birth and death process evaluated at the point ``η = (λ, μ)^{\\prime}``.
Variable `i` is the initial population size, `j` is the final population size,
and `t` is the elapsed time.
"""
function gradient_hessian(
  η::Vector{F},
  i::I,
  j::I,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat,
  I <: Integer
}
  ϵ = floatmin(F)

  if t < ϵ
    fill(zero(F), 2), Symmetric(fill(zero(F), (2, 2)))
  elseif (η[1] < ϵ) && (η[2] < ϵ)
    if i == j
      fill(- i * t, 2), Symmetric([zero(F) (i * t)^2; (i * t)^2 zero(F)])
    else
      fill(F(NaN), 2), Symmetric(fill(F(NaN), (2, 2)))
    end
  elseif η[1] < ϵ
    gradient_hessian_λ_zero(η[2], i, j, t)
  elseif η[2] < ϵ
    gradient_hessian_μ_zero(η[1], i, j, t)
  elseif η[2] ≈ η[1]
    # TODO: Improve numerical accuracy of formulas
    if (F === BigFloat) || (η[1] * t > 0.001)
      gradient_hessian_equal_rates(η[1], F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        l = BigFloat(η[1])
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        g, H = gradient_hessian_equal_rates(l, a, b, s)

        F[g[1], g[2]], Symmetric(F[H[1, 1] H[2, 1]; H[1, 2] H[2, 2]])
      end
    end
  else
    if (F === BigFloat) || (log(η[1] / η[2]) / ((η[1] - η[2]) * t) < 1_000)
      gradient_hessian_unequal_rates(η, F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        l = BigFloat(η[1])
        m = BigFloat(η[2])
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        g, H = gradient_hessian_unequal_rates([l, m], a, b, s)

        F[g[1], g[2]], Symmetric(F[H[1, 1] H[2, 1]; H[1, 2] H[2, 2]])
      end
    end
  end
end

function gradient_hessian(
  η::Vector{F},
  x::ObservationDiscreteTimeEven
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat
}
  ∇ = fill(zero(Float64), 2)
  H = Symmetric(fill(zero(Float64), (2, 2)))

  for n = 1:x.n, s = 2:x.k
    a, b = gradient_hessian(η, x.state[s - 1, n], x.state[s, n], x.u)
    ∇ .+= a
    H .+= b
  end

  ∇, H
end
