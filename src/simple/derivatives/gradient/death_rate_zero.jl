"""
    gradient_μ_zero(λ, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, 0)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_μ_zero(
  λ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  # functions are subject to catastrophic cancellation if too small
  if λ * t > 0.095
    gradient_μ_zero_stable(λ, i, j, t)
  else
    gradient_μ_zero_unstable(λ, i, j, t)
  end
end

"""
    gradient_μ_zero_stable(λ, i, j, t)

Functions are numerically stable and can be applied as they are. We still need
to be careful with overflow.
"""
function gradient_μ_zero_stable(
  λ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)

  θ = λ * t

  ept = exp(θ)
  emt = exp(-θ)
  em1 = expm1(θ)

  if i == 1
    if j > 1
      y = - j * t * expm1(θ - log(j)) / em1

      ∇[1] = y
      ∇[2] = - y - (2 - (j + 1) * emt) / λ
    elseif j == 0
      ∇[1] = - t * ept / em1
    elseif j == 1
      ∇[1] = -t
      ∇[2] = t + 2 * expm1(-θ) / λ
    end
  elseif j == 0
    u = i * t * (ept / em1)

    ∇[1] = -u

    if i > 1
      ∇[2] = -u * expm1(logexpm1(θ) - log(θ))
    end
  elseif j == i
    ∇[1] = -i * t
    ∇[2] = i * (θ + (i - 1) * ept + (i + 1) * emt - 2 * i) / λ
  else
    y = - j * t * expm1(θ + log(i / j)) / em1

    ∇[1] = y

    if j != i - 1
      ∇[2] = - y + (i * j / (i - j - 1)) * (2 - ((i - 1) / j) * ept - ((j + 1) / i) * emt) / λ
    end
  end

  ∇
end

"""
    gradient_μ_zero_unstable(λ, i, j, t)

Functions are subject to catastrophic cancellation and therefore we will do
series expansion to approximate their values.
"""
function gradient_μ_zero_unstable(
  λ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)

  θ = λ * t

  if i == 1
    if j > 1
      y = - j * t * expm1(θ - log(j)) / expm1(θ)

      ∇[1] = y
      ∇[2] = - y - (2 - (j + 1) / exp(θ)) / λ
    elseif j == 0
      ∇[1] = - t * exp(θ) / expm1(θ)
    elseif j == 1
      ∇[1] = -t
      ∇[2] = t + 2 * expm1(-θ) / λ
    end
  elseif j == 0
    u = i * t * (exp(θ) / expm1(θ))

    ∇[1] = -u

    if i > 1
      ∇[2] = -u * expm1(logexpm1(θ) - log(θ))
    end
  elseif j == i
    ∇[1] = -i * t

    ∇[2] = begin
      x0 = 1
      x1 = i
      x2 = 1 / 3
      x3 = i / 12
      x4 = 1 / 60
      x5 = i / 360
      x6 = 1 / 2520
      x7 = i / 20160
      x8 = 1 / 181440
      x9 = i / 1814400

      -i * t * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
    end
  else
    ∇[1] = - j * t * expm1(θ + log(i) - log(j)) / expm1(θ)

    if j != i - 1
      ∇[2] = begin
        y1 = i + j
        y2 = y1^2 - i + j
        y3 = 8 * i * j

        x0 = y1 / 2
        x1 = (5 * y2 - y3) / (12 * (i - j - 1))
        x2 = y1 / 6
        x3 = (31 * y2 - 8 * y3) / (720 * (i - j - 1))
        x4 = y1 / 120
        x5 = (41 * y2 - 10 * y3) / (30240 * (i - j - 1))
        x6 = y1 / 5040
        x7 = (31 * y2 - 8 * y3) / (1209600 * (i - j - 1))
        x8 = y1 / 362880
        x9 = (61 * y2 - 14 * y3) / (239500800 * (i - j - 1))

        -t * (x0 + θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9)))))))))
      end
    end
  end

  ∇
end
