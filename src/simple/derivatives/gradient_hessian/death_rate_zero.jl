"""
    gradient_hessian_μ_zero(λ, i, j, t)

Compute the gradient and Hessian of the log-probability of a simple birth and
death process evaluated at the point ``η = (λ, 0)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_hessian_μ_zero(
  λ::F,
  i::I,
  j::I,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat,
  I <: Integer
}
  # functions are subject to catastrophic cancellation if too small
  if λ * t > 0.095
    gradient_hessian_μ_zero_stable(λ, i, j, t)
  else
    gradient_hessian_μ_zero_unstable(λ, i, j, t)
  end
end

"""
    gradient_hessian_μ_zero_stable(λ, i, j, t)

Functions are numerically stable and can be applied as they are. We still need
to be careful with overflow.
"""
function gradient_hessian_μ_zero_stable(
  λ::F,
  i::I,
  j::I,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)
  H = fill(F(NaN), (2, 2))

  θ = λ * t

  ept = exp(θ)
  emt = exp(-θ)

  em1 = expm1(θ)
  lem1 = logexpm1(θ)

  lθ = log(θ)
  lt = log(t)
  li = log(i)

  if i == 1
    if j > 1
      y = - j * t * expm1(θ - log(j)) / em1

      ∇[1] = y
      ∇[2] = - y - (2 - (j + 1) * emt) / λ

      w1 = - (j - 1) * ept * (t / em1)^2
      w2 = -w1 + (2 - (j + 1) * (1 + θ) * emt) / λ^2
      w3 = w1 + ((j + 1) * (emt + 2 * θ) * emt - 2) / λ^2

      H[1, 1] = w1
      H[2, 1] = w2
      H[1, 2] = w2
      H[2, 2] = w3
    elseif j == 0
      y = t / em1
      ∇[1] = - y * ept
      H[1, 1] = y^2 * ept
      H[2, 2] = F(-Inf)
    elseif j == 1
      ∇[1] = -t
      ∇[2] = t + 2 * expm1(-θ) / λ

      y = (log1p(θ) - θ) / 2
      w = -(2 / λ^2) * (exp(y) + 1) * expm1(y)

      H[1, 1] = zero(F)
      H[2, 1] = w
      H[1, 2] = w
      H[2, 2] = (2 / λ^2) * ((emt + 2 * θ) * emt - 1)
    end
  elseif j == 0
    u = i * t * (ept / em1)

    ∇[1] = -u

    H[1, 1] = exp(li + 2 * (lt - lem1) + θ)

    if i > 1
      ∇[2] = -u * expm1(lem1 - lθ)

      w = if θ > 1
        -exp(li + 2 * (lt - lem1) + θ + log1p((θ - 1) * (em1 / θ)^2))
      else
        x = sqrt(1 - θ) * em1 / θ
        -exp(li + 2 * (lt - lem1) + θ + log1p(x) + log1p(-x))
      end

      H[2, 1] = w
      H[1, 2] = w

      if i > 2
        x1 = exp(θ / 2) + θ / em1
        x2 = exp(θ / 2) - θ / em1

        H[2, 2] = -exp(li + θ + 2 * (lt - lθ)) * muladd(x1, x2, -2 * θ)
      end
    end
  elseif j == i
    ∇[1] = -i * t
    ∇[2] = i * (θ + (i - 1) * ept + (i + 1) * emt - 2 * i) / λ

    w = i * (2 * i - (i - 1) * (1 - θ) * ept - (i + 1) * (1 + θ) * emt) / λ^2

    x1 = (i - 1) * ept * (ept / (2 * θ) - 1)
    x2 = (i + 1) * emt * (1 / (2 * θ * ept) + 1)
    x3 = (i - 1) * (i + 1) * (emt * em1^2 / 2)^2

    H[1, 1] = zero(F)
    H[2, 1] = w
    H[1, 2] = w
    H[2, 2] = - 2 * (i / λ)^2 * (1 - θ * (x1 + x2) / i + x3)
  else
    lj = log(j)

    y = - j * t * expm1(θ + li - lj) / em1
    w1 = (i - j) * exp(θ + 2 * (lt - lem1))

    ∇[1] = y
    H[1, 1] = w1

    if j != i - 1
      ∇[2] = - y + (i * j / (i - j - 1)) * (2 - ((i - 1) / j) * ept - ((j + 1) / i) * emt) / λ

      w2 = -w1 - (i * j / (i - j - 1)) * (1 / λ^2 + t / λ) * ((2 / (1 + θ)) - ((i - 1) / j) * ((1 - θ) / (1 + θ)) * ept - ((j + 1) / i) * emt)

      H[2, 1] = w2
      H[1, 2] = w2

      if j != i - 2
        w3 = (i - 1) * exp(li + lj + log(j + 1) + 4 * lem1 - 2 * (log(λ) + θ) - log((i - j - 1)^2)) / (i - j - 2)
        w4 = (i * j / (i - j - 1)) * (2 / λ^2) * (1 - θ * (((i - 1) / j) * expm1(θ - log(2) - lθ) * ept + ((j + 1) / i) * (exp(-θ - log(2) - lθ) + 1) * emt))
        H[2, 2] = w1 + w3 + w4
      end
    else
      H[2, 2] = F(-Inf)
    end
  end

  ∇, Symmetric(H)
end

"""
    gradient_hessian_μ_zero_unstable(λ, i, j, t)

Functions are subject to catastrophic cancellation and therefore we will do
series expansion to approximate their values.
"""
function gradient_hessian_μ_zero_unstable(
  λ::F,
  i::I,
  j::I,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)
  H = fill(F(NaN), (2, 2))

  θ = λ * t

  if i == 1
    if j > 1
      y = - j * t * expm1(θ - log(j)) / expm1(θ)

      ∇[1] = y
      ∇[2] = - y - (2 - (j + 1) / exp(θ)) / λ

      w1 = - (j - 1) * exp(θ + 2 * (log(t) - logexpm1(θ)))

      H[1, 1] = w1

      w2 = begin
        x0 = (5 / 12) * j + 7 / 12
        x1 = (j + 1) / 3
        x2 = (31 / 240) * j + 29 / 240
        x3 = (j + 1) / 30
        x4 = (41 / 6048) * j + 43 / 6048
        x5 = (j + 1) / 840
        x6 = (31 / 172800) * j + 29 / 172800
        x7 = (j + 1) / 45360
        x8 = (61 / 26611200) * j + 71 / 26611200
        x9 = (j + 1) / 3991680

        t^2 * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
      end

      H[2, 1] = w2
      H[1, 2] = w2

      H[2, 2] = begin
        x0 = (j - 1) / 12
        x1 = (j + 1) / 3
        x2 = (79 / 240) * j + 27 / 80
        x3 = 11 * (j + 1) / 60
        x4 = (2189 / 30240) * j + 2179 / 30240
        x5 = 19 * (j + 1) / 840
        x6 = (7193 / 1209600) * j + 7207 / 1209600
        x7 = 247 * (j + 1) / 181440
        x8 = (22103 / 79833600) * j + 22073 / 79833600
        x9 = 1013 * (j + 1) / 19958400

        t^2 * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
      end
    elseif j == 0
      ∇[1] = - t * exp(θ) / expm1(θ)
      H[1, 1] = (t / expm1(θ))^2 * exp(θ)
      H[2, 2] = F(-Inf)
    elseif j == 1
      ∇[1] = -t
      ∇[2] = t + 2 * expm1(-θ) / λ

      H[1, 1] = zero(F)

      y = (log1p(θ) - θ) / 2

      w = -(2 / λ^2) * (exp(y) + 1) * expm1(y)
      H[2, 1] = w
      H[1, 2] = w

      H[2, 2] = begin
        x1 = 1 / 6
        x2 = 1 / 6
        x3 = 11 / 120
        x4 = 13 / 360
        x5 = 19 / 1680
        x6 = 1 / 336
        x7 = 247 / 362880
        x8 = 251 / 1814400
        x9 = 1013 / 39916800

        -4 * t^2 * θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9))))))))
      end
    end
  elseif j == 0
    u = i * t * (exp(θ) / expm1(θ))

    ∇[1] = -u

    H[1, 1] = exp(log(i) + 2 * (log(t) - logexpm1(θ)) + θ)

    if i > 1
      ∇[2] = -u * expm1(logexpm1(θ) - log(θ))

      w = begin
        x0 = 5 / 12
        x1 = 1 / 3
        x2 = 31 / 240
        x3 = 1 / 30
        x4 = 41 / 6048
        x5 = 1 / 840
        x6 = 31 / 172800
        x7 = 1 / 45360
        x8 = 61 / 26611200
        x9 = 1 / 3991680

        -i * t^2 * (x0 + θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9)))))))))
      end

      H[2, 1] = w
      H[1, 2] = w

      if i > 2
        H[2, 2] = begin
          x0 = 1 / 12
          x1 = 1 / 3
          x2 = 79 / 240
          x3 = 11 / 60
          x4 = 2189 / 30240
          x5 = 19 / 840
          x6 = 7193 / 1209600
          x7 = 247 / 181440
          x8 = 22103 / 79833600
          x9 = 1013 / 19958400

          -i * t^2 * (x0 + θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9)))))))))
        end
      end
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

    H[1, 1] = zero(F)

    w = begin
      x0 = i
      x1 = 2 / 3
      x2 = i / 4
      x3 = 1 / 15
      x4 = i / 72
      x5 = 1 / 420
      x6 = i / 2880
      x7 = 1 / 22680
      x8 = i / 201600
      x9 = 1 / 1995840

      i * t^2 * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
    end

    H[2, 1] = w
    H[1, 2] = w

    H[2, 2] = begin
      x1 = 2 / 3
      x2 = i * (3 * i^2 - 7) / 6
      x3 = 11 / 30
      x4 = i * (15 * i^2 - 41) / 180
      x5 = 19 / 420
      x6 = i * (21 * i^2 - 61) / 3360
      x7 = 247 / 90720
      x8 = i * (255 * i^2 - 757) / 907200
      x9 = 1013 / 9979200

      -i * t^2 * θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9))))))))
    end
  else
    ∇[1] = - j * t * expm1(θ + log(i) - log(j)) / expm1(θ)
    H[1, 1] = (i - j) * exp(θ + 2 * (log(t) - logexpm1(θ)))

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

      w2 = begin
        y1 = i * (i - 1) + j * (j + 1)
        y2 = 2 * i * j

        x0 = (5 * y1 + y2) / (12 * (i - j - 1))
        x1 = (i + j) / 3
        x2 = (31 * y1 - y2) / (240 * (i - j - 1))
        x3 = (i + j) / 30
        x4 = (41 * y1 + y2) / (6048 * (i - j - 1))
        x5 = (i + j) / 840
        x6 = (31 * y1 - y2) / (172800 * (i - j - 1))
        x7 = (i + j) / 45360
        x8 = (61 * y1 - y2) / (26611200 * (i - j - 1))
        x9 = (i + j) / 3991680

        -t^2 * (x0 + θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9)))))))))
      end

      H[2, 1] = w2
      H[1, 2] = w2

      if j != i - 2
        H[2, 2] = begin
          y1 = i * (i - 1)^2 * (i - 2) + j * (j + 2) * (j + 1)^2
          y2 = 2 * i * j
          y3 = (i - j - 1)^2 * (i - j - 2)

          x0 = (i - j) / 12
          x1 = (i + j) / 3
          x2 = (79 * y1 - y2 * (6 * (13 * (i + j)^2 - 19 * (i - j - 1)) - 113 * i * j + 1)) / (240 * y3)
          x3 = (i + j) * (11 / 60)
          x4 = (2189 * y1 - y2 * (2 * (1097 * (i + j)^2 - 2031 * (i - j - 1)) - 4067 * i * j - 5)) / (30240 * y3)
          x5 = (i + j) * (19 / 840)
          x6 = (7193 * y1 - y2 * (2 * (3593 * (i + j)^2 - 6999 * (i - j - 1)) - 13991 * i * j + 7)) / (1209600 * y3)
          x7 = (i + j) * (247 / 181440)
          x8 = (22103 * y1 - y2 * (2 * (11059 * (i + j)^2 - 21957 * (i - j - 1)) - 43929 * i * j - 15)) / (79833600 * y3)
          x9 = (i + j) * (1013 / 19958400)

          - t^2 * (x0 + θ * (x1 + θ * (x2 + θ * (x3 + θ * (x4 + θ * (x5 + θ * (x6 + θ * (x7 + θ * (x8 + θ * x9)))))))))
        end
      end
    else
      H[2, 2] = F(-Inf)
    end
  end

  ∇, Symmetric(H)
end
