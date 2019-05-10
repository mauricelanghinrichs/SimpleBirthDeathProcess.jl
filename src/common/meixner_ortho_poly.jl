"""
  log_meixner_ortho_poly(a, b, x)

Compute the logarithm of the Meixner polynomial
``M_{b}(a; -(a + b - 1), exp(x))`` where `a` and `b` are positive integers, with
`a >= b`.

Define
``u_{n} = 1 + (1 + a (1 - exp(x)) - n) / ((c + 1 - n) exp(x))``
and
``v_{n} = (n - 1) / ((c + 1 - n) exp(x))``

If ``y_{n}`` represents the Meixner polynomial of degree `n`, then
``y_{n} = u_{n} y_{n - 1} + v_{n} y_{n - 2}``
where ``y_{0} = 1`` and ``y_{1} = 1 + a * (1 - exp(x)) / (c * exp(x))``
"""
function log_meixner_ortho_poly(
  a::F,
  b::F,
  x::F
)::F where {
  F <: AbstractFloat
}
  lom = log1mexp(-x)
  lem = logexpm1(x)

  # define H_{b} = y_{b} / y_{b - 1}, i. e. y_{b} = H_{b} * y_{b - 1}
  # y_{0} = 1, y_{1} = 1 + a * (1 - x) / ((a + b - 1) * x)
  tmp = log1mexp(log(a / (a + b - 1)) + lom)

  # v = [log(H), log(y), compensation_error]
  v = [tmp, tmp, F(0)]

  n = F(1)
  while n < b
    n += 1
    kbn_meixner_ortho_poly_iteration!(v, a, n, a + b - 1, lom, lem)
  end

  # finally add the compensation error
  v[2] + v[3]
end

"""
  log_meixner_ortho_poly_joint(a, b, x)

Compute the logarithm of the Meixner polynomials
``M_{b}(a; -(a + b - 1), exp(x))``,
``M_{b - 1}(a - 1; -(a + b - 2), exp(x))``, and
``M_{b - 2}(a - 2; -(a + b - 3), exp(x))`` at the same time. `a` and `b` are
positive integers, `a >= b`. We assume that `a` and `b` are both greater than 1.
"""
function log_meixner_ortho_poly_joint(
  a::F,
  b::F,
  x::F
)::Tuple{F, F, F} where {
  F <: AbstractFloat
}
  lom = log1mexp(-x)
  lem = logexpm1(x)

  # define H_{b} = y_{b} / y_{b - 1}, i. e. y_{b} = H_{b} * y_{b - 1}
  # y_{0} = 1, y_{1} = 1 + a * (1 - x) / (c * x)

  # M_{b}(a; -(a + b - 1), x)
  c1 = a + b - 1
  tmp = log1mexp(log(a / c1) + lom)
  v1 = [tmp, tmp, F(0)]

  # M_{b - 1}(a - 1; -(a + b - 2), x)
  c2 = a + b - 2
  tmp = log1mexp(log((a - 1) / c2) + lom)
  v2 = [tmp, tmp, F(0)]

  if b > 2
    # M_{b - 2}(a - 2; -(a + b - 3), x)
    c3 = a + b - 3
    tmp = log1mexp(log((a - 2) / c3) + lom)
    v3 = [tmp, tmp, F(0)]

    n = F(1)
    while n < b - 2
      n += 1
      kbn_meixner_ortho_poly_iteration!(v1, a, n, c1, lom, lem)
      kbn_meixner_ortho_poly_iteration!(v2, a - 1, n, c2, lom, lem)
      kbn_meixner_ortho_poly_iteration!(v3, a - 2, n, c3, lom, lem)
    end

    # We still need few iterations
    kbn_meixner_ortho_poly_iteration!(v1, a, b - 1, c1, lom, lem)
    kbn_meixner_ortho_poly_iteration!(v2, a - 1, b - 1, c2, lom, lem)

    kbn_meixner_ortho_poly_iteration!(v1, a, b, c1, lom, lem)

    # add compensation errors and return the final values
    v1[2] + v1[3], v2[2] + v2[3], v3[2] + v3[3]
  else
    # M_{2}(a; -(a + 1), x)
    # M_{1}(a - 1; -(a + 2), x)
    # M_{0}(a - 2; -(a + 3), x)
    kbn_meixner_ortho_poly_iteration!(v1, a, F(2), c1, lom, lem)
    v1[2], v2[2], 0
  end
end

"""
    kbn_meixner_ortho_poly_iteration!(v, a, n, c, x)

Basic Kahan iteration operations for computing Meixner orthogonal polynomial.
"""
@inline function kbn_meixner_ortho_poly_iteration!(
  v::Vector{F},
  a::F,
  n::F,
  c::F,
  # lom = log(1 - exp(-x))
  lom::F,
  # lem = log(exp(x) - 1)
  lem::F
) where {
  F <: AbstractFloat
}
  # new value for log(y_{n} / y_{n - 1})
  v[1] = log1mexp(log(a / (c + 1 - n)) + lom +
                  log1mexp(log((n - 1) / a) + logexpm1(-v[1]) - lem))

  # running sum including error
  s = v[2] + v[1]

  # Neumaier modification to Kahan summation
  v[3] += if abs(v[2]) >= abs(v[1])
    (v[1] - s) + v[2]
  else
    (v[2] - s) + v[1]
  end

  v[2] = s

  nothing
end
