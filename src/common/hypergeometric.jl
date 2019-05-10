"""
    log_hypergeometric(a, b, x)

Compute the logarithm of the hypergeometric function
``{}_{2}F_{1}(-a, -b; -(a + b - 1); -exp(x))`` where `a` and `b` are positive
integers, and `a >= b`.
"""
function log_hypergeometric(
  a::F,
  b::F,
  x::F
)::F where {
  F <: AbstractFloat
}
  if b > 1
    # define H_{b} = y_{b} / y_{b - 1}, i. e. y_{b} = H_{b} * y_{b - 1}
    # y_{0} = 1, y_{1} = 1 + (a * z) / (a + 1 - k)
    tmp = log1pexp(x)

    # v = [log(H), log(y), compensation_error]
    v = [tmp, tmp, F(0)]

    n = F(1)
    while n < b
      n += 1
      kbn_2F1_iteration!(v, n, a, x, F(1))
    end

    # finally add the compensation error
    v[2] + v[3]
  elseif b == 1
    # 2F1(-a, -1; -a, -z) = 1 + z
    log1pexp(x)
  else
    0
  end
end

"""
    log_hypergeometric_joint(a, b, x)

Compute the logarithm of the three hypergeometric functions
``{}_{2}F_{1}(-a, -b; -(a + b - 1); -exp(x))``,
``{}_{2}F_{1}(-(a - 1), -(b - 1); -(a + b); -exp(x))``, and
``{}_{2}F_{1}(-(a - 2), -(b - 2); -(a + b + 1); -exp(x))`` at the same time.
`a` and `b` are positive integers with `a >= b`. We assume that `a` and `b` are
both greater than 1.
"""
function log_hypergeometric_joint(
  a::F,
  b::F,
  x::F
)::Tuple{F, F, F} where {
  F <: AbstractFloat
}
  # define H_{b} = y_{b} / y_{b - 1}, i. e. y_{b} = H_{b} * y_{b - 1}
  # y_{0} = 1, y_{1} = 1 + (a * z) / (a + 1 - k)

  # k = 1
  # 2F1(-a, -b; -(a + b - 1); -z)
  tmp = log1pexp(x)
  v1 = [tmp, tmp, F(0)]

  # k = 0
  # 2F1(-(a - 1), -(b - 1); -(a + b); -z)
  tmp = log1pexp(log((a - 1) / a) + x)
  v2 = [tmp, tmp, F(0)]

  if b > 2
    # k = -1
    # 2F1(-(a - 2), -(b - 2); -(a + b + 1); -z)
    tmp = log1pexp(log((a - 2) / a) + x)
    v3 = [tmp, tmp, F(0)]

    n = F(1)
    while n < b - 2
      n += 1
      kbn_2F1_iteration!(v1, n, a, x, F(1))
      kbn_2F1_iteration!(v2, n, a - 1, x, F(0))
      kbn_2F1_iteration!(v3, n, a - 2, x, F(-1))
    end

    # We still need few iterations
    kbn_2F1_iteration!(v1, b - 1, a, x, F(1))
    kbn_2F1_iteration!(v2, b - 1, a - 1, x, F(0))

    kbn_2F1_iteration!(v1, b, a, x, F(1))

    # add compensation errors and return the final values
    v1[2] + v1[3], v2[2] + v2[3], v3[2] + v3[3]
  else
    # log(2F1(-a, -2; -a; -z)) = v1[2] + H
    # log(2F1(-(a - 1), -1; -(a + 2); -z)) = v2[2]
    # log(2F1(-(a - 2), 0; -(a + 3); -z)) = 0
    kbn_2F1_iteration!(v1, F(2), a, x, F(1))
    v1[2], v2[2], 0
  end
end

"""
    kbn_2F1_iteration!(v, n, a, x, k)

Basic Kahan iteration operations for computing
``{}_{2}F_{1}(-a, -b; -(a + b - k); -exp(x))``.
"""
@inline function kbn_2F1_iteration!(
  v::Vector{F},
  n::F,
  a::F,
  x::F,
  k::F
) where {
  F <: AbstractFloat
}
  # new value for log(y_{b} / y_{b - 1})
  v[1] = log1pexp(x + log((a - n + 1) / (a + n - k)) +
                  log1pexp(log((n - 1) * (n - 1 - k) /
                               ((a - n + 1) * (a + n - k - 1))) - v[1]))

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
