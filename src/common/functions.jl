"""
    log1mexp(x)

Compute `log(1 - exp(x))` accurately. See
"**Accurately Computing log(1 - exp(.)) – Assessed by Rmpfr**" by Martin Mächler
(2012) for details:
https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function log1mexp(x::Real)
  if x > -log(2)
    log(-expm1(x))
  else
    log1p(-exp(x))
  end
end

"""
    logexpm1(x)

Compute `log(exp(x) - 1)` accurately. Thresholds were obtained similarly to
"**Accurately Computing log(1 - exp(.)) – Assessed by Rmpfr**" by Martin Mächler
(2012):
https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function logexpm1(x::Real)
  if x >= 37
    x
  elseif x >= 19
    x - exp(-x)
  else
    log(expm1(x))
  end
end

"""
    log1pexp(x)

Compute `log(1 + exp(x))` accurately. See
"**Accurately Computing log(1 - exp(.)) – Assessed by Rmpfr**" by Martin Mächler
(2012) for details:
https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function log1pexp(x::Real)
  if x <= -37
    exp(x)
  elseif x <= 18
    log1p(exp(x))
  elseif x <= 33.3
    x + exp(-x)
  else
    x
  end
end
