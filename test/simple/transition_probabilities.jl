@testset "Exceptions" begin
  # initial population size is negative
  @test_throws DomainError trans_prob(-1, 20, 72.0, [0.1, 0.05])

  # final population size is negative
  @test_throws DomainError trans_prob(10, -1, 72.0, [0.1,  0.05])

  # time is negative
  @test_throws DomainError trans_prob(10, 20, -1.0, [0.1,  0.05])

  # birth rate is negative
  @test_throws DomainError trans_prob(10, 20, 72.0, [-0.1, 0.05])

  # death rate is negative
  @test_throws DomainError trans_prob(10, 20,  1.0, [0.1, -0.05])
end

@testset "Type stability" begin
  @testset for I = [Int8, Int16, Int32, Int64, Int128], F = [Float32, Float64, BigFloat]
    # parameters are zero
    @test @inferred(trans_prob(I(1), I(1), F(0), [F(1), F(1)])) == F(0)
    @test typeof(trans_prob(I(1), I(1), F(0), [F(1), F(1)])) === F

    @test @inferred(trans_prob(I(1), I(1), F(1), [F(0), F(0)])) == F(0)
    @test typeof(trans_prob(I(1), I(1), F(1), [F(0), F(0)])) === F

    @test @inferred(trans_prob(I(1), I(2), F(0), [F(1), F(1)])) == F(-Inf)
    @test typeof(trans_prob(I(1), I(2), F(0), [F(1), F(1)])) === F

    @test @inferred(trans_prob(I(1), I(2), F(1), [F(0), F(0)])) == F(-Inf)
    @test typeof(trans_prob(I(1), I(2), F(1), [F(0), F(0)])) === F

    # equal rates
    # t = 1 / λ, j > 0
    g = trans_prob(I(8), I(4), F(0.5), [F(2), F(2)])
    @test @inferred(trans_prob(I(8), I(4), F(0.5), [F(2), F(2)])) == g
    @test typeof(g) === F

    # t = 1 / λ, j = 0
    g = trans_prob(I(8), I(0), F(0.5), [F(2), F(2)])
    @test @inferred(trans_prob(I(8), I(0), F(0.5), [F(2), F(2)])) == g
    @test typeof(g) === F

    # t != 1 / λ, j = 0
    g = trans_prob(I(8), I(0), F(1), [F(2), F(2)])
    @test @inferred(trans_prob(I(8), I(0), F(1), [F(2), F(2)])) == g
    @test typeof(g) === F

    # t > 1 / λ, j > 0
    g = trans_prob(I(8), I(4), F(1), [F(2), F(2)])
    @test @inferred(trans_prob(I(8), I(4), F(1), [F(2), F(2)])) == g
    @test typeof(g) === F

    # t < 1 / λ, j > 0
    g = trans_prob(I(8), I(4), F(0.1), [F(2), F(2)])
    @test @inferred(trans_prob(I(8), I(4), F(0.1), [F(2), F(2)])) == g
    @test typeof(g) === F

    # pure birth process
    g = trans_prob(I(4), I(8), F(1), [F(2), F(0)])
    @test @inferred(trans_prob(I(4), I(8), F(1), [F(2), F(0)])) == g
    @test typeof(g) === F

    @test @inferred(trans_prob(I(8), I(4), F(1), [F(2), F(0)])) == F(-Inf)
    @test typeof(trans_prob(I(8), I(4), F(1), [F(2), F(0)])) === F

    # pure death process
    g = trans_prob(I(8), I(4), F(1), [F(0), F(2)])
    @test @inferred(trans_prob(I(8), I(4), F(1), [F(0), F(2)])) == g
    @test typeof(g) === F

    @test @inferred(trans_prob(I(4), I(8), F(1), [F(0), F(2)])) == F(-Inf)
    @test typeof(trans_prob(I(4), I(8), F(1), [F(0), F(2)])) === F

    # unequal rates
    # t = log(λ / μ) / (λ - μ), j > 0
    g = trans_prob(I(4), I(8), F(log(2)), [F(2), F(1)])
    @test @inferred(trans_prob(I(4), I(8), F(log(2)), [F(2), F(1)])) == g
    @test typeof(g) === F

    # t = log(λ / μ) / (λ - μ), j = 0
    g = trans_prob(I(4), I(0), F(log(2)), [F(2), F(1)])
    @test @inferred(trans_prob(I(4), I(0), F(log(2)), [F(2), F(1)])) == g
    @test typeof(g) === F

    # t != log(λ / μ) / (λ - μ), j = 0
    g = trans_prob(I(4), I(0), F(1), [F(2), F(1)])
    @test @inferred(trans_prob(I(4), I(0), F(1), [F(2), F(1)])) == g
    @test typeof(g) === F

    # t > log(λ / μ) / (λ - μ), j > 0
    g = trans_prob(I(4), I(8), F(1), [F(2), F(1)])
    @test @inferred(trans_prob(I(4), I(8), F(1), [F(2), F(1)])) == g
    @test typeof(g) === F

    # t < log(λ / μ) / (λ - μ), j > 0
    g = trans_prob(I(4), I(8), F(0.5), [F(2), F(1)])
    @test @inferred(trans_prob(I(4), I(8), F(0.5), [F(2), F(1)])) == g
    @test typeof(g) === F
  end
end

@testset "Time is zero" begin
  @test trans_prob(10, 10, 0.0, [0.1, 0.05]) === 0.0
  @test trans_prob(10,  5, 0.0, [0.1, 0.05]) === -Inf
  @test trans_prob(10, 15, 0.0, [0.1, 0.05]) === -Inf
end

@testset "Both parameters are zero" begin
  @test trans_prob(10, 10, 72.0, [0.0, 0.0]) === 0.0
  @test trans_prob(10,  5, 72.0, [0.0, 0.0]) === -Inf
  @test trans_prob(10, 15, 72.0, [0.0, 0.0]) === -Inf
end

@testset "Pure birth process" begin
  t = 1.0
  η = [1.5, 0.0]

  @test trans_prob(1, 0, t, η) === -Inf
  @test trans_prob(100, 50, t, η) === -Inf

  p = [trans_prob(100, j, t, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0
end

@testset "Pure death process" begin
  t = 1.0
  η = [0.0, 1.5]

  @test trans_prob(1, 2, t, η) === -Inf
  @test trans_prob(100, 150, t, η) === -Inf

  p = [trans_prob(10000, j, t, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0
end

@testset "Equal rates" begin
  i = 1000
  η = [0.1, 0.1]

  # addends are all positive
  p = [trans_prob(i, j, 5.0, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # addends have alternating signs
  p = [trans_prob(i, j, 100.0, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # time is exactly equal to 1 / λ
  p = [trans_prob(i, j, 10.0, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0
end

@testset "Death rate greater than birth rate" begin
  i = 1000
  η = [0.01, 0.02]
  ω = log(η[1] / η[2]) / (η[1] - η[2])

  # addends are all positive
  p = [trans_prob(i, j, ω * 0.5, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # addends have alternating signs
  p = [trans_prob(i, j, ω * 2.0, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # time is exactly equal to log(λ / μ) / (λ - μ)
  p = [trans_prob(i, j, ω, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0
end

@testset "Birth rate greater than death rate" begin
  i = 1000
  η = [0.02, 0.01]
  ω = log(η[1] / η[2]) / (η[1] - η[2])

  # addends are all positive
  p = [trans_prob(i, j, ω * 0.5, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # addends have alternating signs
  p = [trans_prob(i, j, ω * 2.0, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0

  # time is exactly equal to log(λ / μ) / (λ - μ)
  p = [trans_prob(i, j, ω, η, log_value=false) for j = 0:10000]
  @test sum(p) ≈ 1.0
end
