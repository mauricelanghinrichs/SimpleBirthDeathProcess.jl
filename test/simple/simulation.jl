@testset "Exceptions" begin
  # number of simulations is negative
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(-1, 1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(-1, 1, 1, 1, [1, 1])

  # number of simulations is zero
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(0, 1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(0, 1, 1, 1, [1, 1])

  # initial population size is zero or negative
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, -1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1,  0, 1, [1, 1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(-1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous( 0, 1, [1, 1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, -1, 1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1,  0, 1, 1, [1, 1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(-1, 1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete( 0, 1, 1, [1, 1])

  # observation time is negative (continuous)
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, -1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, -1, [1, 1])

  # observation time is negative (discrete)
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, -1, 1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, -1, 1, [1, 1])

  # time lag is negative or zero (discrete)
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, -1, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, -1, [1, 1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, 0, [1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 0, [1, 1])

  # birth rate is negative
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, 1, [-1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, [-1, 1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, 1, [-1, 1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, [-1, 1])

  # death rate is negative
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, 1, [1, -1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, [1, -1])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, 1, [1, -1])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, [1, -1])

  # birth rate and death rate both equal to zero
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, 1, [0, 0])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_continuous(1, 1, [0, 0])

  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, 1, [0, 0])
  @test_throws ErrorException SimpleBirthDeathProcess.rand_discrete(1, 1, 1, [0, 0])
end

@testset "Type stability" begin
  @testset for I = [Int8, Int16, Int32, Int64, Int128], F = [Float32, Float64, BigFloat]
    # all operations happen with the same type passed to the function...
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(2), I(100), I(1), [F(1), F(1)]), Vector{SimpleBirthDeathProcess.ObservationContinuousTime{F}})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(2), I(100), F(1), [F(1), F(1)]), Vector{SimpleBirthDeathProcess.ObservationContinuousTime{F}})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(100), I(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationContinuousTime{F})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(100), F(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationContinuousTime{F})

    @test isa(SimpleBirthDeathProcess.rand_discrete(I(2), I(100), I(1), I(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(2), I(100), I(1), F(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), I(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), F(1), [F(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})

    # ...but if parameters are passed as integers, operations use Float64
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(2), I(100), I(1), [I(1), I(1)]), Vector{SimpleBirthDeathProcess.ObservationContinuousTime{Float64}})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(2), I(100), I(1), [F(1), I(1)]), Vector{SimpleBirthDeathProcess.ObservationContinuousTime{F}})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(2), I(100), I(1), [I(1), F(1)]), Vector{SimpleBirthDeathProcess.ObservationContinuousTime{F}})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(100), I(1), [I(1), I(1)]), SimpleBirthDeathProcess.ObservationContinuousTime{Float64})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(100), I(1), [F(1), I(1)]), SimpleBirthDeathProcess.ObservationContinuousTime{F})
    @test isa(SimpleBirthDeathProcess.rand_continuous(I(100), I(1), [I(1), F(1)]), SimpleBirthDeathProcess.ObservationContinuousTime{F})

    # simulations happen as Float64 but we only see the realizations as integer counts
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(2), I(100), I(1), I(1), [I(1), I(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{Float64, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), I(1), [F(1), I(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), I(1), [I(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})

    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), F(1), [I(1), I(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), F(1), [F(1), I(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
    @test isa(SimpleBirthDeathProcess.rand_discrete(I(100), I(1), F(1), [I(1), F(1)]), SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
  end
end

@testset "Observation time is zero" begin
  I = Int32
  F = Float32

  x = SimpleBirthDeathProcess.rand_continuous(I(2), I(100), I(0), [I(1), I(1)])
  @test isa(x, Vector{SimpleBirthDeathProcess.ObservationContinuousTime{Float64}})
  @test length(x) == 2
  @test x[1].tot_births === zero(Int)
  @test x[1].tot_deaths === zero(Int)
  @test x[1].integrated_jump === zero(Float64)
  @test x[1].sum_log_n === zero(Float64)
  @test x[1].len === zero(Int)
  @test x[1].waiting_time == Float64[]
  @test x[1].initial_population_size == Int(100)
  @test x[1].increment == Int[]
  @test x[2].tot_births === zero(Int)
  @test x[2].tot_deaths === zero(Int)
  @test x[2].integrated_jump === zero(Float64)
  @test x[2].sum_log_n === zero(Float64)
  @test x[2].len === zero(Int)
  @test x[2].waiting_time == Float64[]
  @test x[2].initial_population_size == Int(100)
  @test x[2].increment == Int[]

  x = SimpleBirthDeathProcess.rand_continuous(I(2), I(100), F(0), [F(1), F(1)])
  @test isa(x, Vector{SimpleBirthDeathProcess.ObservationContinuousTime{F}})
  @test length(x) == 2
  @test x[1].tot_births === zero(Int)
  @test x[1].tot_deaths === zero(Int)
  @test x[1].integrated_jump === zero(F)
  @test x[1].sum_log_n === zero(Float64)
  @test x[1].len === zero(Int)
  @test x[1].waiting_time == F[]
  @test x[1].initial_population_size == Int(100)
  @test x[1].increment == Int[]
  @test x[2].tot_births === zero(Int)
  @test x[2].tot_deaths === zero(Int)
  @test x[2].integrated_jump === zero(F)
  @test x[2].sum_log_n === zero(Float64)
  @test x[2].len === zero(Int)
  @test x[2].waiting_time == F[]
  @test x[2].initial_population_size == Int(100)
  @test x[2].increment == Int[]

  x = SimpleBirthDeathProcess.rand_continuous(I(100), I(0), [I(1), I(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationContinuousTime{Float64})
  @test x.tot_births === zero(Int)
  @test x.tot_deaths === zero(Int)
  @test x.integrated_jump === zero(Float64)
  @test x.sum_log_n === zero(Float64)
  @test x.len === zero(Int)
  @test x.waiting_time == Float64[]
  @test x.initial_population_size == Int(100)
  @test x.increment == Int[]

  x = SimpleBirthDeathProcess.rand_continuous(I(100), F(0), [F(1), F(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationContinuousTime{F})
  @test x.tot_births === zero(Int)
  @test x.tot_deaths === zero(Int)
  @test x.integrated_jump === zero(F)
  @test x.sum_log_n === zero(Float64)
  @test x.len === zero(Int)
  @test x.waiting_time == F[]
  @test x.initial_population_size == Int(100)
  @test x.increment == Int[]

  x = SimpleBirthDeathProcess.rand_discrete(I(2), I(100), I(0), I(1), [I(1), I(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{Float64, I})
  @test x.n == Int(2)
  @test x.k == zero(Int)
  @test x.u == one(Float64)
  @test size(x.state) == (1, 2)
  @test x.state[1, 1] == I(100)
  @test x.state[1, 2] == I(100)

  x = SimpleBirthDeathProcess.rand_discrete(I(2), I(100), I(0), F(1), [F(1), F(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
  @test x.n == Int(2)
  @test x.k == zero(Int)
  @test x.u == one(F)
  @test size(x.state) == (1, 2)
  @test x.state[1, 1] == I(100)
  @test x.state[1, 2] == I(100)

  x = SimpleBirthDeathProcess.rand_discrete(I(100), I(0), I(1), [I(1), I(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{Float64, I})
  @test x.n == one(Int)
  @test x.k == zero(Int)
  @test x.u == one(Float64)
  @test size(x.state) == (1, 1)
  @test x.state[1, 1] == I(100)

  x = SimpleBirthDeathProcess.rand_discrete(I(100), I(0), F(1), [F(1), F(1)])
  @test isa(x, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{F, I})
  @test x.n == one(Int)
  @test x.k == zero(Int)
  @test x.u == one(F)
  @test size(x.state) == (1, 1)
  @test x.state[1, 1] == I(100)
end

@testset "Correct output" begin
  # we check if the expected value and variance are close to the theoretical one
  n = 100_000
  i = 10_000
  t = 10.0
  k = 10
  u = 1.0
  λ = 0.001
  μ = 0.002

  θ = (λ - μ) * t
  e = exp(θ)
  v = (λ + μ) * e * expm1(θ) / (i * (λ - μ))

  x = SimpleBirthDeathProcess.rand_continuous(n, i, t, [λ, μ])
  @test isa(x, Vector{SimpleBirthDeathProcess.ObservationContinuousTime{Float64}})
  @test abs(mean([(1 + sum(x[s].increment) / i) for s = 1:n]) - e) < 1e-5
  @test abs(var([(1 + sum(x[s].increment) / i) for s = 1:n]) - v) < 1e-5

  x = SimpleBirthDeathProcess.rand_discrete(n, i, k, u, [λ, μ])
  @test isa(x, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual{Float64, Int64})
  @test size(x.state) == (k + 1, n)
  @test abs(mean(x.state[end, :] ./ i) - e) < 1e-5
  @test abs(var(x.state[end, :] ./ i) - v) < 1e-5
end
