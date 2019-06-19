@testset "Exceptions" begin
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, 0.1, 0.2], [10, 9], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.1, 0.2], [10, 9], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, 0.1], [10, -9], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, -0.1], [10, 9], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.1, 0.0], [9, 10], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, 0.1], [10, 9], 0.05)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, 0.1, 0.1], [10, 9, 8], 1.0)
  @test_throws ErrorException SimpleBirthDeathProcess.observation_continuous_time([0.0, 0.1, 0.2], [10, 9, 7], 1.0)

  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_equal(-1, [10, 5])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_equal(1, [10, -5])

  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_equal(-1, [10 100; 5 20])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_equal(1, [10 100; -5 20])

  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.0, 0.1, 0.2], [10, 5])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.1, 0.2], [10, 5])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.0, 0.1], [10, -5])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.0, -0.1], [10, 5])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.1, 0.0], [5, 10])
  @test_throws ErrorException SimpleBirthDeathProcess.observation_discrete_time_unequal([0.0, 0.1, 0.1], [10, 5, 4])
end

@testset "Constructor: Continuous Time" begin
  t = [0.0, 0.22, 0.3, 0.57, 0.63, 0.67, 0.98]
  x = [1000, 999, 1000, 999, 998, 999, 1000]
  T = 1

  y = SimpleBirthDeathProcess.observation_continuous_time(t, x, T)

  @test isa(y, SimpleBirthDeathProcess.ObservationContinuousTime)
  @test y.tot_births == 3
  @test y.tot_deaths == 3
  @test y.integrated_jump ≈ sum(x[1:(end - 1)] .* diff(t)) + (x[end] * (T - t[end]))
  @test y.sum_log_n ≈ sum(log.(x[1:(end - 1)]))
  @test y.len == 6
  @test all(y.waiting_time .≈ diff(t))
  @test y.initial_population_size == x[1]
  @test all(y.increment .== diff(x))
end

@testset "Constructor: Discrete Time (equidistant sampling)" begin
  u = 1
  x = [1000, 828, 710, 581, 426, 333]

  y = SimpleBirthDeathProcess.observation_discrete_time_equal(u, x)

  @test isa(y, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual)
  @test y.n == 1
  @test y.k == 6
  @test y.u == 1
  @test isa(y.state, Matrix{Int})
  @test y.state == reshape(x, 6, 1)

  u = 1
  x = [1000 2000;
        828 1614;
        710 1331;
        581 1090;
        426  873;
        333  708]

  y = SimpleBirthDeathProcess.observation_discrete_time_equal(u, x)

  @test isa(y, SimpleBirthDeathProcess.ObservationDiscreteTimeEqual)
  @test y.n == 2
  @test y.k == 6
  @test y.u == 1
  @test isa(y.state, Matrix{Int})
  @test y.state == x
end

@testset "Constructor: Discrete Time (unequally spaced time points)" begin
  t = [ 0.0, 1.0, 1.5, 3.0, 3.5, 4.0]
  x = [1000, 828, 710, 581, 426, 333]

  y = SimpleBirthDeathProcess.observation_discrete_time_unequal(t, x)

  @test isa(y, SimpleBirthDeathProcess.ObservationDiscreteTimeUnequal)
  @test all(y.waiting_time .≈ diff(t))
  @test all(y.state .== x)
end
