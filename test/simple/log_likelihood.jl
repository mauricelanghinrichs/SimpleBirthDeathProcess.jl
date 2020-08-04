@testset "Continuous time" begin
  η = [0.009, 0.002]

  t1 = [ 0.0, 0.22, 0.24, 0.33, 0.35, 0.48, 0.72, 0.91, 0.92, 0.95]
  x1 = [1000, 1001, 1002, 1003, 1002, 1003, 1004, 1005, 1004, 1005]
  s1 = 1

  B1 = sum(diff(x1) .== 1)
  D1 = sum(diff(x1) .== -1)
  I1 = sum(x1 .* [diff(t1); s1 - t1[end]])
  N1 = sum(log, x1[1:(end - 1)])

  t2 = [ 0.0, 0.02, 0.16, 0.20, 0.28, 0.41, 0.56, 0.63, 0.69, 0.70, 0.73, 0.87, 0.88, 0.91, 0.96, 1.56, 1.58, 1.72, 1.89, 1.95]
  x2 = [1000, 1001, 1002, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017]
  s2 = 2

  B2 = sum(diff(x2) .== 1)
  D2 = sum(diff(x2) .== -1)
  I2 = sum(x2 .* [diff(t2); s2 - t2[end]])
  N2 = sum(log, x2[1:(end - 1)])

  y1 = SimpleBirthDeathProcess.observation_continuous_time(t1, x1, s1)
  y2 = SimpleBirthDeathProcess.observation_continuous_time(t2, x2, s2)

  y3 = [y1, y2]

  @test loglik(η, y1) ≈ N1 + log(η[1]) * B1 + log(η[2]) * D1 - (η[1] + η[2]) * I1
  @test loglik(η, y2) ≈ N2 + log(η[1]) * B2 + log(η[2]) * D2 - (η[1] + η[2]) * I2
  @test loglik(η, y3) ≈ loglik(η, y1) + loglik(η, y2)
end

@testset "Discrete time (equidistant sampling)" begin
  η = [0.3, 0.5]

  u = 1
  x = [1000, 828, 710, 581, 426, 333]

  y = SimpleBirthDeathProcess.observation_discrete_time_equal(u, x)

end
