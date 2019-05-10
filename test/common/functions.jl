@testset "Exceptions" begin
  @test_throws DomainError SimpleBirthDeathProcess.log1mexp(0.01)
  @test_throws DomainError SimpleBirthDeathProcess.logexpm1(-0.01)
end

@testset "Type stability (log1mexp)" begin
  @testset for F = [Float32, Float64, BigFloat]
    @test isa(SimpleBirthDeathProcess.log1mexp(F(-1)), F)
    @test isa(SimpleBirthDeathProcess.log1mexp(F(-0.5)), F)
    @test isa(SimpleBirthDeathProcess.log1mexp(F(0)), F)
  end
end

@testset "Type stability (logexpm1)" begin
  @testset for F = [Float32, Float64, BigFloat]
    @test isa(SimpleBirthDeathProcess.log1pexp(F(40)), F)
    @test isa(SimpleBirthDeathProcess.log1pexp(F(20)), F)
    @test isa(SimpleBirthDeathProcess.log1pexp(F(10)), F)
  end
end

@testset "Type stability (log1pexp)" begin
  @testset for F = [Float32, Float64, BigFloat]
    @test isa(SimpleBirthDeathProcess.log1pexp(F(-40)), F)
    @test isa(SimpleBirthDeathProcess.log1pexp(F(-10)), F)
    @test isa(SimpleBirthDeathProcess.log1pexp(F(10)), F)
    @test isa(SimpleBirthDeathProcess.log1pexp(F(40)), F)
  end
end

@testset "Accuracy (log1mexp)" begin
  @test SimpleBirthDeathProcess.log1mexp(-10.0) ≈ -0.45400960370489209504e-4
  @test SimpleBirthDeathProcess.log1mexp(-1.0) ≈ -0.45867514538708189103
  @test SimpleBirthDeathProcess.log1mexp(-0.7) ≈ -0.68634100280838510968
  @test SimpleBirthDeathProcess.log1mexp(-0.69) ≈ -0.69630429714405665212
  @test SimpleBirthDeathProcess.log1mexp(-0.01) ≈ -4.6101660193248969177
end

@testset "Accuracy (logexpm1)" begin
  @test SimpleBirthDeathProcess.logexpm1(0.5) ≈ -0.432752129567188571894641000155
  @test SimpleBirthDeathProcess.logexpm1(1.0) ≈ 0.541324854612918108978356354931
  @test SimpleBirthDeathProcess.logexpm1(10.0) ≈ 9.99995459903962951079049555364
  @test SimpleBirthDeathProcess.logexpm1(19.0) ≈ 18.9999999943972035467670684411
  @test SimpleBirthDeathProcess.logexpm1(20.0) ≈ 19.9999999979388463754372650415
  @test SimpleBirthDeathProcess.logexpm1(30.0) ≈ 29.9999999999999064237703115939
  @test SimpleBirthDeathProcess.logexpm1(37.0) ≈ 36.9999999999999999146695237426
  @test SimpleBirthDeathProcess.logexpm1(40.0) ≈ 39.9999999999999999957516457447
  @test SimpleBirthDeathProcess.logexpm1(50.0) ≈ 49.9999999999999999999998071250
end

@testset "Accuracy (log1pexp)" begin
  @test SimpleBirthDeathProcess.log1pexp(-40.0) ≈ 4.24835425528999999097574306078e-18
  @test SimpleBirthDeathProcess.log1pexp(-37.0) ≈ 8.53304762574399963593549108392e-17
  @test SimpleBirthDeathProcess.log1pexp(-10.0) ≈ 0.453988992168646467694878337464e-4
  @test SimpleBirthDeathProcess.log1pexp(0.0) ≈ 0.693147180559945309417232121458
  @test SimpleBirthDeathProcess.log1pexp(10.0) ≈ 10.0000453988992168646467694878
  @test SimpleBirthDeathProcess.log1pexp(18.0) ≈ 18.0000000152299796287364881015
  @test SimpleBirthDeathProcess.log1pexp(20.0) ≈ 20.0000000020611536203143807032
  @test SimpleBirthDeathProcess.log1pexp(30.0) ≈ 30.0000000000000935762296883974
  @test SimpleBirthDeathProcess.log1pexp(40.0) ≈ 40.0000000000000000042483542553
end
