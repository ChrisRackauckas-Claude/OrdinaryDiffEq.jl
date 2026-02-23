using OrdinaryDiffEqBDF, Test, StaticArrays, JLArrays

@testset "Static Array Tests" begin
    # Linear decay ODE: du/dt = -0.5*u
    f_oop(u, p, t) = -0.5 * u
    u0_sv = SVector(1.0, 2.0)
    tspan = (0.0, 1.0)

    prob_sv = ODEProblem(f_oop, u0_sv, tspan)

    @testset "QNDF with SVector" begin
        sol = solve(prob_sv, QNDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa SVector
        @test isapprox(sol.u[end], exp(-0.5) * u0_sv, rtol = 1.0e-5)
    end

    @testset "QNDF1 with SVector" begin
        sol = solve(prob_sv, QNDF1(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa SVector
        @test isapprox(sol.u[end], exp(-0.5) * u0_sv, rtol = 1.0e-3)
    end

    @testset "QNDF2 with SVector" begin
        sol = solve(prob_sv, QNDF2(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa SVector
        @test isapprox(sol.u[end], exp(-0.5) * u0_sv, rtol = 1.0e-5)
    end

    @testset "FBDF with SVector" begin
        sol = solve(prob_sv, FBDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa SVector
        @test isapprox(sol.u[end], exp(-0.5) * u0_sv, rtol = 1.0e-5)
    end
end

@testset "Scalar Tests" begin
    f_scalar(u, p, t) = -0.5 * u
    prob_scalar = ODEProblem(f_scalar, 1.0, (0.0, 1.0))

    @testset "QNDF scalar" begin
        sol = solve(prob_scalar, QNDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa Number
        @test isapprox(sol.u[end], exp(-0.5), rtol = 1.0e-5)
    end

    @testset "FBDF scalar" begin
        sol = solve(prob_scalar, FBDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol.u[end] isa Number
        @test isapprox(sol.u[end], exp(-0.5), rtol = 1.0e-5)
    end
end

@testset "GPU (JLArray) Tests" begin
    # Use broadcast operations to avoid scalar indexing
    function f_gpu!(du, u, p, t)
        @. du = p * u
        return nothing
    end

    u0_cpu = [1.0, 2.0]
    p_cpu = [-0.5, -1.5]
    u0_gpu = jl(u0_cpu)
    p_gpu = jl(p_cpu)
    tspan = (0.0, 1.0)

    prob_cpu = ODEProblem(f_gpu!, copy(u0_cpu), tspan, p_cpu)
    prob_gpu = ODEProblem(f_gpu!, copy(u0_gpu), tspan, p_gpu)

    @testset "QNDF GPU" begin
        sol_cpu = solve(prob_cpu, QNDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        sol_gpu = solve(prob_gpu, QNDF(), abstol = 1.0e-8, reltol = 1.0e-8)
        @test sol_gpu.u[end] isa JLArray
        @test isapprox(Array(sol_gpu.u[end]), sol_cpu.u[end], rtol = 1.0e-5)
    end

    # FBDF GPU requires fixing scalar indexing in bdf_interpolants.jl (_get_theta)
    # which is outside the scope of this refactoring.
    @testset "FBDF GPU" begin
        @test_broken begin
            sol_gpu = solve(prob_gpu, FBDF(), abstol = 1.0e-8, reltol = 1.0e-8)
            sol_cpu = solve(prob_cpu, FBDF(), abstol = 1.0e-8, reltol = 1.0e-8)
            isapprox(Array(sol_gpu.u[end]), sol_cpu.u[end], rtol = 1.0e-5)
        end
    end

    @testset "QNDF1 GPU" begin
        sol_cpu = solve(
            ODEProblem(f_gpu!, copy(u0_cpu), tspan, p_cpu),
            QNDF1(), abstol = 1.0e-8, reltol = 1.0e-8
        )
        sol_gpu = solve(
            ODEProblem(f_gpu!, copy(u0_gpu), tspan, p_gpu),
            QNDF1(), abstol = 1.0e-8, reltol = 1.0e-8
        )
        @test sol_gpu.u[end] isa JLArray
        @test isapprox(Array(sol_gpu.u[end]), sol_cpu.u[end], rtol = 1.0e-5)
    end

    @testset "QNDF2 GPU" begin
        sol_cpu = solve(
            ODEProblem(f_gpu!, copy(u0_cpu), tspan, p_cpu),
            QNDF2(), abstol = 1.0e-8, reltol = 1.0e-8
        )
        sol_gpu = solve(
            ODEProblem(f_gpu!, copy(u0_gpu), tspan, p_gpu),
            QNDF2(), abstol = 1.0e-8, reltol = 1.0e-8
        )
        @test sol_gpu.u[end] isa JLArray
        @test isapprox(Array(sol_gpu.u[end]), sol_cpu.u[end], rtol = 1.0e-5)
    end
end
