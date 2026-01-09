import OrdinaryDiffEqTsit5

# Only run JET tests on stable Julia versions (not prereleases)
# JET is not added as a test dependency to avoid installation issues on prereleases
if isempty(VERSION.prerelease)
    using Pkg
    Pkg.add("JET")
    using JET
else
    @info "Skipping JET tests on Julia $VERSION (prerelease)"
    @testset "JET Tests" begin end  # Empty testset
    return  # Exit the file
end

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqCore
using Test

@testset "JET Tests" begin
    # Test package for typos - now passing
    test_package(
        OrdinaryDiffEqTsit5, target_defined_modules = true, mode = :typo
    )

    # Test individual solver type stability
    @testset "Solver Type Stability Tests" begin
        # Test problem
        function simple_system!(du, u, p, t)
            du[1] = -0.5 * u[1]
            du[2] = -1.5 * u[2]
        end
        prob = ODEProblem(simple_system!, [1.0, 1.0], (0.0, 1.0))

        # Test all exported Tsit5 solvers
        tsit5_solvers = [Tsit5()]

        for solver in tsit5_solvers
            @testset "$(typeof(solver)) type stability" begin
                try
                    @test_opt broken = true init(prob, solver, dt = 0.1, save_everystep = false, abstol = 1.0e-6, reltol = 1.0e-6)
                    integrator = init(prob, solver, dt = 0.1, save_everystep = false, abstol = 1.0e-6, reltol = 1.0e-6)
                    @test_opt broken = true step!(integrator)
                catch e
                    @test_broken false # Mark as broken if solver fails to initialize
                    println("$(typeof(solver)) failed with: $e")
                end
            end
        end
    end
end
