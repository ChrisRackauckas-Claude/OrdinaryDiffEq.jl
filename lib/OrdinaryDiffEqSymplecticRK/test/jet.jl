import OrdinaryDiffEqSymplecticRK

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

@testset "JET Tests" begin
    test_package(
        OrdinaryDiffEqSymplecticRK, target_defined_modules = true, mode = :typo
    )
end
