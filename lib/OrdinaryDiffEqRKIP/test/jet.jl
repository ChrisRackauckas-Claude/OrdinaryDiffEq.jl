import OrdinaryDiffEqRKIP

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

using OrdinaryDiffEqRKIP
using OrdinaryDiffEqCore
using Test

@testset "JET Tests" begin
    # Test package for typos
    test_package(
        OrdinaryDiffEqRKIP, target_modules = (OrdinaryDiffEqRKIP,), mode = :typo
    )
end
