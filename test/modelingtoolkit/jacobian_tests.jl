using OrdinaryDiffEq, ForwardDiff, Test, ADTypes

function d_alembert(du, u, p, t)
    return du[1] = p[1] - p[2] * u[1] + p[3] * t
end

function d_alembert_jac(J, u, p, t)
    return J[1] = -p[2]
end

function d_alembert_analytic(u0, p, t::Number)
    a, b, c = p
    ebt = exp(b * t)
    return @. exp(-b * t) * (-a * b + c + ebt * (a * b + c * (b * t - 1)) + b^2 * u0) / (b^2)
end

p = (1.0, 2.0, 3.0)
u0 = [1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(
    ODEFunction(
        d_alembert,
        jac = d_alembert_jac,
        analytic = d_alembert_analytic
    ),
    u0, tspan, p
)

sol = solve(prob, Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, Rosenbrock23(), abstol = 1.0e-8, reltol = 1.0e-8)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, Rodas4(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, Veldd4(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, Rodas5(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, TRBDF2(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 2.0e-6
sol = solve(prob, Trapezoid(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 2.0e-6
sol = solve(prob, KenCarp3(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 8.0e-4
sol = solve(prob, KenCarp4(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, KenCarp47(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7
sol = solve(prob, KenCarp58(), abstol = 1.0e-10, reltol = 1.0e-10)
@test sol.errors[:l2] < 1.0e-7

using ModelingToolkit
function lotka(du, u, p, t)
    x = u[1]
    y = u[2]
    du[1] = p[1] * x - p[2] * x * y
    return du[2] = -p[3] * y + p[4] * x * y
end

prob = ODEProblem(lotka, [1.0, 1.0], (0.0, 1.0), [1.5, 1.0, 3.0, 1.0])
de = ModelingToolkit.modelingtoolkitize(prob) |> complete
prob2 = ODEProblem(de, [], prob.tspan; jac = true)

sol = solve(prob, TRBDF2())

for Alg in [Rodas5, Rosenbrock23, TRBDF2, KenCarp4]
    @test Array(
        solve(
            prob2,
            Alg(),
            tstops = sol.t,
            adaptive = false
        )
    ) ≈ Array(
        solve(
            prob,
            Alg(),
            tstops = sol.t,
            adaptive = false
        )
    ) atol = 1.0e-4
end

## check chunk_size handling in ForwardDiff Jacobians
const chunksize = 1
function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃, check = p
    if check && eltype(u) <: ForwardDiff.Dual && ForwardDiff.npartials(u[1]) != chunksize
        @show ForwardDiff.npartials(u[1]), chunksize
        error("chunk_size is not as specified")
    end

    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
    du[3] = k₂ * y₂^2
    return nothing
end
prob1 = ODEProblem(rober, [1.0, 0.0, 0.0], (0.0, 1.0e5), (0.04, 3.0e7, 1.0e4, true))
sol1 = solve(prob1, TRBDF2(autodiff = AutoForwardDiff(chunksize = chunksize)))
prob = ODEProblem(rober, [1.0, 0.0, 0.0], (0.0, 1.0e5), (0.04, 3.0e7, 1.0e4, false))
sol = solve(prob, TRBDF2())
@test sol.u[end] == sol1.u[end]
@test length(sol.t) == length(sol1.t)

### DAE Jacobians
# https://github.com/SciML/DifferentialEquations.jl/issues/846

function dae_f(out, du, u, p, t)
    out[1] = -0.04u[1] + 1e4 * u[2] * u[3] - du[1]
    out[2] = +0.04u[1] - 3e7 * u[2]^2 - 1e4 * u[2] * u[3] - du[2]
    out[3] = u[1] + u[2] + u[3] - 1.0
end

dae_jac_used = Ref(false)
function dae_jac(J, du, u, p, gamma, t)
    dae_jac_used[] = true
    J[1, 1] = -0.04 - gamma
    J[1, 2] = 1e4 * u[3]
    J[1, 3] = 1e4 * u[2]
    J[2, 1] = 0.04
    J[2, 2] = -6e7 * u[2] - 1e4 * u[3] - gamma
    J[2, 3] = -1e4 * u[2]
    J[3, 1] = 1.0
    J[3, 2] = 1.0
    J[3, 3] = 1.0
end

dae_u₀ = [1.0, 0, 0]
dae_du₀ = [-0.04, 0.04, 0.0]
dae_tspan = (0.0, 100000.0)

dae_differential_vars = [true, true, false]
dae_prob = DAEProblem(
    DAEFunction(dae_f; jac = dae_jac), dae_du₀, dae_u₀, dae_tspan,
    differential_vars = dae_differential_vars
)
dae_prob2 = DAEProblem(
    dae_f, dae_du₀, dae_u₀, dae_tspan,
    differential_vars = dae_differential_vars
)

dae_jac_used[] = false
dae_sol = solve(dae_prob, DABDF2())
@test dae_jac_used[]
dae_jac_used[] = false
dae_sol2 = solve(dae_prob2, DABDF2())
@test !dae_jac_used[]
@test iszero(maximum(Array(dae_sol) - Array(dae_sol2)))
