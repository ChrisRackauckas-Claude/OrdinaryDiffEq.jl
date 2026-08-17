## Example work-precision diagrams for the DiffEqDevTools tagging/autoplot PR.
using OrdinaryDiffEq, DiffEqDevTools, Plots
using OrdinaryDiffEqRosenbrock: Rosenbrock23, Rodas4, Rodas5, Rodas5P
using OrdinaryDiffEqSDIRK: TRBDF2, KenCarp4, Kvaerno5
using OrdinaryDiffEqBDF: FBDF, QNDF
using OrdinaryDiffEqFIRK: RadauIIA5
using OrdinaryDiffEqTsit5: Tsit5
using ODEProblemLibrary: prob_ode_hires

gr(size = (900, 550), dpi = 130)
const OUT = joinpath(@__DIR__, "plots")
mkpath(OUT)

prob = prob_ode_hires
test_sol = solve(prob, Rodas5P(), abstol = 1.0e-14, reltol = 1.0e-14)

abstols = 1.0 ./ 10.0 .^ (5:10)
reltols = 1.0 ./ 10.0 .^ (2:7)

setups = [
    Dict{Symbol, Any}(:alg => Rosenbrock23(), :tags => [:rosenbrock, :second_order]),
    Dict{Symbol, Any}(:alg => Rodas4(), :tags => [:rosenbrock, :fourth_order]),
    Dict{Symbol, Any}(:alg => Rodas5(), :tags => [:rosenbrock, :fifth_order]),
    Dict{Symbol, Any}(:alg => Rodas5P(), :tags => [:rosenbrock, :fifth_order]),
    Dict{Symbol, Any}(:alg => TRBDF2(), :tags => [:sdirk, :second_order]),
    Dict{Symbol, Any}(:alg => KenCarp4(), :tags => [:sdirk, :fourth_order]),
    Dict{Symbol, Any}(:alg => Kvaerno5(), :tags => [:sdirk, :fifth_order]),
    Dict{Symbol, Any}(:alg => FBDF(), :tags => [:bdf]),
    Dict{Symbol, Any}(:alg => QNDF(), :tags => [:bdf]),
    Dict{Symbol, Any}(:alg => RadauIIA5(), :tags => [:firk, :reference]),
]

# One run, every view below comes out of it.
wall = @elapsed wp_set = WorkPrecisionSet(
    prob, abstols, reltols, setups;
    appxsol = test_sol, error_estimates = [:final, :l2, :L2], numruns = 10
)
@info "WorkPrecisionSet: $(round(wall, digits = 1))s for $(length(wp_set)) methods, errors = $(available_errors(wp_set))"

savefig(
    plot(wp_set, title = "HIRES: every method (what one run gives you today)"),
    joinpath(OUT, "01_all_methods.png")
)

views = autoplot(wp_set; families = [:rosenbrock, :sdirk, :bdf], reference_tags = :reference)
savefig(
    plot(
        [
            plot(
                views["family_$f"], reference_tags = :reference,
                title = "HIRES: $f family vs RadauIIA5"
            ) for f in ("rosenbrock", "sdirk", "bdf")
        ]...,
        layout = (3, 1), size = (900, 1300)
    ),
    joinpath(OUT, "02_autoplot_families.png")
)

savefig(
    plot(
        views["best_of_families"], reference_tags = :reference,
        title = "HIRES: best two of each family"
    ),
    joinpath(OUT, "03_best_of_families.png")
)

savefig(
    plot(
        plot(wp_set, x = :final, title = "final error"),
        plot(wp_set, x = :l2, title = "l2 error"),
        plot(wp_set, x = :L2, title = "L2 error"),
        layout = (1, 3), size = (1500, 500), legend = :bottomleft
    ),
    joinpath(OUT, "04_error_estimates.png")
)

ad_setups = with_autodiff_variants(
    [
        Dict{Symbol, Any}(:alg => Rodas5P(), :tags => [:rosenbrock]),
        Dict{Symbol, Any}(:alg => KenCarp4(), :tags => [:sdirk]),
    ];
    ad_backends = [AutoFiniteDiff()]
)
ad_set = WorkPrecisionSet(
    prob, abstols, reltols, ad_setups; appxsol = test_sol, numruns = 10
)
savefig(
    plot(
        ad_set, reference_tags = :autodiff_finitediff,
        reference_style = (linestyle = :dash, linewidth = 2, alpha = 0.8),
        title = "HIRES: ForwardDiff (solid) vs FiniteDiff (dashed) Jacobians"
    ),
    joinpath(OUT, "05_autodiff_comparison.png")
)

# A timeout keeps one pathological configuration from dominating the wall clock.
slow_setups = [
    Dict{Symbol, Any}(:alg => Rodas5P()),
    Dict{Symbol, Any}(:alg => Rosenbrock23(), :name => "Rosenbrock23 (capped)"),
]
capped = WorkPrecisionSet(
    prob, abstols, reltols, slow_setups; appxsol = test_sol, numruns = 10, timeout = 0.003
)
savefig(
    plot(capped, title = "HIRES: points over the 3ms per-solve budget are dropped"),
    joinpath(OUT, "06_timeout.png")
)

for (name, set) in ("all" => wp_set, "best_of_families" => views["best_of_families"])
    println(rpad(name, 20), " -> ", set.names)
end
println("wp_area ranking within :rosenbrock: ",
    best_by_tag(wp_set, :rosenbrock; n = 4).names)
println("plots written to ", OUT)
