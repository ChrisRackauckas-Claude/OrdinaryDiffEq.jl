const _K_CYCLE_SIZE = 4

@mutable_cache mutable struct Tsit5Cache{
        uType, rateType, uNoUnitsType, StageLimiter, StepLimiter,
        Thread, KCycleType,
    } <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    k1::rateType
    k2::rateType
    k3::rateType
    k4::rateType
    k5::rateType
    k6::rateType
    k7::rateType
    utilde::uType
    tmp::uType
    atmp::uNoUnitsType
    stage_limiter!::StageLimiter
    step_limiter!::StepLimiter
    thread::Thread
    k_cycle::KCycleType           # Vector{NTuple{7,rateType}} or Nothing
    k_cycle_idx::Int              # current index into k_cycle
    k_cycle_sol_k_idxs::Vector{Int}  # which sol.k index borrows each slot (0=free)
    k_cycle_tasks::Vector{Any}    # async materialization tasks (Nothing or Task)
end

function alg_cache(
        alg::Tsit5, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
        dt, reltol, p, calck,
        ::Val{true}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    k1 = zero(rate_prototype)
    k2 = zero(rate_prototype)
    k3 = zero(rate_prototype)
    k4 = zero(rate_prototype)
    k5 = zero(rate_prototype)
    k6 = zero(rate_prototype)
    k7 = zero(rate_prototype)
    utilde = zero(u)
    atmp = similar(u, uEltypeNoUnits)
    recursivefill!(atmp, false)
    tmp = zero(u)

    if calck
        # Pre-allocate N=_K_CYCLE_SIZE sets of 7 k-arrays.
        # Set 1 = the initial k1-k7, sets 2-N = fresh zero arrays.
        k_cycle = Vector{NTuple{7, typeof(k1)}}(undef, _K_CYCLE_SIZE)
        k_cycle[1] = (k1, k2, k3, k4, k5, k6, k7)
        for i in 2:_K_CYCLE_SIZE
            k_cycle[i] = ntuple(_ -> zero(rate_prototype), Val(7))
        end
        k_cycle_idx = 1
        k_cycle_sol_k_idxs = zeros(Int, _K_CYCLE_SIZE)
        k_cycle_tasks = Vector{Any}(undef, _K_CYCLE_SIZE)
        fill!(k_cycle_tasks, nothing)
    else
        k_cycle = nothing
        k_cycle_idx = 0
        k_cycle_sol_k_idxs = Int[]
        k_cycle_tasks = Any[]
    end

    return Tsit5Cache(
        u, uprev, k1, k2, k3, k4, k5, k6, k7, utilde, tmp, atmp,
        alg.stage_limiter!, alg.step_limiter!, alg.thread,
        k_cycle, k_cycle_idx, k_cycle_sol_k_idxs, k_cycle_tasks
    )
end

get_fsalfirstlast(cache::Tsit5Cache, u) = (cache.k1, cache.k7)

OrdinaryDiffEqCore.supports_k_swap(cache::Tsit5Cache) = cache.k_cycle !== nothing

function OrdinaryDiffEqCore.swap_k_buffers!(integrator, cache::Tsit5Cache)
    # Capture FSAL reference before rotation: fsallast points to old k7
    # which has the correct FSAL data from the just-completed step.
    integrator.fsallast = cache.k7

    # Advance to next cycle slot
    cache.k_cycle_idx = mod1(cache.k_cycle_idx + 1, _K_CYCLE_SIZE)
    next_set = cache.k_cycle[cache.k_cycle_idx]

    # Update cache fields to point to the new set's arrays
    cache.k1 = next_set[1]
    cache.k2 = next_set[2]
    cache.k3 = next_set[3]
    cache.k4 = next_set[4]
    cache.k5 = next_set[5]
    cache.k6 = next_set[6]
    cache.k7 = next_set[7]

    # Update integrator.k vector
    integrator.k[1] = cache.k1
    integrator.k[2] = cache.k2
    integrator.k[3] = cache.k3
    integrator.k[4] = cache.k4
    integrator.k[5] = cache.k5
    integrator.k[6] = cache.k6
    integrator.k[7] = cache.k7

    # fsalfirst points to new k1 (update_fsal! will copy fsallast into it)
    integrator.fsalfirst = cache.k1

    # fsallast still points to old set's k7 — correct FSAL data.
    # update_fsal! does recursivecopy!(fsalfirst, fsallast) as usual.

    return nothing
end

function alg_cache(
        alg::Tsit5, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
        dt, reltol, p, calck,
        ::Val{false}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    return Tsit5ConstantCache()
end
