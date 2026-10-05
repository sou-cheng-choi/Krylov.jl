module SpecialSymmetric

using LinearAlgebra
import ..Krylov

export csminares, csminares_range, ssminares, shminares, projected_solve,
       csminares_short, recurrence_solve,
       csminres, ssminres, shminres, csminresqlp, ssminresqlp, shminresqlp,
       csminares_real, csminres_real, csminresqlp_real,
       ssminares_phase, ssminres_phase, ssminresqlp_phase,
       shminares_phase, shminres_phase, shminresqlp_phase,
       cskewminares, cskewminares_real, phase_minares, jhermitian_minares,
       hamiltonian_minares, skewhamiltonian_minares, weighted_minares,
       CSRealification, PhaseHermitian, minimum_norm_refinement

realtype(::Type{T}) where {T} = typeof(real(zero(T)))

# Internal structure policies keep conjugation/sign rules out of the iteration.
# SS and SH share the same algebra; only SS restricts the data to real values.
const SkewStructure = Union{Val{:ss},Val{:sh}}
const OrdinaryStructure = Union{SkewStructure,Val{:hermitian}}
const ConjugatedStructure = Union{Val{:cs},Val{:cskew}}

function structure_policy(structure)
    structure in (:cs, :cskew, :ss, :sh, :hermitian) || throw(ArgumentError("unknown structure"))
    return Val(structure)
end

structure_partner(::Val{:cs}, A) = transpose(A)
structure_partner(::Val{:cskew}, A) = -transpose(A)
structure_partner(::Val{:ss}, A) = -transpose(A)
structure_partner(::Val{:sh}, A) = -adjoint(A)
structure_partner(::Val{:hermitian}, A) = adjoint(A)

# If A*trial_vectors(V) = Vnext*H, then
# A'*V = trial_vectors(Vnext)*adjoint_image(H).
trial_vectors(::ConjugatedStructure, V) = conj.(V)
trial_vectors(::OrdinaryStructure, V) = V
adjoint_image(::Val{:cs}, H) = conj.(H)
adjoint_image(::Val{:cskew}, H) = -conj.(H)
adjoint_image(::SkewStructure, H) = -H
adjoint_image(::Val{:hermitian}, H) = H

validate(A, b, structure::Symbol; check=true) =
    validate(A, b, structure_policy(structure); check)

function validate(A, b, policy::Val{S}; check=true) where {S}
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))
    length(b) == n || throw(DimensionMismatch("length(b) must equal size(A, 1)"))
    n > 0 || throw(ArgumentError("empty systems are not supported"))
    all(isfinite, b) || throw(ArgumentError("b must be finite"))
    if S == :ss
        eltype(A) <: Real && eltype(b) <: Real ||
            throw(ArgumentError("SS methods require real A and b; use SH for complex b"))
    end
    T = promote_type(float(eltype(A)), float(eltype(b)))
    realtype(T) in (Float32, Float64) ||
        throw(ArgumentError("the reference implementation supports Float32 and Float64"))
    if check
        all(isfinite, A) || throw(ArgumentError("A must be finite"))
        partner = structure_partner(policy, A)
        norm(A - partner) <= 100eps(realtype(T)) * norm(A) ||
            throw(ArgumentError("A does not have the requested $S structure"))
    end
    return T
end

# Normal residual using the known symmetry; no adjoint operator is required.
normal_product(A, r, structure::Symbol) = normal_product(A, r, structure_policy(structure))
normal_product(A, r, policy::Val) = adjoint_image(policy, A * trial_vectors(policy, r))

residual_converged(nr, nar, nb, nab, atol, rtol, artol=rtol) =
    nr <= atol + rtol * nb || nar <= atol + artol * nab

function minnorm_ls(B, f, ranktol; factorization=:givens)
    factorization == :givens && return givens_minnorm_ls(B, f, ranktol)
    factorization == :svd || throw(ArgumentError("factorization must be :givens or :svd"))
    F = svd(B; full=false)
    isempty(F.S) && return zeros(eltype(B), size(B, 2))
    cutoff = ranktol * maximum(F.S)
    weights = adjoint(F.U) * f
    for i in eachindex(F.S)
        weights[i] = F.S[i] > cutoff ? weights[i] / F.S[i] : zero(eltype(weights))
    end
    return adjoint(F.Vt) * weights
end

"""
    projected_solve(A, b; structure=:cs, objective=:ares, completion=:invariant,
                    maxiter=length(b), atol=0, rtol=√eps, ranktol=n*eps,
                    breakdown_tol=100eps, factorization=:givens, check=true)

Full-basis solver with two-pass reorthogonalization and rank-revealing Givens
QR/QLP projected solves. `factorization=:svd` retains the independent SVD oracle.
For `:cs` and `:cskew`, the trial basis is conjugate to the generated basis. For `:ss`,
`:sh`, and `:hermitian`, it is the ordinary Krylov basis. `:ares` minimizes
norm(A'*(b-A*x)); `:residual` minimizes norm(b-A*x).

`:invariant` continues to subspace closure, selecting the smallest coefficient
norm at each step. In exact arithmetic, the final answer is A†b. `:stationary`
allows early residual/normal-residual stopping and need not return A†b.
The default is deliberately a research reference, not a short-recurrence solver.
No A'*A is formed. The small product used by `:ares` can square conditioning.
Givens rank selection uses trailing column norms, not singular values; near
the rank threshold it can differ from the SVD option. Both retain full bases
and recompute projected factorizations, so neither is a short-recurrence method.
Statistics separate basis products from products used for explicit diagnostics.
`check=false` permits matrix-free operators supporting size, eltype, and `*`;
the caller must then ensure symmetry and finite operator outputs.
"""
function projected_solve(A, b::AbstractVector; structure=:cs, kwargs...)
    return projected_solve_impl(A, b, structure_policy(structure); kwargs...)
end

# Function barrier: Julia specializes the shared iteration on the structure.
function projected_solve_impl(A, b::AbstractVector, policy::Val{S}; objective=:ares, start=:rhs,
                         completion=:invariant, maxiter=length(b), atol=0,
                         rtol=nothing, ranktol=nothing, breakdown_tol=nothing,
                         factorization=:givens, check=true) where {S}
    T = validate(A, b, policy; check)
    R = realtype(T)
    n = length(b)
    objective in (:ares, :residual) || throw(ArgumentError("unknown objective"))
    factorization in (:givens, :svd) || throw(ArgumentError("factorization must be :givens or :svd"))
    start in (:rhs, :normal) || throw(ArgumentError("unknown start"))
    start == :normal && (S != :cs || objective != :ares) &&
        throw(ArgumentError("normal start is implemented for CS-MinAres only"))
    completion in (:stationary, :invariant) || throw(ArgumentError("unknown completion"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    rtol = isnothing(rtol) ? sqrt(eps(R)) : R(rtol)
    ranktol = isnothing(ranktol) ? n * eps(R) : R(ranktol)
    breakdown_tol = isnothing(breakdown_tol) ? 100eps(R) : R(breakdown_tol)
    all(t -> isfinite(t) && t >= 0, (atol, rtol, ranktol, breakdown_tol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    b = Vector{T}(b)
    beta = norm(b)
    x = zeros(T, n)
    residuals = R[beta]
    normal_b = normal_product(A, b, policy)
    aresiduals = R[norm(normal_b)]
    projected_aresiduals = R[aresiduals[1]]
    diagnostic_products = 1
    iterates = Vector{T}[]
    if beta == 0 || aresiduals[1] == 0
        return x, (; niter=0, solved=true, status=:stationary_zero,
            closed=true, residuals, aresiduals, projected_aresiduals, iterates,
            basis_products=0, diagnostic_products, orthogonality=zero(R), factorization)
    end
    limit = min(n, maxiter)
    V = zeros(T, n, min(n, limit + 2))
    H = zeros(T, min(n, limit + 2), min(n, limit + 1))
    V[:, 1] = start == :rhs ? b / beta : trial_vectors(policy, normal_b) / norm(normal_b)
    nbasis = 1
    ncolumns = 0
    closed = false

    function extend!()
        j = ncolumns + 1
        v = view(V, :, j)
        q = A * trial_vectors(policy, v)
        all(isfinite, q) || throw(ArgumentError("nonfinite operator product"))
        scale = norm(q)
        # Keep every computed projection: forcing tridiagonality after
        # reorthogonalization would invalidate the represented operator.
        for pass in 1:2
            for i in 1:j
                h = dot(view(V, :, i), q)
                H[i, j] += h
                q .-= h .* view(V, :, i)
            end
        end
        tail = norm(q)
        closed = j == n || tail <= breakdown_tol * scale
        if !closed
            H[j + 1, j] = tail
            V[:, j + 1] = q / tail
            nbasis = j + 1
        end
        ncolumns = j
        return nothing
    end

    status = :iteration_limit
    solved = false
    kdone = 0
    for k in 1:limit
        ncolumns < k && extend!()
        m = min(k + 1, nbasis)
        if objective == :ares && ncolumns < m
            extend!() # one look-ahead product
        end
        C = H[1:m, 1:k]
        f = zeros(T, m)
        f[1] = beta
        if objective == :ares
            Hnext = H[1:nbasis, 1:m]
            D = adjoint_image(policy, Hnext)
            B = D * C
            g = D * f
            if start == :normal
                fill!(g, zero(T))
                g[1] = norm(normal_b)
            end
            y = minnorm_ls(B, g, ranktol; factorization)
            projected_norm = norm(g - B * y)
        else
            y = minnorm_ls(C, f, ranktol; factorization)
            projected_norm = R(NaN) # not computed by the residual-only reference
        end
        Z = view(V, :, 1:k)
        x = trial_vectors(policy, Z) * y
        r = b - A * x
        ar = normal_product(A, r, policy)
        diagnostic_products += 2
        push!(residuals, norm(r))
        push!(aresiduals, norm(ar))
        push!(projected_aresiduals, projected_norm)
        push!(iterates, copy(x))
        kdone = k
        solved = residual_converged(norm(r), norm(ar), beta, aresiduals[1], atol, rtol)
        terminal = closed && k == ncolumns
        if terminal
            status = solved ? :invariant_subspace : :invariant_subspace_unconverged
            break
        elseif solved && completion == :stationary
            status = :stationary
            break
        end
    end
    orthogonality = norm(adjoint(V[:, 1:nbasis]) * V[:, 1:nbasis] - I)
    return x, (; niter=kdone, solved, status, closed=closed && kdone == ncolumns,
        residuals, aresiduals, projected_aresiduals, iterates,
        basis_products=ncolumns, diagnostic_products, orthogonality, factorization)
end

csminares(A, b; kwargs...) = projected_solve(A, b; structure=:cs, kwargs...)
"""
    cskewminares(A, b; kwargs...)

Complex skew-symmetric (`transpose(A) == -A`) full-basis MinAres reference.
Uses the conjugated trial basis and A' = -conj(A). This structure differs from
skew Hermitian; a complex right-hand side is supported. Keyword arguments are
those of `projected_solve`; the default continues to subspace completion.
With zero starts and untruncated exact-arithmetic solves, iterate k equals LSMR
at floor(k/2); this entry point is a reference specialization, not a new method.
"""
cskewminares(A, b; kwargs...) = projected_solve(A, b; structure=:cskew, kwargs...)
"""
CS-MinAres with v₁ = conjugate(A'*b)/norm(A'*b). Every trial vector is
in range(A'), so an exactly stationary iterate is already the pseudoinverse
solution. This experimental range-start variant uses a different Saunders
subspace from csminares; its reduced right-hand side is norm(A'*b)*e1.
"""
csminares_range(A, b; completion=:stationary, kwargs...) =
    projected_solve(A, b; structure=:cs, start=:normal, completion, kwargs...)
"Real skew reference; exact untruncated iterate k equals zero-start LSMR at floor(k/2)."
ssminares(A, b; kwargs...) = projected_solve(A, b; structure=:ss, kwargs...)
"Direct SH-MinAres reference, sharing the full-basis engine with csminares and ssminares."
shminares(A, b; kwargs...) = projected_solve(A, b; structure=:sh, kwargs...)
csminres(A, b; kwargs...) =
    projected_solve(A, b; structure=:cs, objective=:residual, kwargs...)
ssminres(A, b; kwargs...) =
    projected_solve(A, b; structure=:ss, objective=:residual, kwargs...)
"Direct residual-minimizing reference on the skew-Hermitian Krylov basis."
shminres(A, b; kwargs...) =
    projected_solve(A, b; structure=:sh, objective=:residual, kwargs...)

"Real symmetric operator [real(A) imag(A); imag(A) -real(A)], applied without assembly."
struct CSRealification{R,M} <: AbstractMatrix{R}
    A::M
end
CSRealification(A, ::Type{R}=realtype(float(eltype(A)))) where {R} =
    CSRealification{R,typeof(A)}(A)
Base.size(H::CSRealification) = (2size(H.A, 1), 2size(H.A, 2))
Base.getindex(H::CSRealification, i::Int, j::Int) = begin
    n = size(H.A, 1)
    i <= n && j <= n ? real(H.A[i, j]) :
    i > n && j > n ? -real(H.A[i-n, j-n]) :
    imag(H.A[mod1(i, n), mod1(j, n)])
end
LinearAlgebra.issymmetric(::CSRealification) = true
LinearAlgebra.ishermitian(::CSRealification) = true
function LinearAlgebra.mul!(y::AbstractVector, H::CSRealification, z::AbstractVector)
    n = size(H.A, 1)
    length(z) == length(y) == 2n || throw(DimensionMismatch("realification dimension"))
    q = H.A * complex.(view(z, 1:n), -view(z, n+1:2n))
    y[1:n] .= real.(q)
    y[n+1:2n] .= imag.(q)
    return y
end

"Hermitian operator im*A for skew-Hermitian A, applied without assembly."
struct PhaseHermitian{T,M} <: AbstractMatrix{T}
    A::M
end
PhaseHermitian(A, ::Type{T}=Complex{realtype(float(eltype(A)))}) where {T} =
    PhaseHermitian{T,typeof(A)}(A)
Base.size(H::PhaseHermitian) = size(H.A)
Base.getindex(H::PhaseHermitian, i::Int, j::Int) = im * H.A[i, j]
LinearAlgebra.ishermitian(::PhaseHermitian) = true
function LinearAlgebra.mul!(y::AbstractVector, H::PhaseHermitian, z::AbstractVector)
    mul!(y, H.A, z)
    y .*= im
    return y
end

"""
    minimum_norm_refinement(A, b, x; structure=:cs, rtol=1e-8, atol=0)

Project away conjugate(r) for CS/complex skew symmetric or r for Hermitian/SS/SH. The exact
minimum-norm theorem assumes a stationary iterate in the corresponding
zero-start Saunders/Krylov subspace. This function checks stationarity but
cannot check the subspace assumption. Returns (refined_x, applied).
"""
function minimum_norm_refinement(A, b, x; structure=:cs, rtol=1e-8, atol=0)
    policy = structure_policy(structure)
    validate(A, b, policy)
    length(x) == length(b) || throw(DimensionMismatch("x has the wrong length"))
    all(isfinite, x) || throw(ArgumentError("x must be finite"))
    all(t -> isfinite(t) && t >= 0, (rtol, atol)) || throw(ArgumentError("invalid tolerance"))
    r = b - A * x
    nr = norm(r)
    nr <= atol + rtol * norm(b) && return copy(x), false
    norm(normal_product(A, r, policy)) <=
        atol + rtol * norm(normal_product(A, b, policy)) ||
        throw(ArgumentError("minimum-norm refinement requires a stationary iterate"))
    p = trial_vectors(policy, r) / nr
    return x - p * dot(p, x), true
end

function check_backend_options(kwargs)
    get(kwargs, :M, I) === I || throw(ArgumentError("preconditioning is not supported by these adapters"))
    iszero(get(kwargs, :λ, 0)) || throw(ArgumentError("pass a structure-preserving shifted A explicitly; backend λ changes the transformed problem"))
    return nothing
end

function verified_stats(A, b, x, backend, structure, kwargs)
    R = realtype(eltype(x))
    atol = get(kwargs, :atol, sqrt(eps(R)))
    rtol = get(kwargs, :rtol, sqrt(eps(R)))
    artol = get(kwargs, :Artol, rtol)
    r = b - A * x
    nr = norm(r)
    nar = norm(normal_product(A, r, structure))
    nab = norm(normal_product(A, b, structure))
    solved = all(isfinite, x) && residual_converged(nr, nar, norm(b), nab, atol, rtol, artol)
    return (; niter=backend.niter, solved,
        status=solved ? :verified : :failed_verification,
        residual_norm=nr, normal_residual_norm=nar,
        diagnostic_products=3, backend)
end

function backend_problem(::Val{:cs}, A, b, ::Type{T}) where {T}
    R = realtype(T)
    return CSRealification(A, R), R[real.(b); imag.(b)]
end

function backend_problem(::SkewStructure, A, b, ::Type{T}) where {T}
    F = Complex{realtype(T)}
    return PhaseHermitian(A, F), F.(im .* b)
end

function recover_solution(::Val{:cs}, z, b)
    n = length(b)
    return complex.(z[1:n], -z[n+1:2n])
end
recover_solution(::Val{:ss}, z, b) = real.(z)
recover_solution(::Val{:sh}, z, b) = z

# One adapter path for validation, backend options, execution and verification.
# Only the equivalent Hermitian problem and solution unpacking depend on symmetry.
function structured_krylov(method, A, b, policy; check=true, kwargs...)
    T = validate(A, b, policy; check)
    check_backend_options(kwargs)
    H, rhs = backend_problem(policy, A, b, T)
    z, stats = method(H, rhs; kwargs...)
    x = recover_solution(policy, z, b)
    return x, verified_stats(A, b, x, stats, policy, kwargs)
end

"Krylov.jl MINARES on a real symmetric embedding; its subspaces differ from csminares."
csminares_real(A, b; kwargs...) = structured_krylov(Krylov.minares, A, b, Val(:cs); kwargs...)
"Krylov.jl MINRES on a real symmetric embedding, not the original Saunders recurrence."
csminres_real(A, b; kwargs...) = structured_krylov(Krylov.minres, A, b, Val(:cs); kwargs...)
"Krylov.jl MINRES-QLP on a real symmetric embedding; use csminres for Saunders iterates."
csminresqlp_real(A, b; kwargs...) = structured_krylov(Krylov.minres_qlp, A, b, Val(:cs); kwargs...)

"SH-MinAres via Krylov.jl MINARES on (im*A, im*b), with the original least-squares objective."
shminares_phase(A, b; kwargs...) = structured_krylov(Krylov.minares, A, b, Val(:sh); kwargs...)
"SS-MinAres via Krylov.jl on (im*A, im*b); complex arithmetic internally, real output."
ssminares_phase(A, b; kwargs...) = structured_krylov(Krylov.minares, A, b, Val(:ss); kwargs...)
shminres_phase(A, b; kwargs...) = structured_krylov(Krylov.minres, A, b, Val(:sh); kwargs...)
shminresqlp_phase(A, b; kwargs...) = structured_krylov(Krylov.minres_qlp, A, b, Val(:sh); kwargs...)
ssminres_phase(A, b; kwargs...) = structured_krylov(Krylov.minres, A, b, Val(:ss); kwargs...)
ssminresqlp_phase(A, b; kwargs...) = structured_krylov(Krylov.minres_qlp, A, b, Val(:ss); kwargs...)

include("csminresqlp.jl")
include("projected_givens.jl")
include("recurrence.jl")

"""
    ssminresqlp(A, b; kwargs...)
    shminresqlp(A, b; kwargs...)

Direct full-basis residual minimization with Givens QR/QLP for real skew
symmetric or skew Hermitian systems. No phase transformation or SVD is used.
The shared projected solver supplies reorthogonalization and minimum-norm
selection on each trial space. These are full-basis implementations, not
ports of historical short recurrences. Keyword arguments follow
`projected_solve`, with objective, structure, and factorization fixed here.
"""
ssminresqlp(A, b; kwargs...) = native_skew_qlp(A, b, :ss; kwargs...)
"Direct full-basis SH-MINRES-QLP; see ssminresqlp for the shared contract."
shminresqlp(A, b; kwargs...) = native_skew_qlp(A, b, :sh; kwargs...)
function native_skew_qlp(A, b, structure; kwargs...)
    any(k -> haskey(kwargs, k), (:structure, :objective, :factorization)) &&
        throw(ArgumentError("native skew QLP fixes structure, residual objective and Givens factorization"))
    x, stats = projected_solve(A, b; structure, objective=:residual,
        factorization=:givens, kwargs...)
    return x, (; stats..., backend_kind=:native_projected_qlp)
end

include("extensions.jl")

end
