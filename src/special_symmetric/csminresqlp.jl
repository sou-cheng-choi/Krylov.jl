# Julia translation of the always-QLP branch of Choi's CS-MINRES-QLP,
# csminresqlp.m (MATLAB Central File Exchange 61151, version 1.1.0.0, 2017),
# including SymOrtho by Sou-Cheng T. Choi and Michael A. Saunders. The original
# source carries its own attribution, copyright, and distribution notices.
# Reference: Choi, arXiv:1304.6782. This port deliberately replaces the legacy
# stopping/diagnostic code; it does not translate the preconditioning interface
# or the MINRES-to-QLP transition. No real embedding or projected SVD is used.

# Hermitian reflection [c s; conj(s) -c] * [a; b] = [r; 0], c real.
function cs_symortho(a::T, b::T) where {T<:Number}
    R = realtype(T)
    aa, ab = abs(a), abs(b)
    if isreal(a) && isreal(b)
        if iszero(b)
            return iszero(a) ? one(R) : sign(real(a)), zero(T), T(aa)
        elseif iszero(a)
            return zero(R), T(sign(real(b))), T(ab)
        end
        radius = hypot(aa, ab)
        return real(a) / radius, b / radius, T(radius)
    elseif iszero(b)
        return one(R), zero(T), a
    elseif iszero(a)
        return zero(R), one(T), b
    end
    if ab > aa
        t = aa / ab
        scale = inv(sqrt(one(R) + t*t))
        s = scale * conj(sign(b) / sign(a))
        return scale*t, s, b / conj(s)
    end
    t = ab / aa
    c = inv(sqrt(one(R) + t*t))
    s = c*t * conj(sign(b) / sign(a))
    return c, s, a / c
end

"""
    csminresqlp(A, b; atol=0, rtol=√eps, maxiter=4length(b),
                       ranktol=length(b)*eps, breakdown_tol=100eps,
                       completion=:invariant, shift=0, reorthogonalize=true,
                       history=true, check=true)

Direct Julia translation of the always-QLP recurrence in the archived MATLAB
CS-MINRES-QLP. Solve `(A-shift*I)x ≈ b`, with `transpose(A)==A`, from zero,
using the conjugated Saunders basis and left/right QLP reflections. Real
symmetric matrices are supported too. Returns `(x, stats)`, as `csminares` does.
The existing `csminresqlp_real` remains the Krylov.jl real-embedding adapter.

The default `completion=:invariant` continues to numerical Saunders closure or
a verified rank-truncated QLP solution. QLP discards a trailing pivot below
`ranktol` times the running operator scale to select minimum norm. After such
truncation, a verified iterate is returned with `status=:rank_truncated` rather
than continuing into nearly singular interior triangular solves.
`:stationary` allows earlier explicit residual
or normal-residual stopping; early stationarity alone does not certify minimum
norm. Neither the estimated numerical rank nor finite-precision closure is an
unconditional minimum-norm guarantee. Inspect solution error in validation runs.

Statistics include `niter`, `solved`, `status`, `closed`, `residuals`, `aresiduals`,
`iterates`, `basis_products`, and `diagnostic_products`. Residual histories use
the shifted original system and its adjoint, explicitly recomputed each step.
`solved` means residual OR normal-residual convergence, not minimum norm.
Two-pass reorthogonalization is enabled by default, storing O(nk) basis entries,
to prevent loss of orthogonality from obscuring singular termination. The QLP
solution updates use short recurrences; there are no projected SVD solves.
`reorthogonalize=false` retains the O(n) recurrence workspace of the original
algorithm, but can lose minimum-norm accuracy on singular systems even with a
small normal residual. `history=true` additionally stores every solution iterate.
With `history=false`, `iterates` is empty while scalar histories remain available.
No normal matrix or real embedding is formed.

This initial port has no preconditioner or MINRES-to-QLP switching option.
`check=false` supports product-only operators with `size`, `eltype`, and `*`;
the caller then guarantees symmetry. Loss of basis orthogonality can affect
ill-conditioned systems; use the reorthogonalized `csminres` as an
independent small-problem comparison.
"""
function csminresqlp(A, b::AbstractVector; atol=0, rtol=nothing,
                           maxiter=4length(b), ranktol=nothing,
                           breakdown_tol=nothing, completion=:invariant,
                           shift=0, reorthogonalize=true, history=true, check=true)
    T0 = validate(A, b, Val(:cs); check)
    shift isa Number && isfinite(shift) || throw(ArgumentError("shift must be finite"))
    T = promote_type(T0, typeof(shift))
    R = realtype(T)
    R in (Float32, Float64) || throw(ArgumentError("unsupported shift precision"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    completion in (:invariant, :stationary) || throw(ArgumentError("unknown completion"))
    history isa Bool || throw(ArgumentError("history must be Boolean"))
    reorthogonalize isa Bool || throw(ArgumentError("reorthogonalize must be Boolean"))
    n = length(b)
    atol = R(atol)
    rtol = isnothing(rtol) ? sqrt(eps(R)) : R(rtol)
    ranktol = isnothing(ranktol) ? n*eps(R) : R(ranktol)
    breakdown_tol = isnothing(breakdown_tol) ? 100eps(R) : R(breakdown_tol)
    all(t -> isfinite(t) && t >= 0, (atol, rtol, ranktol, breakdown_tol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    b = Vector{T}(b)
    sigma = T(shift)
    function product(v)
        q = A*v - sigma*v
        all(isfinite, q) || throw(ArgumentError("nonfinite operator product"))
        return q
    end
    normal(v) = conj.(product(conj.(v)))
    beta1 = norm(b)
    x = zeros(T, n)
    normal_b = norm(normal(b))
    residuals, aresiduals = R[beta1], R[normal_b]
    iterates = Vector{T}[]
    basis_products, diagnostic_products, niter = 0, 1, 0
    closed, solved = false, false
    status = :iteration_limit
    qlp_pivots = R[]
    discarded_pivots = 0

    if iszero(beta1) || iszero(normal_b)
        return x, (; niter, solved=true, status=:stationary_zero, closed=true,
            residuals, aresiduals, iterates, basis_products, diagnostic_products,
            qlp_pivots, discarded_pivots, reorthogonalized=reorthogonalize,
            residual_norm=beta1, normal_residual_norm=normal_b, solution_norm=zero(R),
            backend_kind=:native_qlp, shift=sigma)
    end

    # Saunders vectors, three rotated solution directions, and settled solution.
    r1, r2 = zeros(T, n), copy(b)
    w, wl, xl2 = zeros(T, n), zeros(T, n), zeros(T, n)
    beta, betan = zero(R), beta1
    cs, sn = -one(R), zero(T)
    cr1, sr1, cr2, sr2 = one(R), zero(T), -one(R), zero(T)
    tau = taul = zero(T)
    phi = T(beta1)
    dltan = eplnn = gama = gamal = gamal2 = zero(T)
    eta = etal = etal2 = vepln = veplnl = veplnl2 = zero(T)
    ul3 = ul2 = ul = u = zero(T)
    operator_scale = zero(R)
    limit = reorthogonalize ? min(n,maxiter) : maxiter
    V = Vector{T}[]

    for k in 1:limit
        betal, beta = beta, betan
        v = r2 / beta
        q = product(conj.(v))
        basis_products += 1
        scale = norm(q)
        k > 1 && (q -= (beta/betal)*r1)
        alfa = dot(v, q)
        q -= (alfa/beta)*r2
        if reorthogonalize
            push!(V,copy(v))
            for pass in 1:2, j in 1:k
                q -= dot(V[j],q)*V[j]
            end
        end
        r1, r2 = r2, q
        betan = norm(q)
        operator_scale = max(operator_scale, scale, abs(alfa), beta*(k > 1))
        closed = (reorthogonalize && k == n) || betan <= breakdown_tol * operator_scale
        closed && (betan = zero(R))

        # Previous and current left reflections: Q * Tbar = [R; 0].
        dbar, epln = dltan, eplnn
        dlta = cs*dbar + sn*alfa
        gbar = conj(sn)*dbar - cs*alfa
        eplnn, dltan = sn*betan, -cs*betan
        gamal2, gamal = gamal, gama
        cs, sn, gama = cs_symortho(T(gbar), T(betan))
        taul2, taul, tau = taul, tau, cs*phi
        phi = conj(sn)*phi

        # Right reflections turn the upper factor into a lower QLP factor.
        if k > 2
            veplnl2, etal2, etal = veplnl, etal, eta
            dlta, veplnl = sr2*vepln - cr2*dlta, cr2*vepln + conj(sr2)*dlta
            eta, gama = conj(sr2)*gama, -cr2*gama
        end
        if k > 1
            cr1, sr1, reflected = cs_symortho(conj(gamal), conj(dlta))
            gamal = conj(reflected)
            vepln, gama = conj(sr1)*gama, -cr1*gama
        end

        # Delayed lower-triangular solve. A trailing numerical zero is truncated;
        # a zero interior pivot cannot be repaired by this short recurrence.
        ul4, ul3 = ul3, ul2
        if (k > 2 && iszero(gamal2)) || (k > 1 && iszero(gamal))
            status = :rank_breakdown
            break
        end
        k > 2 && (ul2 = (taul2 - etal2*ul4 - veplnl2*ul3) / gamal2)
        k > 1 && (ul = (taul - etal*ul3 - veplnl*ul2) / gamal)
        truncated = abs(gama) <= ranktol * operator_scale
        if !truncated
            u = (tau - eta*ul2 - vepln*ul) / gama
        else
            u = zero(T)
            discarded_pivots += 1
        end
        push!(qlp_pivots, abs(gama))

        # Always-QLP direction and solution updates from the MATLAB source.
        z = conj.(v)
        if k == 1
            wl2, wl, w = wl, z*conj(sr1), z*cr1
        elseif k == 2
            wl2, wl, w = wl, w*cr1 + z*conj(sr1), w*sr1 - z*cr1
        else
            wl2, wl = wl, w
            w = wl2*sr2 - z*cr2
            wl2 = wl2*cr2 + z*conj(sr2)
            wl, w = wl*cr1 + w*conj(sr1), wl*sr1 - w*cr1
        end
        xl2 += wl2*ul2
        candidate = xl2 + wl*ul + w*u
        if !all(isfinite, candidate)
            status = :numerical_breakdown
            break
        end
        x = candidate
        cr2, sr2, reflected = cs_symortho(conj(gamal), conj(eplnn))
        gamal = conj(reflected)

        r = b - product(x)
        ar = normal(r)
        diagnostic_products += 2
        push!(residuals, norm(r))
        push!(aresiduals, norm(ar))
        history && push!(iterates, copy(x))
        niter = k
        solved = residual_converged(residuals[end], aresiduals[end], beta1,
                                    normal_b, atol, rtol)
        if closed
            status = solved ? :invariant_subspace : :invariant_subspace_unconverged
            break
        elseif truncated && solved
            status = :rank_truncated
            break
        elseif solved && completion == :stationary
            status = :stationary
            break
        end
    end
    closed = closed && niter == basis_products
    return x, (; niter, solved, status, closed, residuals, aresiduals, iterates,
        basis_products, diagnostic_products, qlp_pivots, discarded_pivots,
        residual_norm=residuals[end], normal_residual_norm=aresiduals[end], solution_norm=norm(x),
        reorthogonalized=reorthogonalize, backend_kind=:native_qlp, shift=sigma)
end
