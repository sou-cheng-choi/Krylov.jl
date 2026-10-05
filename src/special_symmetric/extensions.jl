# Small, explicitly assembled research adapters. All use the existing reference
# engine or Krylov.jl MinAres; the symmetry transformations are not new recurrences.

function extension_data(A::AbstractMatrix, b::AbstractVector, extra...)
    T = validate(A, b, Val(:hermitian); check=false)
    for E in extra
        size(E) == size(A) || throw(DimensionMismatch("structure matrix must match A"))
        all(isfinite, E) || throw(ArgumentError("structure matrix must be finite"))
        T = promote_type(T, float(eltype(E)))
    end
    realtype(T) in (Float32, Float64) || throw(ArgumentError("unsupported arithmetic"))
    all(isfinite, A) || throw(ArgumentError("A must be finite"))
    return Matrix{T}(A), Vector{T}(b)
end

# W = S'*S for a weighted problem; factor=nothing denotes Euclidean geometry.
# The weighted normal residual is ||A^sharp r||_W, A^sharp = W\(A'*W).
function extension_metrics(A, b, x, factor)
    r = b - A*x
    ar = A'*r
    if isnothing(factor)
        return norm(r), norm(ar), norm(x), norm(r), norm(ar)
    end
    sr = factor*r
    normal = adjoint(factor) \ (A' * (adjoint(factor) * sr))
    return norm(sr), norm(normal), norm(factor*x), norm(r), norm(ar)
end

function extension_solve(A, b, B, c, recover; structure=:hermitian,
                         factor=nothing, backend=:reference, atol=0, rtol=nothing,
                         maxiter=nothing)
    backend in (:reference, :krylov) || throw(ArgumentError("backend must be :reference or :krylov"))
    R = realtype(eltype(B))
    rtol = isnothing(rtol) ? sqrt(eps(R)) : R(rtol)
    atol = R(atol)
    all(t -> isfinite(t) && t >= 0, (atol, rtol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    limit = isnothing(maxiter) ? (backend == :reference ? length(c) : 4length(c)) : maxiter
    limit isa Integer && limit > 0 || throw(ArgumentError("maxiter must be positive"))
    validate(B, c, structure)
    refined = false
    if backend == :reference
        y, inner = projected_solve(B, c; structure, atol, rtol, maxiter=limit)
        ys = inner.iterates
    else
        # Real skew problems are phased to Hermitian form inside this branch.
        H, d = structure == :hermitian ? (B, c) : (im*B, im*c)
        ys = Vector{eltype(c)}[]
        unpack = eltype(c) <: Real ? y -> real.(y) : identity
        callback = workspace -> begin
            push!(ys, copy(unpack(Krylov.solution(workspace))))
            false
        end
        z, inner = Krylov.minares(H, d; atol, rtol, Artol=rtol, itmax=limit, callback)
        y = unpack(z)
        r = c - B*y
        # The zero-start Hermitian/skew Krylov history provides the subspace
        # assumption needed by terminal residual projection. Never refine an
        # unconverged vector or apply the unweighted formula in x coordinates.
        if norm(B'*r) <= atol + rtol*norm(B'*c)
            y, refined = minimum_norm_refinement(B, c, y; structure, atol, rtol)
            !isempty(ys) && (ys[end] = copy(y))
        end
    end
    x = recover(y)
    iterates = [recover(v) for v in ys]
    initial = extension_metrics(A, b, zero(x), factor)
    metrics = [extension_metrics(A, b, v, factor) for v in iterates]
    final = extension_metrics(A, b, x, factor)
    nr, nar, nx, euclidean_r, euclidean_ar = final
    solved = all(isfinite, x) && residual_converged(nr, nar, initial[1], initial[2], atol, rtol)
    residuals = R[initial[1]; [v[1] for v in metrics]]
    aresiduals = R[initial[2]; [v[2] for v in metrics]]
    return x, (; niter=inner.niter, solved, status=solved ? :verified : :failed_verification,
        metric=isnothing(factor) ? :euclidean : :weighted,
        residual_norm=nr, normal_residual_norm=nar, solution_norm=nx,
        euclidean_residual_norm=euclidean_r, euclidean_normal_residual_norm=euclidean_ar,
        residuals, aresiduals, iterates, refinement_applied=refined,
        transformed_structure=structure, backend_kind=backend, backend=inner)
end

"""
    cskewminares_real(A, b; backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Complex skew-symmetric MinAres via the real skew embedding
`B = [real(A) imag(A); imag(A) -real(A)]`, `c = [real(b); imag(b)]`.
Recover `x = y[1:n] - im*y[n+1:2n]`. This preserves Euclidean residual,
normal-residual, and solution norms. Its real trial spaces differ from the direct
complex-linear spaces used by `cskewminares`. Matrices are explicitly assembled.
`:reference` uses the shared full-basis Givens QR/QLP engine; `:krylov` uses Krylov.jl
MinAres with conditional terminal minimum-norm refinement. The latter has no
guarantee of minimum norm before accurate stationarity. Both start at zero.
"""
function cskewminares_real(A::AbstractMatrix, b::AbstractVector; kwargs...)
    A, b = extension_data(A, b)
    validate(A, b, :cskew)
    n = length(b)
    B = [real.(A) imag.(A); imag.(A) -real.(A)]
    c = [real.(b); imag.(b)]
    recover = y -> complex.(y[1:n], -y[n+1:2n])
    return extension_solve(A, b, B, c, recover; structure=:ss, kwargs...)
end

"""
    phase_minares(A, b; theta, backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Solve a phase-scaled Hermitian system `A = exp(im*theta)*H`, with Hermitian H,
by multiplying A and b by `exp(-im*theta)`. `theta` is a finite real angle in
radians supplied by the caller. Uses the same backend options and Euclidean
metric contract as `cskewminares_real`. This is not a general complex shift.
"""
function phase_minares(A::AbstractMatrix, b::AbstractVector; theta, kwargs...)
    theta isa Real && isfinite(theta) || throw(ArgumentError("theta must be finite and real"))
    A, b = extension_data(A, b)
    R = realtype(eltype(A))
    angle = R(theta)
    isfinite(angle) || throw(ArgumentError("theta is not representable in the input precision"))
    phase = cis(-angle)
    return extension_solve(A, b, phase*A, phase*b, identity; kwargs...)
end

"""
    jhermitian_minares(A, b, J; backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Solve A'*J = J*A with a Hermitian involution J (J' = J, J^2 = I), by solving
`(J*A)x = J*b`. The unitary left transform preserves Euclidean objectives and
minimum-norm solutions. General nonunitary indefinite weights are not accepted.
Matrices are explicitly assembled; backend options match `cskewminares_real`.
"""
function jhermitian_minares(A::AbstractMatrix, b::AbstractVector, J::AbstractMatrix; kwargs...)
    A, b = extension_data(A, b, J)
    J = Matrix{eltype(A)}(J)
    validate(J, b, :hermitian)
    R = realtype(eltype(A))
    norm(J*J - I) <= 100eps(R)*sqrt(length(b)) ||
        throw(ArgumentError("J must be a Hermitian involution (J^2 = I)"))
    return extension_solve(A, b, J*A, J*b, identity; kwargs...)
end

# Canonical J = [0 I; -I 0], applied by row/block permutation without forming J.
function symplectic_rows(A)
    n = size(A, 1)
    iseven(n) || throw(DimensionMismatch("Hamiltonian systems require even order"))
    m = n ÷ 2
    return vcat(A[m+1:n, :], -A[1:m, :])
end

function hamiltonian_solve(A, b, structure; kwargs...)
    eltype(A) <: Real && eltype(b) <: Real ||
        throw(ArgumentError("Hamiltonian adapters currently require real A and b"))
    A, b = extension_data(A, b)
    B = symplectic_rows(A)
    m = length(b) ÷ 2
    c = [b[m+1:end]; -b[1:m]]
    return extension_solve(A, b, B, c, identity; structure, kwargs...)
end

"""
    hamiltonian_minares(A, b; backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Real Hamiltonian system: A'*J + J*A = 0, J = [0 I; -I 0]. Solve the symmetric
system `(J*A)x = J*b`, preserving Euclidean objectives. A and b must be real
and the order must be even. Backend options match `cskewminares_real`.
"""
hamiltonian_minares(A::AbstractMatrix, b::AbstractVector; kwargs...) =
    hamiltonian_solve(A, b, :hermitian; kwargs...)

"""
    skewhamiltonian_minares(A, b; backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Real skew-Hamiltonian system: A'*J = J*A, J = [0 I; -I 0]. Solve the skew
symmetric system `(J*A)x = J*b`. Input and backend contracts match
`hamiltonian_minares`.
"""
skewhamiltonian_minares(A::AbstractMatrix, b::AbstractVector; kwargs...) =
    hamiltonian_solve(A, b, :ss; kwargs...)

raw"""
    weighted_minares(A, b, W; backend=:reference, atol=0, rtol=√eps, maxiter=nothing)

Weighted self-adjoint system: A'*W = W*A, with Hermitian positive-definite W.
Factor W = S'*S using Cholesky, solve `(S*A/S)y = S*b`, and recover x = S\y.
The minimized normal residual is ||A^sharp*(b-A*x)||_W, where
A^sharp = W\(A'*W) = A and ||v||_W = ||S*v||₂. At exact subspace completion,
the reference returns the minimum-W-norm weighted least-squares solution:
`S \ (pinv(S*A/S) * (S*b))`, generally different from `pinv(A)*b`.

Stats mark `metric=:weighted`; residuals, aresiduals, and solution_norm use W.
Separate euclidean_residual_norm and euclidean_normal_residual_norm fields
describe the original Euclidean problem. Backend options match
`cskewminares_real`. This dense Cholesky-based adapter is a small research
reference; it neither forms inv(W) nor accepts an indefinite weight.
"""
function weighted_minares(A::AbstractMatrix, b::AbstractVector, W::AbstractMatrix; kwargs...)
    A, b = extension_data(A, b, W)
    W = Matrix{eltype(A)}(W)
    validate(W, b, :hermitian)
    F = cholesky(Hermitian(W); check=false)
    issuccess(F) || throw(ArgumentError("W must be positive definite"))
    S = F.U
    B = (S*A) / S
    c = S*b
    return extension_solve(A, b, B, c, y -> S \ y; factor=S, kwargs...)
end
