# Shared helpers and fixtures for the Krylov.SpecialSymmetric tests.

const RNG = MersenneTwister(20261005)

# No elementwise access: this exercises the documented operator interface.
struct ProductOnly{T,M} <: AbstractMatrix{T}
    data::M
end

ProductOnly(A) = ProductOnly{eltype(A),typeof(A)}(A)
Base.size(A::ProductOnly) = size(A.data)
Base.getindex(::ProductOnly, i::Int, j::Int) = error("elementwise access unavailable")
Base.:*(A::ProductOnly, x::AbstractVector) = A.data * x
LinearAlgebra.mul!(y::AbstractVector, A::ProductOnly, x::AbstractVector) = mul!(y, A.data, x)

function cs_matrix(n, rank=n)
    U = Matrix(qr(randn(RNG, ComplexF64, n, n)).Q)
    s = [collect(range(0.5, 2.0; length=rank)); zeros(n-rank)]
    return U * Diagonal(s) * transpose(U)
end

function skew_matrix(n)
    Q = Matrix(qr(randn(RNG, n, n)).Q)
    D = zeros(n, n)
    for j in 1:2:n-1
        D[j, j+1], D[j+1, j] = j, -j
    end
    return Q * D * transpose(Q)
end

function check_reference(A, b, structure)
    x, s = projected_solve(A, b; structure, rtol=1e-11)
    target = pinv(A; rtol=1e-12) * b
    @test s.solved
    @test s.closed
    @test all(isfinite, x)
    @test x ≈ target rtol=2e-9 atol=2e-10
    @test norm(A' * (b-A*x)) <= 2e-10 * max(norm(A)*norm(b), 1)
    @test s.orthogonality <= 1e-12
    @test all(diff(s.aresiduals) .<= 1e-10 * max(s.aresiduals[1], 1))
    @test s.projected_aresiduals ≈ s.aresiduals atol=1e-10 rtol=1e-9
    @test s.basis_products <= s.niter + 1
    @test s.diagnostic_products == 1 + 2s.niter
    # Independent oracle: minimize over the returned iterate's trial space,
    # constructed from raw powers instead of the implementation's projections.
    raw = reshape(copy(b), :, 1)
    for k in 1:s.niter
        Zraw = structure == :cs ? conj.(raw) : raw
        Z = Matrix(qr(Zraw).Q)[:, 1:k]
        oracle = Z * (pinv(A' * A * Z; rtol=1e-11) * (A' * b))
        @test norm(A'*(b-A*s.iterates[k])) ≈ norm(A'*(b-A*oracle)) atol=5e-9
        v = structure == :cs ? A * conj.(raw[:, end]) : A * raw[:, end]
        raw = hcat(raw, iszero(norm(v)) ? v : v / norm(v))
    end
end

# Two matrices (full rank, rank deficient) and two right-hand sides for each
# transformed structure: complex skew, phase Hermitian, J-Hermitian,
# Hamiltonian, skew-Hamiltonian, and weighted self-adjoint.
function extended_systems(; n=8, seed=20261005)
    n >= 4 && iseven(n) || throw(ArgumentError("use an even n >= 4"))
    rng = MersenneTwister(seed)
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    J = Diagonal([ones(n÷2); -ones(n÷2)])
    E = Matrix{Float64}(I, n÷2, n÷2)
    Z = zeros(n÷2, n÷2)
    Jsp = [Z E; -E Z]
    S = Matrix(Diagonal(collect(range(1., 2.; length=n))))
    for j in 1:n-1
        S[j, j+1] = 0.2
    end
    W = S'*S
    theta = 0.63
    specs = NamedTuple[]
    for singular in (false, true)
        index = singular ? 2 : 1
        d = collect(range(-2., 2.; length=n))
        singular && (d[end-1:end] .= 0)
        H = U*Diagonal(d)*U'
        Hr = Q*Diagonal(d)*Q'
        K = zeros(n, n)
        for j in 1:2:(singular ? n-2 : n)
            K[j, j+1] = 0.5 + j/n
            K[j+1, j] = -K[j, j+1]
        end
        expected_rank = singular ? n-2 : n
        for (id, family, A, solve, factor) in (
            ("CK", "Complex skew symmetric", U*K*transpose(U),
                (A,b; kwargs...) -> cskewminares_real(A,b; kwargs...), nothing),
            ("PH", "Phase Hermitian", cis(theta)*H,
                (A,b; kwargs...) -> phase_minares(A,b; theta, kwargs...), nothing),
            ("JH", "J-Hermitian", J*H,
                (A,b; kwargs...) -> jhermitian_minares(A,b,J; kwargs...), nothing),
            ("HA", "Hamiltonian", -Jsp*Hr,
                (A,b; kwargs...) -> hamiltonian_minares(A,b; kwargs...), nothing),
            ("KH", "Skew-Hamiltonian", -Jsp*(Q*K*Q'),
                (A,b; kwargs...) -> skewhamiltonian_minares(A,b; kwargs...), nothing),
            ("WE", "Weighted self-adjoint", S\H*S,
                (A,b; kwargs...) -> weighted_minares(A,b,W; kwargs...), S))
            for rhs_kind in (:manufactured, :random)
                z = randn(rng, eltype(A), n)
                b = rhs_kind == :manufactured ? A*z : z
                target = isnothing(factor) ? pinv(A; rtol=1e-12)*b :
                    factor \ (pinv(factor*A/factor; rtol=1e-12)*(factor*b))
                push!(specs, (; id="$id$index", family, A, b, solve, factor, target,
                    expected_rank, rhs_kind, singular))
            end
        end
    end
    return specs
end

# Algorithmic translations of the three examples in Choi's csminresqlp.m,
# MATLAB Central submission 61151, version 1.1.0.0 (2017-01-14).
# The grid in example 2 is reduced from 50 to 5 for inexpensive validation.
# Its all-ones three-point stencil is not the usual graph Laplacian.
function legacy_examples()
    n = 100
    A1 = im * spdiagm(-1 => fill(-2., n-1), 0 => fill(4., n), 1 => fill(-2., n-1))
    b1 = im .* vec(sum(A1; dims=2))
    m = 5
    B = spdiagm(-1 => ones(m-1), 0 => ones(m), 1 => ones(m-1))
    A2 = im * kron(B, B)
    b2 = im .* vec(sum(A2; dims=2))
    A3 = im * Diagonal(collect(-10.:10.))
    b3 = im .* ones(21)
    return [("tridiagonal", A1, b1), ("block_stencil", A2, b2), ("singular_diagonal", A3, b3)]
end
