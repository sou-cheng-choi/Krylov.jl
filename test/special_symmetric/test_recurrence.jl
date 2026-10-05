function structured_matrix(rng, structure, T, n, singular)
    d = collect(range(0.5, 2; length=n))
    singular && (d[end-1:end] .= 0)
    unitary(S) = Matrix(qr(randn(rng, S, n, n)).Q)
    if structure == :cs
        U = unitary(T)
        return U * Diagonal(d) * transpose(U)
    elseif structure in (:hermitian, :sh)
        U = unitary(T)
        H = U * Diagonal(d .* (-1).^(1:n)) * U'
        return structure == :sh ? im * H : H
    end
    K = zeros(T, n, n)
    for j in 1:(singular ? n÷2 - 1 : n÷2)
        K[2j-1, 2j], K[2j, 2j-1] = 0.5 + j/n, -(0.5 + j/n)
    end
    U = unitary(T)
    return structure == :cskew ? U * K * transpose(U) : U * K * U'
end

@testset "Short recurrence versus full projected iterates" begin
    rng = MersenneTwister(20261008)
    cases = ((:cs, ComplexF64), (:cs, Float64), (:cskew, ComplexF64),
             (:ss, Float64), (:sh, ComplexF64), (:hermitian, ComplexF64))
    for (structure, T) in cases, n in (6, 9), singular in (false, true), compatible in (false, true)
        A = structured_matrix(rng, structure, T, n, singular)
        b = randn(rng, T, n)
        compatible && (b = A * b)
        target = pinv(A; rtol=1e-12) * b
        x, full = projected_solve(A, b; structure, rtol=1e-10, factorization=:svd)
        for reorthogonalize in (true, false)
            y, s = recurrence_solve(A, b; structure, rtol=1e-10, reorthogonalize, history=true)
            @test s.solved
            @test s.factorization == :short && s.reorthogonalized == reorthogonalize
            if reorthogonalize
                # A nullspace direction found one step before detected closure
                # ends the recurrence early with the same minimum-norm answer.
                @test s.status in (:invariant_subspace, :rank_truncated)
                @test s.niter <= full.niter
                s.status == :invariant_subspace && @test s.niter == full.niter && s.closed == full.closed
            else
                @test s.status in (:invariant_subspace, :roundoff, :rank_truncated)
            end
            @test y ≈ x atol=1e-10 rtol=1e-10
            @test y ≈ target atol=1e-10 rtol=1e-10
            m = min(s.niter, full.niter)
            for k in 1:m
                @test s.iterates[k] ≈ full.iterates[k] atol=1e-10 rtol=1e-10
            end
            @test s.projected_aresiduals[1:m+1] ≈ full.projected_aresiduals[1:m+1] atol=1e-10 rtol=1e-10
            @test s.aresiduals[end] ≈ norm(A' * (b - A * y)) atol=1e-13
            @test all(diff(s.aresiduals) .<= 1e-10 * max(s.aresiduals[1], 1))
            @test s.diagnostic_products == 1 + 2s.niter
            @test s.niter <= s.basis_products <= s.niter + 1
        end
    end
end

@testset "Short recurrence versus Krylov.jl recurrences" begin
    rng = MersenneTwister(20261009)
    zero_tols = (; atol=0.0, rtol=0.0)
    for trial in 1:4
        # Skew normal-residual iterates equal LSMR at half the step count.
        for T in (Float64, ComplexF64)
            M = randn(rng, T, 18, 18)
            A = M - transpose(M)
            b = randn(rng, T, 18)
            structure = T <: Real ? :ss : :cskew
            _, s = recurrence_solve(A, b; structure, maxiter=10, ranktol=0, history=true)
            for k in 2:10
                xl, _ = Krylov.lsmr(A, b; itmax=fld(k, 2), btol=0.0, axtol=0.0,
                                    etol=0.0, conlim=0.0, zero_tols...)
                @test norm(s.iterates[k] - xl) <= 1e-9 * max(1, norm(xl))
            end
        end
        # Hermitian normal-residual iterates are MinAres iterates.
        U = Matrix(qr(randn(rng, ComplexF64, 18, 18)).Q)
        A = U * Diagonal(range(-2, 2; length=18) .+ 0.05) * U'
        b = randn(rng, ComplexF64, 18)
        _, s = recurrence_solve(A, b; structure=:hermitian, maxiter=10, ranktol=0, history=true)
        for k in 1:10
            xm, _ = Krylov.minares(A, b; itmax=k, Artol=0.0, zero_tols...)
            @test norm(s.iterates[k] - xm) <= 1e-9 * max(1, norm(xm))
        end
    end
end

@testset "Short-recurrence range start and minimum norm" begin
    rng = MersenneTwister(20261010)
    for rank in (8, 6)
        A = cs_matrix(8, rank)
        b = randn(rng, ComplexF64, 8)
        x, full = csminares_range(A, b; completion=:invariant)
        y, s = csminares_short(A, b; start=:normal, reorthogonalize=true, history=true)
        @test s.niter == full.niter
        @test y ≈ x atol=1e-10
        @test y ≈ pinv(A) * b atol=1e-10
        N = nullspace(A)
        @test all(v -> norm(N' * v) < 1e-10, s.iterates)
    end
    # Exactly stationary early iterate versus the minimum-norm closure.
    A, b = Diagonal([1., 0.]), [1., 1.]
    early, se = csminares_short(A, b; completion=:stationary, history=true)
    full, sf = csminares_short(A, b)
    @test se.niter == 1 && early ≈ [1., 1.]
    @test sf.closed && sf.status == :invariant_subspace
    @test full ≈ [1., 0.] atol=1e-14
    @test sf.pivots[end] <= 1e-14
    # A*b = 0 but A'*b != 0: the adjoint must drive the objective.
    A = ComplexF64[1 im; im -1]
    for b in (ComplexF64[1, im], ComplexF64[1, 0])
        x, s = csminares_short(A, b)
        @test s.solved
        @test x ≈ pinv(A) * b atol=1e-12
    end
end

@testset "Short recurrence storage, precision and operators" begin
    rng = MersenneTwister(20261011)
    n = 60
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    A = U * Diagonal(range(0.5, 2; length=n)) * transpose(U)
    b = randn(rng, ComplexF64, n)
    x, s = csminares_short(A, b; rtol=1e-12)
    @test s.solved && s.status in (:invariant_subspace, :roundoff)
    @test x ≈ A \ b rtol=1e-9
    @test isempty(s.iterates) && length(s.residuals) == 2
    @test isnan(s.orthogonality)
    @test s.diagnostic_products == 3
    y, st = csminares_short(A, b; rtol=1e-8, completion=:stationary)
    @test st.status == :stationary && st.niter < s.niter
    @test norm(A' * (b - A * y)) <= 1e-7 * norm(A' * b)
    for op in (A, ProductOnly(A), sparse(A))
        z, so = csminares_short(op, b; check=false, rtol=1e-12)
        @test so.solved
        @test z ≈ x rtol=1e-10
    end
    for T in (Float32, ComplexF32)
        A = T[3 1; 1 2]
        T <: Complex && (A .*= T(1 + im))
        x, s = csminares_short(A, A * T[1, -2])
        @test eltype(x) == T && s.solved
        @test x ≈ T[1, -2] rtol=1e-4
    end
    for A in (zeros(3, 3), zeros(ComplexF64, 3, 3)), b in (zeros(3), ones(3))
        x, s = csminares_short(A, b)
        @test iszero(x) && s.status == :stationary_zero
    end
    x, s = csminares_short(Diagonal([1., 0.]), [0., 1.])
    @test iszero(x) && s.status == :stationary_zero
    x, s = csminares_short(reshape([2im], 1, 1), ComplexF64[3 + im])
    @test x ≈ [(3 + im) / (2im)] && s.closed && s.niter == 1
    _, s = csminares_short(cs_matrix(8), ones(ComplexF64, 8); maxiter=1, rtol=0, ranktol=0)
    @test s.niter == 1 && s.status == :iteration_limit
end

@testset "Short recurrence validation" begin
    @test_throws DimensionMismatch csminares_short(ones(2, 3), ones(2))
    @test_throws DimensionMismatch csminares_short(ones(2, 2), ones(3))
    @test_throws ArgumentError csminares_short(zeros(0, 0), zeros(0))
    @test_throws ArgumentError csminares_short([1. 2; 0 1], ones(2))
    @test_throws ArgumentError csminares_short(ones(2, 2), [NaN, 1.])
    @test_throws ArgumentError recurrence_solve(ones(2, 2), ones(2); structure=:unknown)
    @test_throws ArgumentError recurrence_solve([0. 1; -1 0], ones(2); structure=:ss, start=:normal)
    for kw in ((; rtol=-1), (; atol=Inf), (; ranktol=NaN), (; breakdown_tol=-1),
               (; maxiter=0), (; maxiter=1.5), (; completion=:unknown), (; start=:unknown),
               (; history=1), (; reorthogonalize=1))
        @test_throws ArgumentError csminares_short(ones(2, 2), ones(2); kw...)
    end
end
