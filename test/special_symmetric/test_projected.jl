@testset "Structured projected MinAres" begin
    for n in (3, 6, 9), rank in (n, n-1)
        A = cs_matrix(n, rank)
        check_reference(A, randn(RNG, ComplexF64, n), :cs)
        check_reference(A, A * randn(RNG, ComplexF64, n), :cs)
    end
    for n in (3, 6, 9)
        A = skew_matrix(n)
        check_reference(A, randn(RNG, n), :ss)
        check_reference(A, A * randn(RNG, n), :ss)
        H = im .* (A + A') + A # same real skew matrix, complex storage
        check_reference(H, randn(RNG, ComplexF64, n), :sh)
    end
    U = Matrix(qr(randn(RNG, ComplexF64, 7, 7)).Q)
    H = U * Diagonal([-3., -2, -1, 0, 1, 2, 4]) * U'
    check_reference(H, randn(RNG, ComplexF64, 7), :hermitian)
    check_reference(im * H, randn(RNG, ComplexF64, 7), :sh)
end

@testset "Adjoint objective and singular completion" begin
    A = ComplexF64[1 im; im -1]
    b = ComplexF64[1, im]
    @test norm(A*b) == 0
    @test norm(A'*b) > 0
    x, s = csminares(A, b)
    @test x ≈ pinv(A)*b
    @test norm(b-A*x) < 1e-13
    @test s.niter == 1
    A = Diagonal([1., 0.])
    b = [1., 1.]
    early, s = csminares(A, b; completion=:stationary)
    @test norm(A'*(b-A*early)) < 1e-12
    @test norm(early - pinv(A)*b) > 0.5
    lifted, applied = minimum_norm_refinement(A, b, early)
    @test applied
    @test lifted ≈ pinv(A)*b atol=1e-12
    full, s = csminares(A, b)
    @test full ≈ pinv(A)*b atol=1e-12
    @test s.niter == 2
    @test_throws ArgumentError minimum_norm_refinement(A, b, zeros(2))
end

@testset "Edges, scales, sparse matrices and validation" begin
    for A in (zeros(3, 3), zeros(ComplexF64, 3, 3))
        for b in (zeros(3), ones(3))
            x, s = csminares(A, b)
            @test x == zeros(3)
            @test s.solved && s.niter == 0
        end
    end
    for scale in (1e-50, 1.0, 1e50)
        A = scale .* ComplexF64[2+im 1; 1 3-im]
        b = A * ComplexF64[1, -2im]
        x, s = csminares(A, b; rtol=1e-11)
        @test s.solved
        @test x ≈ [1, -2im] rtol=1e-10
    end
    A = ComplexF32[2+im 1; 1 3-im]
    x, s = csminares(A, A * ComplexF32[1, -2im])
    @test eltype(x) == ComplexF32
    @test x ≈ [1, -2im] rtol=1e-4
    A = im * spdiagm(-1=>fill(-1., 7), 0=>fill(3., 8), 1=>fill(-1., 7))
    x, s = csminares(A, A * ones(8))
    @test x ≈ ones(8) atol=1e-10
    x, s = csminares(reshape([2im], 1, 1), ComplexF64[3+im])
    @test x ≈ [(3+im)/(2im)]
    @test_throws DimensionMismatch csminares(ones(2, 3), ones(2))
    @test_throws DimensionMismatch csminares(ones(2, 2), ones(3))
    @test_throws ArgumentError csminares([1. 2; 0 1], ones(2))
    @test_throws ArgumentError csminares(zeros(2, 2), [NaN, 1])
    @test_throws ArgumentError csminares([Inf 0.; 0 1], ones(2))
    @test_throws ArgumentError csminares(ones(2, 2), ones(2); rtol=-1)
    @test_throws ArgumentError csminares(ones(2, 2), ones(2); maxiter=0)
    @test_throws ArgumentError ssminares(ComplexF64[0 1; -1 0], ones(2))
    @test_throws ArgumentError csminares(zeros(0, 0), zeros(0))
    A = cs_matrix(8)
    _, s = csminares(A, ones(8); maxiter=1, rtol=0, atol=0)
    @test s.status == :iteration_limit
    @test s.niter == 1 && !s.closed
end

@testset "Krylov.jl transformations and legacy solver ports" begin
    for rank in (5, 4)
        A = cs_matrix(5, rank)
        b = randn(RNG, ComplexF64, 5)
        H = CSRealification(A)
        z = randn(RNG, 10)
        hz = similar(z)
        mul!(hz, H, z)
        @test hz ≈ Matrix(H) * z
        @test Matrix(H) ≈ transpose(Matrix(H)) atol=1e-14
        x = complex.(z[1:5], -z[6:10])
        rho = [real.(b); imag.(b)] - hz
        @test norm(rho) ≈ norm(b-A*x)
        @test norm(Matrix(H)*rho) ≈ norm(A'*(b-A*x))
        for method in (csminres_real, csminresqlp_real, csminares_real)
            x, s = method(A, b; atol=1e-12, rtol=1e-12, itmax=100)
            @test all(isfinite, x)
            @test s.normal_residual_norm ≈ norm(A'*(b-A*x)) atol=1e-12
            if method != csminres_real || rank == 5
                @test norm(A'*(b-A*x)) < 1e-8
            else
                # Ordinary MINRES can develop a huge nullspace component.
                # The adapter must expose failure when true diagnostics fail.
                @test s.solved == (s.residual_norm <= 1e-12*(1+norm(b)) ||
                    s.normal_residual_norm <= 1e-12*(1+norm(A'*b)))
            end
            if method == csminresqlp_real
                @test x ≈ pinv(A)*b atol=1e-8
            end
        end
        xr, sr = csminres(A, b; rtol=1e-11)
        @test xr ≈ pinv(A)*b atol=1e-9
        @test all(diff(sr.residuals) .<= 1e-11)
    end
    for n in (5, 6)
        A = skew_matrix(n)
        b = randn(RNG, n)
        for method in (ssminres_phase, ssminresqlp_phase)
            x, s = method(A, b; atol=1e-12, rtol=1e-12, itmax=100)
            @test eltype(x) <: Real
            @test norm(A'*(b-A*x)) < 1e-8
        end
        A = ComplexF64.(A)
        b = randn(RNG, ComplexF64, n)
        for method in (shminres_phase, shminresqlp_phase, shminares_phase)
            x, s = method(A, b; atol=1e-12, rtol=1e-12, itmax=100)
            if method != shminres_phase || iseven(n)
                @test norm(A'*(b-A*x)) < 1e-8
            else
                @test s.solved == (s.residual_norm <= 1e-12*(1+norm(b)) ||
                    s.normal_residual_norm <= 1e-12*(1+norm(A'*b)))
            end
        end
    end
    @test_throws ArgumentError csminresqlp_real(Matrix{Float64}(I, 2, 2), ones(2); λ=1.)
    @test_throws ArgumentError shminares_phase(im*Matrix{Float64}(I, 2, 2), ones(2); M=2I)
end

@testset "Saunders enhancement and subspace comparisons" begin
    for rank in (8, 5)
        A = cs_matrix(8, rank)
        b = randn(RNG, ComplexF64, 8)
        x, s = csminares_range(A, b; rtol=1e-11)
        @test s.solved
        @test x ≈ pinv(A)*b atol=1e-9
        N = nullspace(A)
        for xk in s.iterates
            @test norm(N'*xk) < 1e-10
        end
        @test s.aresiduals ≈ s.projected_aresiduals atol=1e-10
        # At even k, the original Saunders trial space contains the LSMR
        # normal-equation Krylov space with half as many basis vectors.
        for j in 1:3
            xcs, _ = csminares(A, b; maxiter=2j)
            G = reshape(A'*b, :, 1)
            for k in 2:j
                G = hcat(G, A'*(A*G[:, end]))
            end
            Q = Matrix(qr(G).Q)[:, 1:j]
            xl = Q * ((A'*A*Q) \ (A'*b))
            @test norm(A'*(b-A*xcs)) <= norm(A'*(b-A*xl)) + 1e-10
        end
    end
    # A general complex shift changes the Saunders space; do not reuse T-σI.
    A = Diagonal([1., 2., 4.])
    b = ComplexF64[1, im, 1+im]
    Q = Matrix(qr(hcat(b, A*conj.(b))).Q)[:, 1:2]
    shifted = (A-I)*conj.(b)
    @test norm(shifted - Q*(Q'*shifted)) > 0.1
    x, s = csminares_range(zeros(2, 2), ones(2))
    @test x == zeros(2) && s.solved
end

@testset "Product-only operators" begin
    A = ComplexF64[2+im 1; 1 3-im]
    b = ComplexF64[1, 2im]
    op = ProductOnly(A)
    for method in (csminares, csminares_range, csminares_real, csminresqlp_real)
        x, stats = method(op, b; check=false)
        @test stats.solved
        @test x ≈ A \ b rtol=1e-9
    end
    H = im * ComplexF64[2 im; -im 3]
    x, stats = shminares_phase(ProductOnly(H), b; check=false)
    @test stats.solved
    @test x ≈ H \ b rtol=1e-9
end

@testset "Shared skew solvers and phase equivalence" begin
    rng = MersenneTwister(20261006)
    for R in (Float32, Float64)
        tol = R(R == Float32 ? 3e-4 : 1e-10)
        # Genuine complex skew-Hermitian matrices, rather than real skew
        # matrices stored as complex. Include a known nullspace direction.
        U = Matrix(qr(randn(rng, Complex{R}, 5, 5)).Q)
        for singular in (false, true)
            d = R[-3, -1, singular ? 0 : 1, 2, 4]
            A = im * U * Diagonal(d) * U'
            for compatible in (false, true)
                b = randn(rng, Complex{R}, 5)
                compatible && (b = A * b)
                target = pinv(A; rtol=tol/10) * b
                for method in (shminares, shminres)
                    x, s = method(A, b; rtol=tol)
                    @test s.solved
                    @test x ≈ target atol=tol rtol=tol
                    @test eltype(x) == Complex{R}
                end
                # At every fixed dimension, the phase transform has the same
                # trial space and objectives. This catches misplaced signs
                # or CS conjugations in the shared engine.
                for objective in (:ares, :residual)
                    x, s = projected_solve(A, b; structure=:sh, objective, maxiter=3)
                    y, h = projected_solve(im*A, im*b; structure=:hermitian, objective, maxiter=3)
                    @test x ≈ y atol=tol rtol=tol
                    @test s.residuals ≈ h.residuals atol=tol rtol=tol
                    @test s.aresiduals ≈ h.aresiduals atol=tol rtol=tol
                end
            end
        end
        A = R[0 2 0; -2 0 0; 0 0 0]
        b = R[1, 2, 3]
        x, direct = ssminares(A, b; rtol=tol)
        y, complex_direct = shminares(complex.(A), complex.(b); rtol=tol)
        @test eltype(x) == R
        @test x ≈ y atol=tol rtol=tol
        @test direct.aresiduals ≈ complex_direct.aresiduals atol=tol rtol=tol
        for matrix in (A, ProductOnly(A))
            x, s = ssminares_phase(matrix, b; check=false, rtol=tol, atol=tol, Artol=tol)
            @test eltype(x) == R
            @test s.solved
            @test norm(A'*(b-A*x)) <= tol * norm(A'*b)
            x, _ = minimum_norm_refinement(A, b, x; structure=:ss, rtol=tol)
            @test x ≈ pinv(A)*b atol=tol rtol=tol
        end
    end
    A = ComplexF64[2im 1+im; -1+im 3im]
    b = ComplexF64[1, 2im]
    for method in (shminares, shminres)
        x, s = method(ProductOnly(A), b; check=false)
        @test s.solved && x ≈ A \ b
        z, zs = method(zeros(ComplexF64, 3, 3), ones(ComplexF64, 3))
        @test zs.solved && iszero(norm(z))
        @test_throws ArgumentError method(Matrix{ComplexF64}(I, 2, 2), b)
        @test_throws DimensionMismatch method(A, ones(ComplexF64, 3))
    end
    @test_throws ArgumentError ssminares_phase(real.(A), b)
    @test_throws ArgumentError ssminares_phase(zeros(2,2), ones(2); λ=1.)
    @test_throws ArgumentError projected_solve(A, b; structure=:unknown)
    @test_throws ArgumentError shminares(A, b; start=:normal)
end
