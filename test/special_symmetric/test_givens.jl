@testset "Givens complete orthogonal projected solves" begin
    rng = MersenneTwister(20261006)
    solve = SpecialSymmetric.givens_minnorm_ls
    # Rectangular full-rank, rank-deficient and zero-rank systems, including
    # wide matrices where a basic QR solution is generally not minimum norm.
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = real(T) == Float32 ? 2e-4 : 2e-11
        ranktol = real(T) == Float32 ? 1e-5 : 1e-12
        for (m,n) in ((7,4), (4,7), (5,5)), r in (0, 2, min(m,n))
            U = Matrix(qr(randn(rng,T,m,m)).Q)
            V = Matrix(qr(randn(rng,T,n,n)).Q)
            B = U[:,1:r] * Diagonal(T.(range(1,2; length=r))) * V[:,1:r]'
            f = randn(rng,T,m)
            x = solve(B,f,ranktol)
            oracle = pinv(B; rtol=ranktol)*f
            @test all(isfinite,x)
            @test x ≈ oracle atol=tol rtol=tol
            @test norm(B'*(f-B*x)) <= tol*max(norm(B)*norm(f),1)
            @test norm(V[:,r+1:end]'*x) <= tol*max(norm(x),1)
            @test solve(B[:,end:-1:1],f,ranktol) ≈ x[end:-1:1] atol=tol rtol=tol
        end
    end
    @test solve([1. 1.], [2.], 1e-12) ≈ [1.,1.]
    for scale in (1e-50,1.,1e50)
        B = scale .* ComplexF64[1 im 2; 2 2im 4; 0 1 1]
        f = scale .* ComplexF64[1, 3im, 2]
        @test solve(B,f,1e-12) ≈ pinv(B;rtol=1e-12)*f atol=1e-12 rtol=1e-12
    end
    @test solve(Diagonal([1.,1e-14,0.]), ones(3), 1e-12) ≈ [1.,0.,0.]
    @test solve(zeros(0,3), Float64[], 1e-12) == zeros(3)
    @test isempty(solve(zeros(3,0), ones(3), 1e-12))
    @test_throws DimensionMismatch solve(ones(2,2),ones(3),1e-12)
    @test_throws ArgumentError solve(ones(2,2),ones(2),-1.)
    @test_throws ArgumentError csminares(zeros(2,2),zeros(2);factorization=:unknown)
end

@testset "Direct Givens methods versus independent SVD oracle" begin
    rng = MersenneTwister(20261007)
    for n in (7,10), singular in (false,true)
        U = Matrix(qr(randn(rng,ComplexF64,n,n)).Q)
        d = collect(range(0.5,2;length=n))
        singular && (d[end-1:end] .= 0)
        C = U*Diagonal(d)*transpose(U)
        H = im*(U*Diagonal(d)*U')
        R = randn(rng,n,n)
        S = R-R'
        # Complex skew symmetry uses a unitary congruence, not a similarity.
        K = U*S*transpose(U)
        for (A,structure) in ((C,:cs),(H,:sh),(S,:ss),(K,:cskew))
            b = randn(rng,eltype(A),n)
            for objective in (:ares,:residual)
                x,g = projected_solve(A,b; structure,objective,rtol=1e-10)
                oracle,s = projected_solve(A,b; structure,objective,rtol=1e-10,factorization=:svd)
                @test g.factorization == :givens && s.factorization == :svd
                @test x ≈ oracle atol=1e-8 rtol=1e-8
                @test x ≈ pinv(A;rtol=1e-12)*b atol=1e-8 rtol=1e-8
                @test g.aresiduals ≈ s.aresiduals atol=1e-8 rtol=1e-8
            end
        end
    end
    for (solver,structure,T) in ((ssminresqlp,:ss,Float64),
                                 (shminresqlp,:sh,ComplexF64))
        for n in (5,8)
            if structure == :ss
                A = skew_matrix(n)
            else
                U = Matrix(qr(randn(rng,T,n,n)).Q)
                d = [(-1.).^(1:n-1) .* collect(range(0.5,2;length=n-1)); 0.]
                A = im*(U*Diagonal(d)*U')
            end
            b = randn(rng,T,n)
            x,g = solver(ProductOnly(A),b;check=false,rtol=1e-11)
            oracle,s = projected_solve(A,b;structure,objective=:residual,factorization=:svd)
            @test g.solved && g.factorization == :givens
            @test g.backend_kind == :native_projected_qlp
            @test x ≈ pinv(A)*b atol=1e-9
            @test g.residuals ≈ s.residuals atol=1e-10
            @test norm(A'*(b-A*x)) <= 1e-10norm(b)
        end
        x,s = solver(zeros(T,3,3),ones(T,3))
        @test iszero(norm(x)) && s.solved
        @test_throws ArgumentError solver(Matrix{T}(I,3,3),ones(T,3))
        @test_throws ArgumentError solver(zeros(T,3,3),ones(T,3);factorization=:svd)
    end
    A = ComplexF64[im 1+im; -1+im 3im]
    b = ComplexF64[1,im]
    x,s = shminares(A,b)
    @test s.factorization == :givens && x ≈ A\b
end
