@testset "Complex skew-symmetric conjugated basis" begin
    rng = MersenneTwister(903)
    for R in (Float32, Float64), n in (5, 6), singular in (false, true)
        T = Complex{R}
        tol = R == Float32 ? 3f-4 : 1e-10
        U = Matrix(qr(randn(rng, T, n, n)).Q)
        K = zeros(T, n, n)
        for j in 1:2:n-1
            singular && j == 1 && continue
            K[j,j+1], K[j+1,j] = R(j), -R(j)
        end
        A = U*K*transpose(U)
        b = randn(rng, T, n)
        x, stats = cskewminares(A,b; rtol=tol)
        @test stats.solved && stats.closed
        @test eltype(x) == T
        @test x ≈ pinv(A; rtol=tol/10)*b atol=tol rtol=tol
        @test stats.aresiduals ≈ stats.projected_aresiduals atol=tol rtol=tol
        @test all(diff(stats.aresiduals) .<= tol*max(norm(A'*b),1))
        @test norm(A'*b + conj.(A*conj.(b))) < tol*max(norm(A'*b),1)
        # Independent fixed-subspace oracle from raw antilinear powers.
        raw = reshape(copy(b),:,1)
        for k in 1:min(3,stats.niter)
            Z = Matrix(qr(conj.(raw)).Q)[:,1:k]
            oracle = Z * (pinv(A'*A*Z; rtol=tol/10)*(A'*b))
            @test norm(A'*(b-A*stats.iterates[k])) ≈ norm(A'*(b-A*oracle)) atol=10tol rtol=10tol
            v = A*conj.(raw[:,end])
            raw = hcat(raw, v/max(norm(v),eps(R)))
        end
    end
    A = ComplexF64[0 1+im; -1-im 0]
    b = ComplexF64[1,2im]
    @test_throws ArgumentError shminares(A,b)
    @test_throws ArgumentError cskewminares(im*Matrix{Float64}(I,2,2),b)
    @test cskewminares(ProductOnly(A),b; check=false)[1] ≈ A\b
    @test cskewminares(sparse(A),b)[1] ≈ A\b
    @test cskewminares(zeros(ComplexF64,3,3),ones(3))[1] == zeros(3)
end

@testset "Additional structures and both solver backends" begin
    cases = extended_systems()
    @test length(cases) == 24
    @test length(unique(p.id for p in cases)) == 12
    for p in cases, backend in (:reference, :krylov)
        A,b = p.A,p.b
        x, s = p.solve(A,b; backend, rtol=1e-10)
        @test s.solved
        @test length(x) == length(b) && all(isfinite,x)
        @test x ≈ p.target atol=1e-8 rtol=1e-8
        @test rank(A) == p.expected_rank
        @test s.euclidean_residual_norm ≈ norm(b-A*x) atol=1e-13
        @test s.euclidean_normal_residual_norm ≈ norm(A'*(b-A*x)) atol=1e-13
        @test length(s.residuals) == length(s.iterates)+1
        @test s.residuals[end] ≈ s.residual_norm atol=1e-13
        @test s.aresiduals[end] ≈ s.normal_residual_norm atol=1e-13
        @test all(diff(s.aresiduals) .<= 1e-8*max(s.aresiduals[1],1))
        if isnothing(p.factor)
            @test s.metric == :euclidean
            @test s.normal_residual_norm ≈ norm(A'*(b-A*x)) atol=1e-13
        else
            S = p.factor
            W = S'*S
            @test s.metric == :weighted
            @test s.normal_residual_norm ≈ norm(S*(W\(A'*(W*(b-A*x))))) atol=1e-12
            @test s.solution_norm ≈ norm(S*x)
            # Weighted LS stationarity and W-orthogonality to the nullspace.
            @test norm(A'*W*(b-A*x)) < 1e-8
            @test norm(nullspace(A)'*W*x) < 1e-8
        end
        if startswith(p.id,"CK")
            z, direct = cskewminares(A,b; rtol=1e-10)
            @test direct.solved
            @test z ≈ x atol=1e-8 rtol=1e-8
        end
    end
end

@testset "Extension geometry and edge cases" begin
    rng = MersenneTwister(905)
    # Direct checks at arbitrary x catch transformations that happen to return
    # the right answer while reporting the wrong convergence metric.
    A = ComplexF64[0 1+2im 2; -1-2im 0 im; -2 -im 0]
    b = randn(rng,ComplexF64,3)
    x = randn(rng,ComplexF64,3)
    B = [real.(A) imag.(A); imag.(A) -real.(A)]
    c = [real.(b);imag.(b)]
    y = [real.(x);-imag.(x)]
    @test B ≈ -B'
    @test norm(c-B*y) ≈ norm(b-A*x)
    @test norm(B'*(c-B*y)) ≈ norm(A'*(b-A*x))
    @test norm(y) ≈ norm(x)

    # Weighted and Euclidean least squares really differ on inconsistent data.
    S = [1. 1; 0 2]
    W = S'*S
    A = S\Diagonal([1.,0.])*S
    b = [0.,1.]
    for backend in (:reference,:krylov)
        x,s = weighted_minares(A,b,W; backend, rtol=1e-10)
        @test x ≈ [1.,0.] atol=1e-10
        @test norm(x-pinv(A)*b) > 0.5
        @test s.normal_residual_norm < 1e-10
        @test s.euclidean_normal_residual_norm > 1
        @test s.solved && s.metric == :weighted
    end

    for R in (Float32,Float64), backend in (:reference,:krylov)
        tol = R == Float32 ? 1f-4 : 1e-10
        H = R[2 1; 1 -1]
        b = R[1,2]
        J = Diagonal(R[1,-1])
        x,s = jhermitian_minares(J*H,b,J; backend,rtol=tol)
        @test eltype(x) == R && s.solved
        @test x ≈ (J*H)\b atol=tol rtol=tol
        theta = R(0.4)
        x,s = phase_minares(cis(theta)*H,b; theta,backend,rtol=tol)
        @test eltype(x) == Complex{R} && s.solved
        @test x ≈ (cis(theta)*H)\b atol=tol rtol=tol
        x,s = weighted_minares(H,b,Matrix{R}(I,2,2); backend,rtol=tol)
        @test eltype(x) == R && s.solved
        @test x ≈ H\b atol=tol rtol=tol
    end

    A = Matrix{Float64}(I,2,2)
    b = ones(2)
    for backend in (:reference,:krylov), rhs in (zeros(2),ones(2))
        for solve in ((A,b;kw...) -> phase_minares(A,b;theta=0.,kw...),
                      (A,b;kw...) -> jhermitian_minares(A,b,Matrix{Float64}(I,2,2);kw...),
                      (A,b;kw...) -> weighted_minares(A,b,Matrix{Float64}(I,2,2);kw...),
                      hamiltonian_minares, skewhamiltonian_minares, cskewminares_real)
            x,s = solve(zeros(2,2),rhs;backend)
            @test iszero(norm(x)) && s.solved
        end
    end
    @test_throws DimensionMismatch phase_minares(ones(2,3),b;theta=0)
    @test_throws DimensionMismatch weighted_minares(A,b,ones(3,3))
    @test_throws ArgumentError phase_minares(A,[NaN,1];theta=0)
    @test_throws ArgumentError phase_minares([Inf 0.;0 1],b;theta=0)
    @test_throws ArgumentError phase_minares(A,b;theta=NaN)
    @test_throws ArgumentError phase_minares(A,b;theta=1im)
    @test_throws ArgumentError phase_minares(A,b;theta=0.2)
    @test_throws ArgumentError phase_minares(A,b;theta=0,rtol=-1)
    @test_throws ArgumentError phase_minares(A,b;theta=0,maxiter=0)
    @test_throws ArgumentError phase_minares(A,b;theta=0,backend=:unknown)
    @test_throws ArgumentError jhermitian_minares(A,b,2A)
    @test_throws ArgumentError jhermitian_minares(A,b,[1. 1;0 -1])
    @test_throws ArgumentError jhermitian_minares([1. 1;0 1],b,A)
    @test_throws ArgumentError weighted_minares(A,b,Diagonal([1.,-1]))
    @test_throws ArgumentError weighted_minares(A,b,Diagonal([1.,0]))
    @test_throws ArgumentError weighted_minares(A,b,[1. 1;0 1])
    @test_throws ArgumentError weighted_minares([1. 1;0 1],b,A)
    @test_throws DimensionMismatch hamiltonian_minares(zeros(3,3),ones(3))
    @test_throws ArgumentError hamiltonian_minares(complex.(A),b)
    @test_throws ArgumentError hamiltonian_minares(A,b)
    @test_throws ArgumentError skewhamiltonian_minares([1. 0;0 -1],b)
    @test_throws ArgumentError cskewminares_real(A,b)
    _,s = phase_minares(Diagonal([1.,2.,3.]),ones(3);theta=0,maxiter=1,rtol=0)
    @test !s.solved && s.status == :failed_verification
end
