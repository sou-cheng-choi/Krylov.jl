@testset "Native CS-MINRES-QLP reflections" begin
    # Check the defining identities, including zeros and extreme finite scales.
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        pairs = [(T(0), T(0)), (T(-2), T(0)), (T(0), T(-3)), (T(2), T(-3))]
        if T <: Complex
            append!(pairs, [(T(2+3im), T(-4+im)), (T(0), T(2im)), (T(3im), T(0))])
        end
        for scale in (sqrt(floatmin(R)), one(R), sqrt(floatmax(R))/4), (a0,b0) in pairs
            a, b = scale*a0, scale*b0
            c, s, radius = SpecialSymmetric.cs_symortho(a, b)
            Q = [c s; conj(s) -c]
            @test all(isfinite, Q) && isfinite(radius)
            @test Q'*Q ≈ I atol=10eps(R)
            @test norm(Q*[a,b]-[radius,zero(T)]) <= 20eps(R)*max(norm([a,b]),floatmin(R))
        end
    end
end

@testset "Native CS-MINRES-QLP and independent residual reference" begin
    rng = MersenneTwister(20261006)
    for T in (Float64, ComplexF64), n in (4, 7), rank in (n, n-2)
        U = Matrix(qr(randn(rng,T,n,n)).Q)
        A = U*Diagonal([1 .+ collect(1:rank)/n; zeros(n-rank)])*transpose(U)
        for b in (randn(rng,T,n), A*randn(rng,T,n))
            x, s = csminresqlp(A,b; rtol=1e-10)
            xr, ref = csminres(A,b; rtol=1e-10, factorization=:svd)
            target = pinv(A; rtol=1e-12)*b
            @test s.solved
            @test s.backend_kind == :native_qlp
            @test x ≈ target rtol=1e-8 atol=1e-9
            @test x ≈ xr rtol=1e-8 atol=1e-9
            @test length(s.residuals) == length(s.aresiduals) == s.niter+1
            @test length(s.iterates) == s.niter
            @test s.basis_products == s.niter
            @test s.diagnostic_products == 1+2s.niter
            @test s.residuals[end] ≈ norm(b-A*x) atol=1e-13
            @test s.aresiduals[end] ≈ norm(A'*(b-A*x)) atol=1e-13
            @test all(diff(s.residuals) .<= 1e-9*norm(b))
            # Independent reorthogonalized/SVD oracle at each common dimension.
            # This detects conjugation/reflection errors hidden by terminal solves.
            for k in 1:min(s.niter, ref.niter)
                # Near-singular intermediate projections amplify differences
                # between QLP reflections and SVD; terminal vectors are checked
                # more tightly above, and residual objectives separately here.
                @test s.iterates[k] ≈ ref.iterates[k] rtol=1e-7 atol=2e-9
                @test norm(b-A*s.iterates[k]) ≈ norm(b-A*ref.iterates[k]) atol=1e-10*norm(b)
            end
            y, nohistory = csminresqlp(A,b; rtol=1e-10, history=false)
            @test isempty(nohistory.iterates)
            @test y ≈ x
            @test nohistory.residuals == s.residuals
        end
    end
end

@testset "Native QLP nullspace and singular termination" begin
    A, b = Diagonal([1.,0.]), [1.,1.]
    early, se = csminresqlp(A,b; completion=:stationary, rtol=1e-12)
    full, sf = csminresqlp(A,b; rtol=1e-12)
    @test se.niter == 1 && se.solved && !se.closed
    @test early ≈ [1.,1.]
    @test sf.closed && sf.solved && sf.niter == 2
    @test sf.discarded_pivots == 1
    @test full ≈ [1.,0.] atol=1e-14
    @test norm(b-A*full) ≈ norm(b-A*early)
    @test norm(full) < norm(early)
    # A*r and A'*r differ: the normal-residual diagnostic must use the latter.
    A = ComplexF64[1 im; im -1]
    for b in (ComplexF64[1,im], ComplexF64[1,0])
        x,s = csminresqlp(A,b; rtol=1e-12)
        @test s.solved && all(isfinite,x)
        @test x ≈ pinv(A)*b atol=1e-12
        @test s.aresiduals[end] ≈ norm(A'*(b-A*x)) atol=1e-14
    end
    for A in (zeros(3,3), zeros(ComplexF64,3,3)), b in (zeros(3),ones(3))
        x,s = csminresqlp(A,b)
        @test iszero(x) && s.solved && s.closed && s.niter==0
    end
    x,s = csminresqlp(Diagonal([1.,0.]),[0.,1.])
    @test iszero(x) && s.status==:stationary_zero
    x,s = csminresqlp(reshape([2im],1,1),ComplexF64[3+im])
    @test x ≈ [(3+im)/(2im)] && s.closed && s.niter==1
end

@testset "Native QLP precision, scale, shifts and operators" begin
    for T in (Float32, ComplexF32)
        A = T[3 1;1 2]
        T <: Complex && (A .*= T(1+im))
        b = A*T[1,-2]
        x,s = csminresqlp(A,b)
        @test eltype(x)==T
        @test s.solved
        @test x ≈ T[1,-2] rtol=1e-4
    end
    for scale in (1e-100,1.,1e100)
        A = scale*ComplexF64[2+im 1;1 3-im]
        b = A*ComplexF64[1,-2im]
        x,s = csminresqlp(A,b; rtol=1e-10)
        @test s.solved
        @test x ≈ [1,-2im] rtol=1e-9
    end
    A = spdiagm(-1=>fill(-1+0.1im,5),0=>fill(3+im,6),1=>fill(-1+0.1im,5))
    b = ComplexF64[1,2im,3,4im,5,6im]
    for op in (A,ProductOnly(A)), shift in (0,0.2,0.3im)
        x,s = csminresqlp(op,b; check=false,shift,rtol=1e-10)
        B = Matrix(A)-shift*I
        @test s.solved
        @test x ≈ B\b rtol=1e-8
        @test s.residuals[end] ≈ norm(b-B*x) atol=1e-13
        @test s.aresiduals[end] ≈ norm(B'*(b-B*x)) atol=1e-13
    end
    # A shift making the system singular changes the target and normal action.
    A = Diagonal(ComplexF64[1+im,2+im,3+im])
    b = ComplexF64[1,2im,3]
    x,s = csminresqlp(A,b; shift=2+im,rtol=1e-10)
    @test s.solved
    @test x ≈ pinv(Matrix(A)-(2+im)*I)*b atol=1e-10
    _,s = csminresqlp(Matrix(A),b; maxiter=1,rtol=0,atol=0)
    @test !s.solved && s.status==:iteration_limit && s.niter==1
    # Exercise the original low-storage recurrence on a well-conditioned system.
    A = ComplexF64[3+im 1;1 2-im]
    b = ComplexF64[1,2im]
    x,s = csminresqlp(A,b;reorthogonalize=false,history=false,rtol=1e-10)
    @test !s.reorthogonalized && isempty(s.iterates) && s.solved
    @test x ≈ A\b atol=1e-10
    # Structure-preserving congruence can be supplied explicitly (not an M API).
    C = Diagonal([0.5,2.])
    y,s = csminresqlp(C\A/C,C\b;rtol=1e-10)
    @test s.solved && transpose(C\A/C) ≈ C\A/C
    @test C\y ≈ A\b atol=1e-10
end

@testset "Native QLP validation" begin
    @test_throws DimensionMismatch csminresqlp(ones(2,3),ones(2))
    @test_throws DimensionMismatch csminresqlp(ones(2,2),ones(3))
    @test_throws ArgumentError csminresqlp(zeros(0,0),zeros(0))
    @test_throws ArgumentError csminresqlp([1. 2;0 1],ones(2))
    @test_throws ArgumentError csminresqlp([Inf 0.;0 1],ones(2))
    @test_throws ArgumentError csminresqlp(ProductOnly([Inf 0.;0 1]),ones(2);check=false)
    @test_throws ArgumentError csminresqlp(ones(2,2),[NaN,1.])
    for kw in ((;rtol=-1), (;atol=Inf), (;ranktol=NaN), (;breakdown_tol=-1),
               (;maxiter=0), (;maxiter=1.2), (;completion=:unknown),
               (;shift=Inf), (;shift="bad"), (;history=1), (;reorthogonalize=1))
        @test_throws ArgumentError csminresqlp(ones(2,2),ones(2);kw...)
    end
    # Do not silently accept unsupported legacy preconditioners.
    @test_throws MethodError csminresqlp(ones(2,2),ones(2);M=I)
end

@testset "Native QLP archived MATLAB help examples" begin
    for (name,A,b) in legacy_examples()
        x,s = csminresqlp(A,b;rtol=1e-10,history=false)
        target = pinv(Matrix(A);rtol=1e-12)*b
        @test s.solved
        @test norm(x-target) <= 1e-8*max(norm(target),1)
        @test s.normal_residual_norm ≈ norm(A'*(b-A*x)) atol=1e-12
        @test s.niter <= length(b)
        if name=="block_stencil"
            # Continuing through multiple numerical zero pivots used to add a
            # large nullspace component despite a tiny verified normal residual.
            @test s.status==:rank_truncated
            @test !s.closed && s.discarded_pivots==1
        end
    end
end
