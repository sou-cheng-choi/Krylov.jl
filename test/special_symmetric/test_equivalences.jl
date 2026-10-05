# Exact-arithmetic consequences of the trial-space structure: skew parity
# reductions to LSMR/LSQR, the consistent-case trial range, and the odd-step
# range-start bound. The full-basis implementation and Krylov.jl use
# independent projected solves.
@testset "Skew parity equivalences to LSMR and LSQR" begin
    rng = MersenneTwister(20261006)
    for trial in 1:6, complex_case in (false, true)
        T = complex_case ? ComplexF64 : Float64
        M = randn(rng, T, 18, 18)
        A = M - transpose(M)
        b = randn(rng, T, 18)
        structure = complex_case ? :cskew : :ss
        _, s = projected_solve(A, b; structure, maxiter=10, ranktol=0, rtol=0)
        if !complex_case
            _, sr = projected_solve(A, b; structure, objective=:residual,
                                    maxiter=10, ranktol=0, rtol=0)
        end
        for k in 1:10
            j = fld(k, 2)
            kw = (; itmax=j, atol=0.0, btol=0.0, axtol=0.0,
                    rtol=0.0, etol=0.0, conlim=0.0)
            xls = j == 0 ? zero(b) : first(Krylov.lsmr(A, b; kw...))
            @test norm(s.iterates[k] - xls) <= 1e-10 * max(1, norm(xls))
            if !complex_case
                xqr = j == 0 ? zero(b) : first(Krylov.lsqr(A, b; kw...))
                @test norm(sr.iterates[k] - xqr) <= 1e-10 * max(1, norm(xqr))
            end
        end
    end
end

@testset "Consistent trial range and odd-step range-start bound" begin
    rng = MersenneTwister(20261007)
    for trial in 1:6
        U = Matrix(qr(randn(rng, ComplexF64, 18, 18)).Q)
        A = U * Diagonal(vcat(collect(range(0.5, 2; length=12)), zeros(6))) * transpose(U)
        bc = A * randn(rng, ComplexF64, 18)
        x, s = csminares(A, bc; rtol=1e-12, completion=:stationary)
        target = pinv(A; rtol=1e-12) * bc
        @test s.solved
        @test norm(x - target) <= 1e-10 * max(1, norm(target))
        Pnull = I - pinv(A; rtol=1e-12) * A
        @test all(v -> norm(Pnull*v) <= 1e-10 * max(1, norm(v)), s.iterates)
        b = randn(rng, ComplexF64, 18)
        _, rs = csminares_range(A, b; maxiter=9, completion=:invariant, ranktol=0, rtol=0)
        for j in 1:5
            xl, _ = Krylov.lsmr(A, b; itmax=j, atol=0.0, btol=0.0,
                               axtol=0.0, rtol=0.0, etol=0.0, conlim=0.0)
            lhs = norm(A' * (b - A * rs.iterates[2j-1]))
            rhs = norm(A' * (b - A * xl))
            @test lhs <= rhs + 1e-10 * norm(A' * b)
        end
    end
end
