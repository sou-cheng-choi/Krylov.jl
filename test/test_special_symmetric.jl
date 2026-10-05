# A module keeps these helpers out of the namespace shared by the other test files.
module TestSpecialSymmetric

using Test, LinearAlgebra, Random, SparseArrays
import Krylov
using Krylov.SpecialSymmetric

include("special_symmetric/utils.jl")

@testset "special_symmetric" begin
  include("special_symmetric/test_projected.jl")
  include("special_symmetric/test_extensions.jl")
  include("special_symmetric/test_native_qlp.jl")
  include("special_symmetric/test_equivalences.jl")
  include("special_symmetric/test_givens.jl")
  include("special_symmetric/test_recurrence.jl")
end

end
