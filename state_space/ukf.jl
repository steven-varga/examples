#=
   ALL RIGHTS RESERVED.
   _________________________________________________________________________________
   NOTICE: All information contained  herein is, and remains the property  of  Varga
   Consulting and  its suppliers, if  any. The intellectual and  technical  concepts
   contained herein are proprietary to Varga Consulting and its suppliers and may be
   covered  by  Canadian and  Foreign Patents, patents in process, and are protected
   by  trade secret or copyright law. Dissemination of this information or reproduc-
   tion  of  this  material is strictly forbidden unless prior written permission is
   obtained from Varga Consulting.

   Copyright © <2018> Varga Consulting, Toronto, On          info@vargaconsulting.ca
   _________________________________________________________________________________
=#


##
# references:
# 1) The Unscented Kalman Filter for Nonlinear Estimation, Eric A. Wan and Rudolph van der Merwe
# 2) THE SQUARE-ROOT UNSCENTED KALMAN FILTER FOR STATE AND PARAMETER-ESTIMATION, Rudolph van der Merwe and Eric A. Wan, 
# 3) Low Rank Updates for the Cholesky Decomposition, Matthias Seeger, 2008 
# 4) A New Extension of the Kalman Filter to Nonlinear Systems, Simon J. Julier Jeffrey K. Uhlmann


abstract KF{T}
struct UKF_LF{T} <: KF{T}
	@define(Matrix{T}, P,Pʸʸ,Pˣʸ,Q,R,σ,σˣ,σʸ)
	@define(Vector{T}, x,wᵐ,wᶜ,ŷ)
	@define(Function,H)
	@define(T, α,β,κ,λ,γ)
	L::Int64        # size as used in paper, first index = 1 !!!
end
struct UKF{T} <: KF{T}
	@define(Matrix{T},P,Pʸʸ,Pˣʸ,Q,R,σ,σˣ,σʸ)
	@define(Vector{T}, x,wᵐ,wᶜ,ŷ)
	@define(Function, F,H)
	@define(T, α,β,κ,λ,γ)
	L::Int64        # size as used in paper, first index = 1 !!!
end


function sqrt_svd{T}( P::Matrix{T}, L::Int64, λ::T )
	U,D,V = svd(P)
	S = diagm( sqrt( (L+λ)* D ) )
	return U*S*U'
end

#sigma point generation
function sigma!{T}( x::Vector{T}, P::Matrix{T}, σ::Matrix{T}, L::Int64, α::T, β::T, λ::T  )
	σ[:,1] = x
	S = sqrt_svd(P, L,λ)

	for i = 2:L+1
        σ[:,i]   = x + S[:,i-1]
        σ[:,i+L] = x - S[:,i-1]
    end
    return σ
end
sigma!{T}(u::UKF{T}) = sigma!(u.x, u.P, u.σ, u.L, u.α, u.β, u.λ )

include("ukf-lf.jl")

## tests: 





