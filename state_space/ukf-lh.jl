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
abstract type ukf{T} <: kf{T}
end

mutable struct UKF{T} <: ukf{T}
	# θ holds referece to single cont. mem chunk for all variables
	@define(Vector{T}, θ,x̂,wᵐ,wᶜ,ŷ) 
	@define(Matrix{T}, P,Pʸʸ,Pˣʸ,Q,R,σ,σˣ,σʸ)
	@define(Function,F,H)
	@define(T, α,β,κ,λ)
	L::Int64        # state size as used in paper, first index = 1 !!!
end

## CTOR
# L - state size, N - output size
function UKF(x::Vector{T}, F::Function, H::Function, α::T = T(0.001), β::T = T(2.0), κ::T = T(0.0) ) where {T}
	L = length(x)
	N = length(H(x))
	@shared(T,θ, Matrix(L,2L+1,  σ,σˣ), Matrix(N,2L+1, σʸ), Vector(2L+1, wᵐ,wᶜ),
			Matrix(L,L,P), Matrix(N,N, Pʸʸ), Matrix(L,N, Pˣʸ), Matrix(L,L, Q), Matrix(N,N, R)  )
	# compute scaling parameter
	λ = α^2*(L+κ)-L

	# calculate weights
	wᵐ[1] = λ/(L+λ);  	wᶜ[1] = λ/(L+λ) + (1-α^2+β);		
	wᵐ[2:2L+1] = wᶜ[2:2L+1] = 1/(2*(L+λ)); 
	
	# predicted output
	ŷ = Vector{T}( N );
	return UKF(θ, x,wᵐ,wᶜ,ŷ, P,Pʸʸ,Pˣʸ,Q,R,σ,σˣ,σʸ, F,H, α,β,κ,λ, L)
end

####
# TIME-UPDATE
# 
# priory state estimate, which contains no information of current update
function predict( ukf::ukf{T}, u::Vector{T} ) where {T}
	# create meaningful aliases - compiler will optimize them out	
	@attach(ukf, P,Q,R,σ,σˣ,σʸ,Pʸʸ,Pˣʸ,x̂ )
	# compute sigma points from current x state and P covariance: 
	# x x+√P x-√P
	# x x+√P x-√P  

	σ  = sigma!(ukf)
	###
	# sampling from state transition:
	# state variable can be reused as mean, since σ points preserve state in the first column
	fill!(x̂, 0)
	# propagate sigma points through state transition 
	for i = 1:2L+1
		σˣ[:,i] = F( σ[:,i], u ) # propagating sigma points
		x̂[:] += wᵐ[i] * σˣ[:,i]  # the weighted average of projected points
	end
	## compute covariance of state projection
	# P =  FPF'
	fill!(P, 0)
	for i = 1:2L+1
		P[:,:] += wᶜ[i] * ( σˣ[:,i] - x̂ ) * (σˣ[:,i] - x̂)' 
	end
	# additive state noise noise
	P[:,:] += Q # diagm(rand(10))

	###
	# sampling from observation function:
	# draw NEW sigma points, then propagate them through state transition
	σ = sigma!(ukf)
	fill!(ŷ, 0)
	for i = 1:2L+1
		σʸ[:,i] = H( σ[:,i] ) 	# propagating sigma points
		ŷ[:] += wᵐ[i] * σʸ[:,i]  # the weighted average of projected points
	end

	# predicted mean and covariance
	return ŷ,Pʸʸ
end

####
# MEASUREMENT-UPDATE
#
# current state is combined with observation, to refine state
function update!( ukf::ukf{T}, z::Vector{T} ) where {T}
	@attach(ukf ,σˣ,σʸ, wᵐ,wᶜ, x̂,ŷ, P,Q,Pʸʸ,H,R,L,Pˣʸ)
	#
	fill!(Pʸʸ, 0)
	for i = 1:2L+1
		Pʸʸ[:,:] += wᶜ[i] * ( σʸ[:,i] - ŷ ) * (σʸ[:,i] - ŷ)' 
	end
	Pʸʸ[:,:] += R[:,:]

	fill!(Pˣʸ, 0)
	for i = 1:2L+1
		Pˣʸ[:,:] += wᶜ[i] * ( σˣ[:,i] - x̂ ) * (σʸ[:,i] - ŷ)'
	end

	K = Pˣʸ * inv(Pʸʸ)
	x̂ += K * (z - ŷ)
	P[:,:] -= K*Pʸʸ*K'
end

function sqrt_svd( P::Matrix{T}, L::Int64, λ::T ) where {T}
	U,D,V = svd(P)
	S = diagm( sqrt( (L+λ)* D ) )
	return U*S*U'
end

#sigma point generation
function sigma!( x::Vector{T}, P::Matrix{T}, σ::Matrix{T}, L::Int64, α::T, β::T, λ::T) where {T} 
	σ[:,1] = x
	#S = sqrt(L+λ)*full(chol( P ))
	S = sqrt_svd(P, L,λ)

	for i = 2:L+1
        σ[:,i]   = x + S[:,i-1]
        σ[:,i+L] = x - S[:,i-1]
    end
    return σ
end
sigma!(u::ukf) = sigma!(u.x̂, u.P, u.σ, u.L, u.α, u.β, u.λ )

