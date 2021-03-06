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


## CTOR
function UKF(L::Int64, N::Int64, α::T, β::T, κ::T ) where {T}
	@shared(T,θ, Matrix(L,2L+1,  σ,σˣ), Matrix(N,2L+1, σʸ), Vector(2L+1, wᵐ,wᶜ),
			Matrix(N,N, Pʸʸ), Matrix(L,N, Pˣʸ))
	# compute scaling parameter
	λ = α^2*(L+κ)-L

	# calculate weights
	wᵐ[1] = λ/(L+λ);  	wᶜ[1] = λ/(L+λ) + (1-α^2+β);		
	wᵐ[2:2L+1] = wᶜ[2:2L+1] = 1/(2*(L+λ)); 
	
	# predicted output
	ŷ = Vector{T}( n ); Pʸʸ = Matrix{T}(n,n); Pˣʸ = Matrix{T}(L,n)
	return UKF(x,P,  F,Q,  H,R,   α,β,κ,λ, σ,σˣ,σʸ,   wᵐ,wᶜ,   ŷ,Pʸʸ,Pˣʸ,   L)
end

function UKF(x::Vector{T}, F::Function, H::Function, R::Matrix{T};  p=T(1e-4), q=T(1e-4),  α=T(1e-3), β=T(2.0), κ=T(0.0)  ) where {T}
	n = length(x)
	P = p * diagm(ones(T,n)); Q = q * diagm(ones(T,n))
	UKF(x,P,F,Q,H,R, α,β,κ)
end



####
# TIME-UPDATE
# 
# priory state estimate, which contains no information of current update
function predict( ukf::UKF{T}, u::Vector{T} ) where {T}
	# create meaningful aliases - compiler will optimize them out	
	@attach(ukf, P,Q,R,σ,σˣ,σʸ,Pʸʸ,Pˣʸ )
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
function update!( ukf::UKF{T}, z::Vector{T} ) where {T}
	@attach(ukf, F,wᵐ,x̂,X,P,Q,Pʸʸ,H,wᶜ,ŷ,Y,R,L,Pˣʸ)
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





















