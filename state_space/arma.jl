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

import analytics: fit!
export arma,set_state,set_φ,set_ϑ,set_Q,set_R,
	FH2w,w2FH,a2H
"""
sets state variables of DUAL KF from passes ARMA(φ,ϑ) model parameters  
"""
function set_state(x::Vector{T}, φ::Matrix{T}, ϑ::Matrix{T}) where {T}
	m,d = size(φ)
	if length(x) ≠ m*(2d - 1)
		error( "modelparamters of φ and ϑ degree doesn't match" ); end
	for i = 0:m-1
		k = i*(2d-1)
		for j = 1:d
			@inbounds x[k+j] = φ[i+1,j]
		end
		for j = 1:d-1
			@inbounds x[k+j+d] = ϑ[i+1,j]
		end
	end
end
"""
sets φ model parameters of ARMA to F state transition matrix of PRIMAL KF
"""
function set_φ( F::AbstractMatrix, φ::AbstractMatrix )
	m,p = size(φ); n = ncols(F)
	for i = 0:m-1
		for j = 1:p
			k = i*p
			@inbounds F[k+1,k+j] = φ[i+1,j]
		end
	end
	return nothing
end
"""
sets ϑ model parameters of ARMA to H observation matrix of PRIMAL KF
"""
function set_ϑ( H::AbstractMatrix, ϑ::AbstractMatrix )
	m,n = size(H); p = ncols(ϑ) + 1
	for i = 0:m-1
		for j = 2:p
			k= i*p+j
			@inbounds H[i+1,k] = ϑ[i+1,j-1]
		end
	end
	return nothing
end
function set_Q( w::ZeroMeanDiagNormal, ω::Vector )
	m = div( length(w), length(values))
	w.Σ.diag[1:m:end] = ω
	nothing
end
function set_R( v::ZeroMeanDiagNormal, ν::Vector )

end
"""
copies KFᵃ F state transition that hold φ and H observation with ϑ
into KFʷ state w
"""
function FH2w( F::AbstractMatrix, H::AbstractMatrix, w::AbstractVector )
	m,n = size(H); p = div( length(w)+m,2m )
	for i = 0:m-1
		for j = 0:p-1
			k = i*p+1
			a = i*(2p-1)+j+1
			@inbounds w[a] = F[k,k+j]
		end
		for j = 2:p
			k = i*p+j
			a = i*(2p-1)+j+p-1
			@inbounds w[a] = H[i+1,k]
		end
	end
	return nothing
end
"""
copies KFʷ x state, the model parameters: φ,ϑ into KFᵃ F state transition
and H observation matrix
complexity: O(n)
"""
function w2FH( w::AbstractVector, F::AbstractMatrix, H::AbstractMatrix)
	m,n = size(H); p = div( length(w)+m,2m )
	for i = 0:m-1
		for j = 0:p-1
			k = i*p+1
			a = i*(2p-1)+j+1
			@inbounds F[k,k+j] = w[a]
		end
		for j = 2:p
			k = i*p+j
			a = i*(2p-1)+j+p-1
			@inbounds H[i+1,k] = w[a]
		end
	end
	return nothing
end
"""
copies KF₁ x state vector into KF₂ H observation function of an
ARMA model
complexity: O(n)
"""
function a2H( a::AbstractVector, H::AbstractMatrix )
	m,q = size(H)
	p = div( length(a), m)
	for i = 0:m-1
		j = 1; k = 0
		while j < p 
			k = i*(2p-1) + j
			@inbounds H[i+1,k] = H[i+1,k+p] = a[i*p+j]
			j += 1
		end
		@inbounds H[i+1, k+1] = a[i*p+j]
	end
	return nothing
end

struct arma{T}
	kfᵃ::KF{T}
	kfʷ::KF{T}
end

"""
arma{T}( p::Matrix{T}, q::Matrix{}; state::Matrix{T} = rand(length(p))  ) 
	returns with an ARMA model with a given initial state
	that can be used to draw samples from
	see also:
	sample( x::arma )
	
"""
function arma(p::Matrix{T}, q::Matrix{T}, wᵢ::Vector{T}; state = rand( length(p) ), noise_level::T=T(1e-5) ) where {T}
	# x=nx1 y=mx1 F=nxn H=mxn w=nx1 Q=nxn R=mxm, S=mxm, P=nxn K=nxm
	m,d = size(p)  # number of output X AR and MA degree
	kfᵃ,kfʷ = KF(T),KF(T)
	begin # primal KF for state estimation
		n = m*d        # system matrix size 
		x,y = state, Vector{T}(m)
		F,H,P = ss_transition_ar(p), ss_observer_ma(q), posdef_block_diag(m,d)
		S,K,M = spzeros(m,m), spzeros(n,m), spzeros(n,n)
		w,v = MvNormal( noise_level*rand(n)), MvNormal(noise_level*rand(m))
		Q,R = w.Σ,v.Σ
		# set covariance
		Q.diag[1:d:end] = wᵢ
		@set(kfᵃ, x,y,F,H,P,w,v,Q,R,S,K,M)
	end
	begin # dual KF for model parameter estimation
		n = 2d-1
		x,y = Vector{T}(m*n), Vector{T}(m)
		F,H,P = spdiagm(ones(m*n)), ss_observer_ma(m,n), posdef_block_diag(m,n)
		S,K,M = spzeros(m,m), spzeros(m*n,m), spzeros(n*m,n*m)
		w,v = MvNormal( noise_level*rand(m*n)), MvNormal(noise_level*rand(m))
		Q,R = w.Σ,v.Σ
		@set(kfʷ, x,y,F,H,P,w,v,Q,R,S,K,M)
	end
	model =  arma(kfᵃ,kfʷ)
	return model
end

function show(io::IO, model::arma{T}) where {T}
	@attach(model.kf,x,y, F,H,P,Q,R)
	m,n = size(H)
	Base.Printf.@printf(io,"ARMA with DUAL KF                     :\n")
	Base.Printf.@printf(io,"---------------------------------------\n")
end

function sample( this::KF{T}, n::Int64 ) where {T}
	@attach(this, F,H,w,v)
	x = copy(this.x) # save current state
	Z = Matrix{T}( length(this.y) ,n)	
	for i =1:n
		x[:]   = F*x + rand(w)
		Z[:,i] = H*x + rand(v)
	end
	return Z
end

function sample( this::arma{T}, n::Int64 ) where {T}
	@attach(this, kfᵃ)
	sample(kfᵃ,n)
end

function fit!(this::arma{T}, A::Matrix{T} ) where {T}
	@attach(this, kfᵃ,kfʷ)
	d = length(kfᵃ.x)
	εᵃ,εʷ,n = kfᵃ.y,kfʷ.y, kfʷ.y
	
	# set initial condition
	kfᵃ.x[:] = A[1:d] 			# initialize primal KF state
	FH2w(kfᵃ.F, kfᵃ.H, kfʷ.x) 	# move model parameters to dual KF state
	a2H(kfᵃ.x, kfʷ.H) 			# and xₖ₋₁ state to dual observer
	# now KFᵃ and KFʷ describe the same problem and predict the same yₜ₊₁ output

	for i = d+1:ncols(A)
		z = vec( A[:,i] )
		predict( kfᵃ ) # roll over to t+1 state
		# predict( kfʷ ) # this is the same since F = I
		update!(kfᵃ, z)  # clean state: a
		update!(kfʷ, z)  # model parameters: w 
		w2FH(kfʷ.x, kfᵃ.F, kfᵃ.H) 	# move estimated model parameters to primal KF
		B[:,i] = kfᵃ.H * kfᵃ.x
	end
	nothing
end


