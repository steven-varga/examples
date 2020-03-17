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

###
# x - nx1 z = px1 F=nxn H=mxn w=nx1 Q=nxn R=mxm P=nxn K=nxm, S=mxm, M=nxn
#
mutable struct KF{T} <: kf{T}
	@define(Vector{T}, x,y) 
	# θ  placeholder for all state variables defined below
	# x  nx1 initial state value, with mean and covariance
	@define(SparseMatrixCSC{T,Int64}, P,F,M, H,K,S)
	# F state transition matrix which is applied to the previous state Xₜ-₁
	# w ~ ℕ(0,ℚ) GRV drawn from 0 mean, Q covariance
	# H observation transition matrix applied on x̂ state
	# v ~ ℕ(0,ℚ) GRV drawn from 0 mean, R covariance
	@define(ZeroMeanDiagNormal, w,v)
	@define(PDiagMat{T,Vector{T}},  Q,R)
end
function KF( T::DataType )
	w,v = MvNormal([0.]), MvNormal([0.])
	Q,R = w.Σ,v.Σ
	return KF{T}(
	#= x,y =# Vector{T}(), Vector{T}(), 
	#= P,F,M: =# spzeros(T,0,0), spzeros(T,0,0), spzeros(T,0,0),
	#= H,K,S: =# spzeros(T,0,0), spzeros(T,0,0), spzeros(T,0,0),
	#= w,v  : =# w,v, Q,R)
end
# returns x̂ estimate and P covariance estimate
function predict( m::KF{T} ) where {T}
	@attach(m, x,P,F,Q)
	x[:] = F * x             	# x = F*x + B*u -- mean of gaussian, B = 0 -- no control signal
	P[:,:] = F * P * F' + Q   	# P = F*P*Fᵀ + Q  w ~ ℕ(0,ℚ)
	nothing
end

function update!(this::KF, z::Vector{T} ) where {T} # zₘₓ₁
	@attach(this, x,y,P,H,K,S,M,R)
	
	y[:] = z - H * x           	# y  =  z - H * xₖ-₁
	S[:] = H * P * H' + R 		# S  =  Hₖ * Pₖ * Hᵀ + Rₖ
	sq = cholfact( Symmetric(S) )
	Si  = sq\speye(length(y))
	K[:] = P * H' * Si 			# Kₖ =  Pₖ * Hᵀ * inv(S)
	x[:] = x + K * y            # x  =  x + K * y
	M[:] = (eye(P) - K * H) * P # stable Joseph's formula
	P[:] = M*M' + K*R*K'  
	nothing
end

function resize!( this::KF{T}, instruments::Int64, ar_degree::Int64, ξ::T=T(1e-8) ) where {T}
	n,m,d = instruments*ar_degree,instruments,ar_degree
	# x=nx1 y=mx1 F=nxn H=mxn w=nx1 Q=nxn R=mxm, S=mxm, P=nxn K=nxm
	x,y = Vector{T}(n), Vector{T}(m)
	F,H,P = state_transition_arma(m,d), state_observer_arma(m,d), state_covariance_arma(m,d)
	S,K,M = spzeros(m,m), spzeros(n,m), spzeros(n,n)
	w,v = MvNormal( ξ * rand(n) ), MvNormal( ξ * rand(m) )
	Q,R = w.Σ,v.Σ
	@set(this, x,y,F,H,P,S,K,M, w,v,Q,R)
end
size( this::KF{T} ) where {T} = size( this.K )


function similar( this::KF{T} ) where {T}
	@attach(this, x,y, F,H,P, Q,R,S,K,M)
	return KF{T}(similar(x),similar(y),
				 similar(F),similar(H),similar(P),similar(Q),similar(R),similar(S),similar(K),similar(M) )
end
function copy( this::KF{T} ) where {T}
	@attach(this, x,y, F,H,P, Q,R,S,K,M)
	return KF{T}(copy(x),copy(y),
				 copy(F),copy(H),copy(P),copy(Q),copy(R),copy(S),copy(K),copy(M) )
end




