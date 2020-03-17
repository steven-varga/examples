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

import Base: show
import StatsBase: sample
export 	ss_observer_ar, ss_observer_ma, ss_observer_arma, ss_transition_ar,ss_transition_ma,ss_transition_arma,
		ss_covariance_arma,ss_covariance_arma,ss_covariance_arma,
		ss_ar,ss_ma,ss_arma, arma, sample, 
		posdef_block_diag, update_Q, update_R

"""
	F = state_transition_ma( m::Int64, d::Int64)

	m - number of instruments, d - degree of auto regression
	returns a [m*d x m*d] dimension sparse block permutation 
			   matrix shifting rows down within d size block 
"""
function ss_transition_ma( m::Int64, n::Int64 )
	## state F 
	A = spzeros(m*n,m*n) 
	idx =  diagind(A)[ filter((x)-> x % n !=0, 1:n*m-1) ] .+ 1
	A[idx] = 1.0
	return A
end
"""
	F = ss_transition_ra( m::Int64, d::Int64)

	m - number of instruments, d - degree of auto regression
	returns a [m*d x m*d] dimension sparse block permutation 
			   matrix shifting rows down within d size block 
"""
function ss_transition_ar( m::Int64, n::Int64 )
	## state F 
	A = spzeros(m*n,m*n) 
	idx =  diagind(A)[ filter((x)-> x % n !=0, 1:n*m-1) ] .+ 1
	A[idx] = 1.0
	for i=1:n:m*(n-1)
		A[i,i:i+n-1] = .4*rand(n)
	end
	return A
end
function ss_transition_ar( p::Matrix{Float64} )
	m,n = size(p)
	## state F 
	A = spzeros(m*n,m*n) 
	idx =  diagind(A)[ filter((x)-> x % n !=0, 1:n*m-1) ] .+ 1
	A[idx] = 1.0
	k = 1
	for i=1:n:m*(n-1)
		A[i,i:i+n-1] = p[k,:]
		k = k + 1
	end
	return A
end

"""
	H = ss_observer_ma( m::Int64, d::Int64)

	m - number of instruments, d - degree of moving average
	returns a [m*d x m] dimension sparse block matrix where 
	the nzval values are the MA coefficients 
"""
function ss_observer_ma( m::Int64, n::Int64 )
	H = spzeros(m*n,m)
	i=1
	for j = 1:m
		H[i,j] = 1.0
		i = i+n
	end
	return H'
end
function ss_observer_ma( q::Matrix{Float64} )
	m,n = size( q )
	k = n+1; i=1
	H = ss_observer_ma(m,k)
	
	for j = 1:m
		H[j,i+1:i+k-1 ] = q[j,:]
		i = i+k	
	end
	return H
end
"""
	H = ss_observer_ar( m::Int64, d::Int64)

	m - number of instruments, d - degree of auto regression
	returns a [m*d x m] dimension sparse block matrix where 
	the nzval values are 1.0 0.0 ... 0.0 picking off the AR prediction
	from the top of state x
"""
function ss_observer_ar( m::Int64, n::Int64 )
	B = spzeros(m*n,m)
	i=0
	for j = 1:m
		B[i+1,j] = 1.0
		i = i+n
	end
	return B'
end



function block_diag( m::Int64, n::Int64 )
	Q = spzeros(Float64,m*n,m*n)
	for i=1:n:n*m
		Q[i:i+n-1,i:i+n-1] = ones(n,n)
	end
	return Q
end
"""
	P = posdef_block_diag( m::Int64, b::Int64) 
	∵ P = Q'Q

	m - number of blocks, b - block size
	returns a [m*b x m*b] dimension sparse positive definite block matrix where 
	the nzval values are the decoupled covariances
"""
function posdef_block_diag(m::Int64, n::Int64)
	Q = tril(block_diag(m,n))
	n = length( Q.nzval ) 
	Q.nzval[:] = rand(n) ./ 100.0 
	return Q'Q
end


function ss_ar(m::Int64, n::Int64)
	F = ss_transition_ar(m,n)
	H = ss_observer_ar(m,n)
	P = posdef_block_diag(m,n)
	x,y = rand(m*n),rand(m)
	return (x,y,F,H,P)
end
function ss_ma(m::Int64, n::Int64)
	F = ss_transition_ma(m,n)
	H = ss_observer_ma(m,n)
	P = posdef_block_diag(m,n)
	x,y = rand(m*n),rand(m)
	return (x,y,F,H,P)
end
function ss_arma(m::Int64, n::Int64)
	F = ss_transition_ar(m,n)
	H = ss_observer_ma(m,n)
	P = posdef_block_diag(m,n)
	x,y = rand(m*n),rand(m)
	return (x,y,F,H,P)
end








