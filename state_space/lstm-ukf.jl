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
using Distributions
export forward,LSTM,initialization,σ,dσ,dtanh

struct LSTM{T}
	@define( Vector{T}, θ,a)
	@define( Matrix{T},	wⁱ,wᶠ,wᵒ,wᶻ,Rⁱ,Rᶠ,Rᵒ,Rᶻ,wʸ)
	@define( Vector{T}, pⁱ,pᶠ,pᵒ,bⁱ,bᶠ,bᵒ,bᶻ, i,f,o,z,c,h,y ) 
	@define( Function,  σ,ϱ,ϰ )
end



σ( x::Vector{T} ) where {T}   = 1.0 ./ (1.0 + exp.( -x ) )
dσ(x::Vector{T} ) where {T}   = σ(x) .* (1.0 - σ(x))
dtanh(x::Vector{T}) where {T} = 1.0 .-  tanh.(x).^2


#############
# a = unsafe_wrap(Array, pointer(A,4),3,false)
# 46 -- 
# j - blocks, k - streams/minibatch, m - input, n - output

function LSTM(T::DataType,M::Int64,N::Int64,K::Int64,  σ::Function,ϱ::Function,ϰ::Function )
	# nxm 
	@shared(T,θ,
		Matrix(N,M,  wⁱ,wᶠ,wᵒ,wᶻ), Matrix(N,N,  Rⁱ,Rᶠ,Rᵒ,Rᶻ),
		Vector(N, pⁱ,pᶠ,pᵒ,bⁱ,bᶠ,bᵒ,bᶻ),Matrix(N,K, wʸ))
	@shared(T,a, Vector(N,  i,f,o,z,c,h), Vector(K, y) )  
	LSTM(θ,a, 
		wⁱ,wᶠ,wᵒ,wᶻ, Rⁱ,Rᶠ,Rᵒ,Rᶻ,wʸ, pⁱ,pᶠ,pᵒ,bⁱ,bᶠ,bᵒ,bᶻ,  i,f,o,z,c,h,y, σ,ϱ,ϰ)
end

import Base.size

function size(x::LSTM)
	length(x.θ),length(x.a)	
end

###
# training :
# 


function initialization(lstm::LSTM, F::Function = x ->sqrt(6.)/sqrt(x) )
	@attach(lstm, 
			wⁱ,wᶠ,wᵒ,wᶻ,Rⁱ,Rᶠ,Rᵒ,Rᶻ, pⁱ,pᶠ,pᵒ,bⁱ,bᶠ,bᵒ,bᶻ)
	m,n = F(sum(size(wᵢ))), F(size(Rⁱ,1))  
	@rand!( Uniform(-m,m),  wⁱ,wᶠ,wᵒ,wᶻ, pⁱ,pᶠ,pᵒ) 
	@rand!( Uniform(-n,n), Rⁱ,Rᶠ,Rᵒ,Rᶻ) 
	pⁱ,pᶠ,pᵒ
	# LSTM in RNN by Gers (2001)  pg:38 however
	# bᶠ = 5.0 by 
	@fill!(0.0, bⁱ,bᶻ); fill!(bᶠ, 5.0); fill!(bᵒ, +2.0) 
end

# u - input here, as there is internal state 'z'
function forward(lstm::LSTM{T}, u::Vector{T} ) where {T}
	@attach(lstm, 
			wⁱ,wᶠ,wᵒ,wᶻ,wʸ, Rⁱ,Rᶠ,Rᵒ,Rᶻ, pⁱ,pᶠ,pᵒ,bⁱ,bᶠ,bᵒ,bᶻ,
			i,f,o,z,c, h,y, σ,ϱ,ϰ )
	
	z[:] = ϱ( wᶻ*u+Rᶻ*h+bᶻ )
	i[:] = σ( wⁱ*u+Rⁱ*h+bⁱ + pⁱ.*c )
	f[:] = σ( wᶠ*u+Rᶠ*h+bᶠ + pᶠ.*c ) 
	c[:] = z .* i + c .* f
	o[:] = σ( wᵒ*u+Rᵒ*h+bᵒ + pᵒ.*c )
	h[:] = ϰ(c) .* o 
	
	y[:] = wʸ'h
end

#=
x = rand(5,1000)
print("001 ")
f{T}(x::Vector{T}) = tanh.(x)

lstm = @time LSTM(Float64,5,800,2, σ, f , f )
println( size(lstm) )


print("002 ")
@time for i in 1:1000
	forward(lstm,x[:,i])  
end
=#
