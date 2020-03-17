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
export P002
"""
	P002{T}( γ::Float64,  budget::Float64, cov=true, max_short::Float64, max_long::Float64 )

	This strategy caps maximum short and long positions
"""
mutable struct P002{T} <: Portfolio{T}
	budget::Float64
	γ::Float64
	A::SparseMatrixCSC{Float64,Int}
	b::Vector{Float64}
	c::Vector{Float64}
	K₁::ConeVec
	K₂::ConeVec
	model
	covariance::Bool
	max_short::Float64
	max_long::Float64
	## CTOR
	function P002{T}(solver::AbstractMathProgSolver; γ::Float64=500.0,
					 budget=1e6, cov=true, max_short=2e4, max_long=2e4 ) where {T<:Real}	
		b=Vector{Float64}(); c=Vector{Float64}()
		A=Matrix{Float64}(0,0)
		K₁ = ConeVec(); K₂=ConeVec()
		model = ConicModel(solver)

		new( budget,γ, A,b,c,K₁,K₂, model, cov, max_short, max_long )
	end

end

String(st::P002) = @sprintf("budget: %.2f risk: %.2f", st.budget, st.γ )

function resize!(x::P002, m::Int64, n::Int64 )
	n = x.covariance ? m : n; 
	max_short, max_long = x.max_short,x.max_long
	# {p,β,R}  are fake values to be overwritten in  price update: optimize!(...)
	p = ones(1,m); β₁ₓₘ = ones(1,m); Rₙₓₘ = ones(n,m); 
	####
	#--- SPARSE REPRESENTATION: based on socp4.jl---
	k = 5m;
	
	eₖₓ₁,e₁ₓₖ,e₁ₓₘ,e₁ₓ₁ = ones(k,1), ones(1,k), ones(1,m), [1.0]'
	O₁ₓₘ,O₁ₓₖ,Oₖₓ₁,Oₖₓ₁,Oₖₓₘ  = spzeros(m)', spzeros(1,k), spzeros(k,1), spzeros(k,1), spzeros(k,m) 
	Oₖₓₖ,Oₙₓ₁,Oₙₓₖ, Oₘₓₘ,Oₘₓₖ = spzeros(k,k), spzeros(n,1), spzeros(n,k), spzeros(m,m), spzeros(m,k)
	O₁ₓ₁,O₁ₓₘ,Oₘₓ₁,Oₖₓₘ = [0.0]', spzeros(1,m),spzeros(m,1), spzeros(k,m); 
	#  x = short + long
	xₘₓₘ  = spdiagm(ones(m))  					# plain diagonal '1'-s 
	xₘₓₖ = [100xₘₓₘ 200xₘₓₘ  400xₘₓₘ  800xₘₓₘ 1600xₘₓₘ]
	lₖₓₖ = sₖₓₖ = spdiagm(ones(k))

	α₁ₓₖ  = [1e-5e₁ₓₘ 1e-6e₁ₓₘ 1e-7e₁ₓₘ 1e-8e₁ₓₘ 1e-9e₁ₓₘ] 
	b₁ₓₖ  = [100p 200p 400p 800p 1600p] 

	#-------
    #       t₁ₓ₁   x₁ₓₘ    m x (t,s) ∈ SOC²     
	c    = [O₁ₓ₁  -β₁ₓₘ    α₁ₓₖ   α₁ₓₖ ] #   
	x.A  = [e₁ₓ₁   O₁ₓₘ	   O₁ₓₖ   O₁ₓₖ   #  γ  
		    Oₙₓ₁   Rₙₓₘ    Oₙₓₖ   Oₙₓₖ   #  residuals ∈ SOCᵐ⁺¹     
		    O₁ₓ₁   O₁ₓₘ    b₁ₓₖ   b₁ₓₖ   #  budget
			Oₘₓ₁   xₘₓₘ    xₘₓₖ  -xₘₓₖ   #  x = short + long
			Oₖₓ₁   Oₖₓₘ	   sₖₓₖ   Oₖₓₖ   #      short + _____  
			Oₖₓ₁   Oₖₓₘ	   Oₖₓₖ   lₖₓₖ   #      _____ +  long  
			O₁ₓ₁   O₁ₓₘ    b₁ₓₖ   O₁ₓₖ   #  sum(short) ≦  max_positions ÷ 2 
			O₁ₓ₁   O₁ₓₘ    O₁ₓₖ   b₁ₓₖ ] #  sum(long)  ≦ max_positions ÷ 2
	#     trajectory residuals         
	x.b  =  vec([ x.γ;   Oₙₓ₁;      x.budget;      Oₘₓ₁;   eₖₓ₁;     eₖₓ₁;  max_short; max_long ])
    #             γ - risk,       budget, x = s + l,   short,    long,
	x.K₁ = cone([ (:SOC,n+1),  (:NonNeg,1), (:Zero,m), (:NonNeg,k), (:NonNeg,k), (:NonPos,1), (:NonNeg,1) ])	# constraint cones 
	#            t            x          t,s           t,s
	x.K₂ = cone([ (:Zero,1), (:Free,m), (:NonNeg,k), (:NonNeg,k) ])   			# cones of variables 
	x.c  = vec(c)
	#----------------
end



function show(io::IO, x::P002)
	println(io,"--------------------------------------")
end

function optimize!(t::dtime, st::P002{T}, β::Vector{T}, F::Matrix{T}, price::Vector{T} ) where {T}
	m,n = size(F)
	A,b,c,K₁,K₂,k,p,model, mp = st.A, st.b, st.c, st.K₁, st.K₂,5m, price, st.model, MathProgBase.SolverInterface
	try
		c[2:m+1] = -β; 
		A[2:n+1,2:m+1]= F'    # trajectory, risk 
		A[n+2,m+2:end] = [100p; 200p; 400p; 800p; 1600p; 100p; 200p; 400p; 800p; 1600p]

		mp.loadproblem!(model, c, A, b, K₁,K₂ )
		mp.optimize!(model)
		X = mp.getsolution(model)
		x = X[2:m+1]
		return x,mp.status(model)
	catch ex
		@warn ex
		return zeros(length(price)),:Error
	end
end


