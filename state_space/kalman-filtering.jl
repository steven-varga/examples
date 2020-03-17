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
abstract type kf{T} end
import Base: filter, filter!,resize!,similar,copy,size
export filter,filter!,predict,update!,KF,LKF,UKF,DKF,DKF_arma, DualKF,EKF,similar,arma_dual!,copy,resize!

include("lstm-ukf.jl")
include("lkf.jl")
include("ukf-lh.jl")
include("ss.jl")
include("arma.jl")

mutable struct DKF{T} <: kf{T}
	a::kf{T}
	w::kf{T}
end

function DKF_arma( a::KF{T}, w::KF{T} ) where {T}
	a.x, w.x = w.H.nzval, a.H.nzval
	w.F = spdiagm( ones(a.x) )
	DKF{T}(a,w)
end

function update!(this::DKF, z::Vector{T} ) where {T}
	@attach(this, a,w)
	update!(a, z) # estimate the state with a given input
	update!(w, z) # then the model parameters
end




###
## single iteration 
function filter!(model::kf{T}, u::Vector{T}, z::Vector{T}) where {T}
	ŷ = predict(model,u)[1] 
	update!(model, z )
	ŷ
end
filter!(model::kf{T}, u::Vector{T}, z::T ) where {T} = filter!(model, u,[z])

function filter!(model::kf{T}, u::Matrix{T}, z::Vector{T}) where {T}
	m,n = size(u)
	ŷ = zeros(z)
	for i = 1:n
		ŷ[i] = predict(model,vec(u[:,i]) )
		update!(model, [z[i]] ) 
	end
	return ŷ
end
function filter!(model::kf{T}, u::Matrix{T}, z::Matrix{T}) where {T}
	m,n = size(u)
	ŷ = zeros(z)
	for i = 1:n
		ŷ[:,i],q = predict(model,vec(u[:,i]) )
		update!(model,vec(z[:,i]) )
	end
	return ŷ
end

function smooth!(model::kf{T}, u::Vector{T}, z::Vector{T}; degree::Int64 = 200) where {T}
	copy(model)
end

