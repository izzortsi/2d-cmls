#%%
using CUDA
using LinearAlgebra
using LazyGrids
using Random
using GLMakie
#%%
const SIZE = 1 << 9
const MID = SIZE ÷ 2
const O = SIZE - MID
Y, X = ndgrid(-O:(O-1), -O:(O-1))
# %%
include("convolution.jl")
using .Convolution
# %%

CUDA.allowscalar(false)
# %%
mutable struct Particle

    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    speed::Float64
    mass::Float64
    activation::Float64
    interaction_radius::Float64
    nb_size::Float64

    function Particle(id, )

end

struct System
    particles::Vector{Particle}
    update_functions::Vector{Function}
    parameters::Dict{Any, Any}
end

# %%
