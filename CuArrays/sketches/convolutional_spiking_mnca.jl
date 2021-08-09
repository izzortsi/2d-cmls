##
using CUDA
using LinearAlgebra
using Dates
using Images, TestImages, Colors
using OffsetArrays
using GLMakie
using ImageFiltering
using LazyGrids
##
include("convolution.jl")
using .Convolution
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##
function frames(
    state, niter,
    kernels; 
    bin=0.93, 
    e=0.66, 
    r=1.3, 
    k=0.0
        )
    params = Dict{String,Any}(["bin" => bin, "e" => e, "r" => r, "k" => k])
    
    state_seq = [state]
    k1, k2, k3 = kernels
    kdim1, = size(k1)
    kdim2, = size(k2)
    kdim3, = size(k3)
    for i = 1:niter
        state = state_seq[end]
        convolved = CUDA.zeros(n, n)
        #S = (state .>= bin) .* state # the spiking neurons
        #nS = (state .< bin) .* state # the complimentary matrix
        #spike = nS + (r * S)
        conv(n, state, k1, convolved, kdim1) # the spiking neuron have a 1.3fold greater influence over its neighbors
        state1 = e * state + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        state1 = ((state1 .>= 1.0) .* state1)*e + (state1 .< 1.0) .* state1
        S = (state1 .>= 0.5) .* state1
        nS = (state1 .< 0.5) .* state1
        conv(n, S + nS*0.5, k2, convolved, kdim2)
        state2 = (1-e) * state1 + e * convolved
        conv(n, 0.5*S + nS, k3, convolved, kdim3)
        state3 =  (1-e) * state1 + e * convolved
        #S = (state2 .>= 0.8) .* state2
        #nS = (state2 .< 0.3) .* state2
        #conv(n, S+nS, k3, convolved, kdim3)
        state = (state2 + state3) ./ 2
        #state = (state1 .+ state2 .+ state3) ./3
        state = state .- CUDA.minimum(state)
        state = state ./ CUDA.maximum(state)
        # state = e * (( r * nS) + (k * S)) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        
        push!(state_seq, deepcopy(state))
    end
    return state_seq, params
end
##
const n = 400
##
conv = setup_convolution(n)
##
bin = 0.93
e = 0.66
r = 1.1
k = 0.0

b = 1.01
a = 0.909
ρ = 1.5

niter = 500
##

##
Z2_region = [[x, y] for x in -n ÷ 2:1: (n ÷ 2) -1 , y in n ÷ 2:-1:-(n ÷2) + 1]
#%%
transpose(Z2_region)

#%%
distances_field = norm.(Z2_region) ./ n
#%%

f(x) = cos(400x)^3 + sin(200x + π/2)
#%%
g(x; μ=1.9, σ=1.5) = (1/2π*σ^2) * exp(-(x - μ)^2 / σ^2 )
#%%
f_dfield = f.(distances_field)
g_dfield = g.(distances_field, μ=0.02, σ=0.01)
#%%
GLMakie.heatmap(f_dfield)
##
GLMakie.heatmap(g_dfield)
#%%
k1 = f_dfield[((n ÷ 2) - 13):((n ÷ 2) + 13), ((n ÷ 2) - 13):((n ÷ 2) + 13)]
k1 .-= minimum(k1)
k1 ./= maximum(k1)
GLMakie.heatmap(k1)

#%%
k2 = g_dfield[((n ÷ 2) - 15):((n ÷ 2) + 15), ((n ÷ 2) - 15):((n ÷ 2) + 15)]
k2 .-= minimum(k2)
k2 ./= maximum(k2)
GLMakie.heatmap(k2)
#%%

#%%



ckern_expr = :([b * a b b * a; b e * b b; b * a b b * a])
ckern = cu(eval(ckern_expr))
ckern ./= (sum(ckern) / ρ)
#%%
GLMakie.heatmap(ckern |> Array)

##
init_state = CUDA.rand(n, n)
##
k1 = cu(k1)
k2 = cu(k2)
#%%
flist, params = frames(init_state, niter, [ckern, k1, k2], r=r)
#%%
host_outs = Array.(flist)
##
push!(params, "a" => a)
push!(params, "b" => b)
push!(params, "kerpattern" => string(ckern_expr))
field = Node(host_outs[1])
fig, hm = GLMakie.heatmap(field)
makie_record(fig, field, host_outs, params, niter, "spiking")