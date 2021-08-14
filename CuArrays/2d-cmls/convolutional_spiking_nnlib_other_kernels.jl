##
using CUDA
using LinearAlgebra
using Dates
using Images, TestImages, Colors
using OffsetArrays
using GLMakie
using NNlibCUDA
using LazyGrids
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##
function frames(
    state, niter; 
    ckern=cu([1. 1 1; 1 0 1; 1 1 1]), 
    bin=Float32(0.9), 
    e=Float32(0.75), 
    r=Float32(1.4), 
    k=Float32(0.0)
        )
    kdim, = size(ckern)
    state_seq = [state]
    for i = 1:niter
        state = state_seq[end]
        S = (state .>= bin) .* state # the spiking neurons
        nS = (state .< bin) .* state # the complimentary matrix
        spike = nS .+ (r .* S)
        #println(typeof(spike))
        convolved = NNlibCUDA.conv(spike, ckern, pad = kdim ÷ 2) ./ Float32(63^1.7 -1)
        #conv(n, spike, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
        state = e .* (nS .+ (k .* S)) .+ (Float32(1) - e) .* convolved  # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        #e*(nS + k*S) + (1-e)*conv
        # state = e * (( r * nS) + (k * S)) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        push!(state_seq, deepcopy(state))
    end
    return state_seq
end
##
const SIZE = 256
const MID = SIZE ÷ 2
const O = (SIZE - MID) ÷ 4 -0.5
Y, X = ndgrid(-O:(O-1), -O:(O-1))
##


# bin=0.93
# e=0.66
# b=1.01

# r=1.3
# k=0.0
# a=0.909

D(X, Y, R) = sqrt.((X/R) .^2 + (Y/R) .^2)
bell(x, m, s) = exp(-((x-m)/s)^2 / 2)
growth(U, m, s) = bell(U, m, s)*2-1
F(x) = cos(4*SIZE*x)^3 + sin((4*SIZE ÷2)*x + π/2)
#%%

DGRID = D(X, Y, SIZE/1.3)
#%%



K1 = (DGRID .< 1) .* bell.(DGRID, 0.5, 0.15)
K2 = clamp.((DGRID .< 1) .* F.(bell.(DGRID, 0.5, 0.15)), 0, 1)
#%%
fig = Figure()
ax, hm = heatmap(fig[1,1], K2, colorrange=(0, 1))
Colorbar(fig[1,2], hm)
#%%
fig
#%%
K2 = cu(reshape(K2[:], (63, 63, 1, 1)) )
#%%


sum(K2)
#%%


niter = 200
##

##

#ckern_expr = :([b * a b b * a; b e * b b; b * a b b * a])
#ckern = eval(ckern_expr)


#%%
#ckern = cu(reshape(ckern[:], (3, 3, 1, 1)))
#%%
#ckern ./= (sum(ckern) / ρ)
##
init_state = CUDA.rand(SIZE, SIZE, 1, 1)
#%%
#rkern = CUDA.rand( 5, 5, 1, 1)
# #%%
# NNlibCUDA.conv(init_state, ckern, pad = 1)
# #%%
# S = (init_state .>= bin) .* init_state
# #%%
# nS = (init_state .< bin) .* init_state # th
# #%%
# spike = nS + (Float32(r) * S)

# #%%
# NNlibCUDA.conv(spike, ckern, pad = 1)
#%%
kdim, = size(K2)
#%%
kdim ÷ 2
#%%

2
NNlibCUDA.conv(init_state, K2, pad = 31)


##
@elapsed flist = frames(init_state, niter, ckern=K2)
#%%
#flist
#%%


@elapsed host_outs = Array.(flist)
##
host_outs
#%%
host_outs[1][:,:]
#%%
h_outs = [frame[:,:] for frame in host_outs]
#%%


params = Dict()
push!(params, "a" => "none")
push!(params, "b" => "none")
#push!(params, "kerpattern" => string(ckern_expr))
#%%


field = Node(h_outs[1])
fig, hm = GLMakie.heatmap(field, colorrange=(0,1))
#%%
fig
#%%

makie_record(fig, field, h_outs, params, niter, "spiking")