##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##
include("convolution.jl")
using .Convolution
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##
harr = ((x -> heatmap(x, clims=(0, 1))) ∘ Array)
##
function frames(
    state, niter; 
    ckern=cu([1. 1 1; 1 0 1; 1 1 1]), 
    bin=0.93, 
    e=0.66, 
    r=1.3, 
    k=0.0
    )

    kdim, = size(ckern)
    state_seq = []
    for i = 1:niter
        convolved = CUDA.zeros(n, n)
        S = (state .>= bin) .* state # the spiking neurons
        nS = (state .< bin) .* state # the complimentary matrix

        conv(n, nS + r * S, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors

        state = e * (nS + k * S) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        push!(state_seq, state)
    end
    return state_seq
end
##
const n = 250
##

conv = setup_convolution(n)
##
# conv(n, nS + r * S, ckern, outs, kdim)
##
bin = 0.93
e = 0.66
r = 1.3
k = 0.0

b = 1.01
a = 0.909

niter = 30

# alternative kernels
# ckern = [b*a b b*a; b 0 b; b*a b b*a] |> CuArray
# ckern = [b*a b b*a; b e*b b; b*a b b*a] |> CuArray
# ckern = [1. 1 1; 1 0 1; 1 1 1] |> CuArray
ckern = cu([b * a b b * a; b e * b b; b * a b b * a])
ckern ./= sum(ckern)
##
kdim, = size(ckern)
##
state_seq = []
init_state = CUDA.rand(n, n)
##
convolved = CUDA.zeros(n, n)
##
S = (init_state .>= bin) .* init_state # the spiking neurons
nS = (init_state .< bin) .* init_state # the complimentary matrix
##
harr(S)
##
harr(nS)
##
harr(nS + (r * S))
##
conv(n, nS + r * S, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
harr(convolved)
##
init_state = e * (nS + k * S) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
push!(state_seq, init_state)
harr(init_state)
##
state_seq[end] == state_seq[end - 1]
##
flist = frames(init_state, niter; ckern=ckern)

##
host_outs
##
host_outs = Array.(flist)

length(host_outs)
##
opath = pwd()
opath
##
Aux.make_gif(host_outs, fps=8, path=opath)
