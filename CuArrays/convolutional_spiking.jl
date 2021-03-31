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

##
function frames(
    state, niter; 
    ckern=cu([1. 1 1; 1 0 1; 1 1 1]), 
    bin=0.93, 
    e=0.66, 
    r=1.3, 
    k=0.0
    )
    params = Dict{String,Any}(["bin" => bin, "e" => e, "r" => r, "k" => k])
    kdim, = size(ckern)
    state_seq = [state]
    for i = 1:niter
        state = state_seq[end]
        convolved = CUDA.zeros(n, n)
        S = (state .>= bin) .* state # the spiking neurons
        nS = (state .< bin) .* state # the complimentary matrix
        spike = nS + (r * S)
        conv(n, spike, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
        state = e * (nS + (k * S)) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        # state = e * (( r * nS) + (k * S)) + (1 - e) * convolved # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
        push!(state_seq, deepcopy(state))
    end
    return state_seq, params
end
##
const n = 250
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

niter = 300
##

##
# alternative kernel patterns
# ckern = [b*a b b*a; b 0 b; b*a b b*a] |> CuArray
# ckern = [b*a b b*a; b e*b b; b*a b b*a] |> CuArray
# ckern = [1. 1 1; 1 0 1; 1 1 1] |> CuArray
##
ckern_expr = :([b * a b b * a; b e * b b; b * a b b * a])
ckern = cu(eval(ckern_expr))
ckern ./= (sum(ckern) / ρ)
##
init_state = CUDA.rand(n, n)
flist, params = frames(init_state, niter; ckern=ckern, r=r)
host_outs = Array.(flist)
##
opath = pwd() * "/CuArrays/outputs/conv_spiking/"
mkpath(opath)
push!(params, "a" => a)
push!(params, "b" => b)
push!(params, "kerpattern" => string(ckern_expr))
##
filename = "$(Dates.Time(Dates.now()))"
open(opath * filename * ".txt", "w") do io  
    for (key, val) in params
        println(io, "$key: $val")
    end
end
make_gif(host_outs, fps=8, path=opath, filename=filename)
