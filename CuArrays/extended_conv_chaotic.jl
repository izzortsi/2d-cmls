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
function frames(state, niter; ckern=cu([1. 1 1; 1 0 1; 1 1 1]), bin = 0.5, r = 1.4173, with_conv = true)
    params = Dict{String,Any}(["bin" => bin, "r" => r])
    state_seq = [state]
    for i = 1:niter
        kdim, = size(ckern)
        state = state_seq[end]
        A = (state .< bin * r) .* state # the spiking neurons
        B = (state .>= bin * r) .* state # the complimentary matrix
        A *= r
        B = r * (1 .- B)
        if with_conv
            convolved = CUDA.zeros(N, N)
            econv(N, A+B, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
            state = (A+B) * e + (1-e) * convolved
        else
            state = A + B
        end
        push!(state_seq, deepcopy(state))
    end
    return state_seq, params
end

function init_state()
    img = testimage("cam")
    img = Gray.(img)
    img = imrotate(imresize(img, ratio = 2 / 3), π)
    n, m = size(img)
    img = OffsetArray(img, 1:n, 1:m)
    img = convert(Array{Float64}, img)
    A = cu(img)
    return A
end
##
#const n = 250
##
bin = 0.5
r = 1.3
niter = 200

#%%
b = 1.01
e = 0.66
a = 0.909
ρ = 1.5

ckern_expr = :([b * a b b * a; b e * b b; b * a b b * a])
ckern = cu(eval(ckern_expr))
ckern ./= (sum(ckern) / ρ)
#%%


istate
##
istate = init_state()
#%%
#heatmap(istate)
#%%


N, = size(istate)

#%%
econv = setup_convolution(N, is_extended = true)

#%%



flist, params = frames(istate, niter; ckern)

host_outs = Array.(flist)
#%%
heatmap(host_outs[1])

#%%


##
opath = pwd() * "/CuArrays/outputs/chaotic/"
mkpath(opath)
##
filename = replace("$(Dates.Time(Dates.now()))", ":" => "_")
open(opath * filename * ".txt", "w") do io
    for (key, val) in params
        println(io, "$key: $val")
    end
end
##
make_gif(
    host_outs,
    fps = 8,
    path = opath,
    filename = filename,
)
