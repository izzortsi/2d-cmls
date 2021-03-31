##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##
# include("convolution.jl")
# using .Convolution
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##

##
function frames(
    state, niter;
    bin=0.5,  
    r=1.4173
    )
    params = Dict{String,Any}(["bin" => bin, "r" => r])
    state_seq = [state]
    for i = 1:niter
        state = state_seq[end]
        A = (state .< bin * r) .* state # the spiking neurons
        B = (state .>= bin * r) .* state # the complimentary matrix
        A *= r
        B = r * (1 .- B)
        # convolved = CUDA.zeros(n, n)
        # conv(n, spike, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
        # A = (config < a*mconv) .* config
        # B = (config >= a*mconv) .* config
        # A *= r
        # B = r*(1-B)
        state = A + B
        push!(state_seq, deepcopy(state))
    end
    return state_seq, params
end

function init_state()
    img = testimage("cam");
    img = Gray.(img);
    img = imrotate(imresize(img, ratio=2 / 3), π);
    n, m = size(img)
    img = OffsetArray(img, 1:n, 1:m)
    img = convert(Array{Float64}, img);
    A = cu(img)
    return A
end
##
const n = 250
##
bin = 0.5
r = 1.3
niter = 200
##
istate = init_state()
flist, params = frames(istate, niter)
host_outs = Array.(flist)
##
opath = pwd() * "/CuArrays/outputs/chaotic/"
mkpath(opath)
##
filename = "$(Dates.Time(Dates.now()))"
open(opath * filename * ".txt", "w") do io  
    for (key, val) in params
        println(io, "$key: $val")
    end
end
make_gif(host_outs, fps=8, path=opath, filename=filename)
