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
function frames(state, niter, conv; ckern=cu([1. 1 1; 1 0 1; 1 1 1]), kfuns = nothing, bin = 0.5, r = 1.4173)
    params = Dict{String,Any}(["bin" => bin, "r" => r])
    state_seq = [state]
    for i = 1:niter
        kdim, = size(ckern)
        state = state_seq[end]
        convolved = CUDA.zeros(N, N)
        if kfuns !== nothing
            conv(N, state, ckern, kfuns, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
            push!(state_seq, convolved)
        else
            conv(N, state, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
            push!(state_seq, convolved)
        end
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
niter = 50

##
istate = init_state()
#%%
#heatmap(istate)
#%%


N, = size(istate)

#%%
econv = setup_convolution(N, is_extended = true)
#%%
conv = setup_convolution(N, is_extended = false)

#%%
using ImageFiltering
#%%
n, m = 5,5
kernel = Kernel.gaussian((1.2, 1.2), (n, m))
kernel = OffsetArray(kernel, 1:n, 1:m)
kernel = convert(Array{Float64}, kernel)


#%%
#heatmap(kernel)

#%%
kernel = cu(kernel)
#%%
#%%
#dkernel = [x -> (x <= 0.5 ? x*2 : x*kernel[i]) for i in CartesianIndices(kernel)]
#dkernel = [dynamic_cufunction(x -> (x <= 0.5 ? x*2 : x*k)) for k in kernel]
#%%
kfun(x, μ, k) = (x <= 0.5 ? x*k*3/μ : x*k)
#%%
#f = dynamic_cufunction(kfun)
#%%
#CUDA.CuDeviceMatrix{CUDA.DeviceKernel{var"#23#25"{Float64}, Tuple{}}}((5,5), Ref(dkernel))
#%%
(x, μ, k) -> (x <= 0.5 ? x*k*3/μ : x*k)
#%%


hkernel = Kernel.gaussian((1.2, 1.2), (n, m))
hkernel = OffsetArray(kernel, 1:n, 1:m)
hkernel = convert(Array{Float64}, kernel)
#%%
#dk = [(x, μ) -> (x <= 0.5 ? x*μ : x*k) for k in hkernel]
dk = [(x, μ) -> (x >= 10*k ? x*k^μ : x*k) for k in hkernel]
#%%
dk[2,2](0.6, 1.2)

#%%


kfuns = cu([(x, μ) -> (x >= 10*k ? x*k^μ : x*k) for k in hkernel])
#%%



flist, params = frames(istate, niter, econv, ckern= kernel, kfuns = kfuns)

host_outs = Array.(flist)
#%%
heatmap(host_outs[10])

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
