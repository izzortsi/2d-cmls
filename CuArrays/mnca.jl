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

mutable struct MNCA
    space::Matrix
    kernels::Array
    transition_functions::Array
    states::Array
end


include("convolution.jl")
using .Convolution
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##
function frames(configuration, niter, kernels, functions, states; params)
    #params = Dict{String,Any}(["bin" => bin, "e" => e, "r" => r, "k" => k])
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
const n = 256
##
setup_convolution(n)
##
niter = 300
##
Z2_region = [[x, y] for x in -128:1:127, y in 128:-1:-127]
#%%
transpose(Z2_region)

#%%
distances_field = norm.(Z2_region)

##
heatmap(distances_field)
#%%
F(X) = sin(X[1])^3 + cos(X[2] + π/2)

f(x) = sin(x)^3 + cos(x + π/2)
#%%
trig_field = F.(Z2_region)
#%%
trig_field = f.(distances_field)

#%%
gaussian_kernel = Kernel.gaussian(3)
#%%


#%%
gkernel = no_offset(gaussian_kernel)

#%%
#heatmap(gkernel)


#%%
#heatmap(trig_field)
#%%
kdim, = size(gkernel)
convolved = CUDA.zeros(n, n)
#%%
device_field = cu(trig_field)
device_kernel = cu(gkernel)

#%%



conv_field_gkern = conv(n, device_field, device_kernel, convolved, kdim)
#%%
conv(n, device_field, device_kernel, convolved, kdim)
#%%
host_conv = conv_field_gkern |> Array


#%%
conv
#%%



init_state = CUDA.rand(n, n)
##
@elapsed flist, params = frames(init_state, niter; ckern=ckern, r=r)
#%%
@elapsed host_outs = Array.(flist)
##

push!(params, "a" => a)
push!(params, "b" => b)
push!(params, "kerpattern" => string(ckern_expr))

#%%
field = Node(host_outs[1])
fig, hm = GLMakie.heatmap(field)
#%%
makie_record(fig, field, host_outs, params, niter, "spiking")