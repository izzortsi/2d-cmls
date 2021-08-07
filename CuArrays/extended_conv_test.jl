##
using CUDA
using LinearAlgebra
# using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
using GLMakie
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
# kernel = OffsetArray(kernel, 1:n, 1:m)
# kernel = convert(Array{Float64}, kernel)


# #%%
# #heatmap(kernel)

# #%%
# kernel = cu(kernel)
#%%
#%%
#dkernel = [x -> (x <= 0.5 ? x*2 : x*kernel[i]) for i in CartesianIndices(kernel)]
#dkernel = [dynamic_cufunction(x -> (x <= 0.5 ? x*2 : x*k)) for k in kernel]
#%%
#kfun(x, μ, k) = (x <= 0.5 ? x*k*3/μ : x*k)
#%%
#f = dynamic_cufunction(kfun)
#%%
#CUDA.CuDeviceMatrix{CUDA.DeviceKernel{var"#23#25"{Float64}, Tuple{}}}((5,5), Ref(dkernel))
#%%
#(x, μ, k) -> (x <= 0.5 ? x*k*3/μ : x*k)
#%%


hkernel = Kernel.gaussian((1.2, 1.2), (n, m))
hkernel = OffsetArray(hkernel, 1:n, 1:m)
hkernel = convert(Array{Float64}, hkernel)
kernel = cu(hkernel)
#%%
#dk = [(x, μ) -> (x <= 0.5 ? x*μ : x*k) for k in hkernel]
#dk = [(x, μ) -> x*(1/2*π*5)*exp((x - μ )^2*(x-μ)/5) <= 0.5 ? (1/2*π*5)*exp((x - μ )^2*(x-μ)/5) : x*k for k in hkernel]
#%%
#dk[2,2](0.6, 1.2)

#%%


#kfuns = cu([(x, μ) -> x*(1/2*π*2.5)*exp((x - μ )^(x-μ)/2.5^2) <= 0.8 ? (1/2*π*2.5)*exp((x - μ )^(x-μ)/2.5^2) : x*k for k in hkernel])
#%%
#kfuns = cu([(x, μ) -> x*(1/2*π*2.5)*exp((x - μ )^(x-μ)/2.5^2) >=1/2 ? k *x*(1 - x) : (x - 1)* 2*μ  for k in hkernel])
kfuns = cu([(x, μ) -> x >= 1 ? x-1 : x+cos(1-x)*μ  for k in hkernel])

#%%
#%%

using LazyGrids
x, y = ndgrid(-2π:0.01:2π, -2π:0.01:2π)
#%%
xy= cat(x, y, dims=3)

#%%
dgrid =(@. sqrt(xy[:, :, 1] ^ 2 + xy[:, :, 2]^2))
    
#%%

g(x, μ, σ) = (1/2*π*σ)*exp((x - μ )^2/σ^2)

#%%

g.(dgrid, 1.0, 1.0)
#%%

heatmap(g.(dgrid, 4.0, 5.0)/1e26)

#%%
#%%


x= range(-2π, 2π, length=9)
#%%
y = x'

#%%
sgrid = (@. sqrt(x ^ 2 + y^2))
#%%
heatmap(sgrid)
#%%
F(x) = cos(x)^3 + sin(x)

#%%

heatmap(F.(sgrid))
#%%
kernel = F.(sgrid)
kernel ./ sum(kernel)
#%%

344


flist, params = frames(istate, 160, econv, ckern= kernel, kfuns = kfuns)

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
# make_gif(
#     host_outs,
#     fps = 8,
#     path = opath,
#     filename = filename,
# )


#points = Node(Matrix{Float32})
#%%
frame = Node(host_outs[1])
#%%


fig, hm = heatmap(frame)
#limits!(ax, 0, 30, 0, 30)

num_frames = 1:niter
fig
#%%
#frame[] = host_outs[2][:,:]
#%%
function run(fig, fps, obs, data, niter)

    for i in 1:niter
        obs[] = data[i][:,:]
        sleep(1 / fps)
    end
end

#%%
run(fig, 20, frame, host_outs, 159)
#%%

ff  
function rec_run(fig, fps, obs, data, niter)

    stream = VideoStream(fig, framerate=fps)
    println("Recording started.")

    for i in 1:niter
        obs[] = data[i][:,:]
        recordframe!(stream)
        sleep(1 / fps)
    end

    save(opath, stream)
end

#%%
#rec_run(fig, 20, frame, host_outs, niter)
