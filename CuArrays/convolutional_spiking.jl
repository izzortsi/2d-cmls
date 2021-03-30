##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##

function frames(A, scheme; steps=8)

    frames_list = [A]

    for i in 1:steps
        outs = similar(A)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) spiking_kernel(n, F, r0, frames_list[end], scheme, outs_)
        push!(frames_list, outs_)
    end

    return frames_list
end
##

function make_gif(clist; path::String="~", fps=2)
    steps = length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c=cgrad([:black, :white]), xaxis=true, yaxis=true, clims=(0., 1.))
        title!("frame $i")

    end every 1
    ct = "heatmap"
    gif(anim, "$(path)/$(Dates.Time(Dates.now())).gif", fps=fps)
end
##


##
dev = CuDevice(0)
max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads) / 2 |> Int

numblocks = ceil(Int, n / max_threads_per_dim)
threads = n ÷ numblocks

##

function frames(state; kernel=[1. 1 1; 1 0 1; 1 1 1] |> CuArray, bin=0.9, r=1.3, k=0.0)


    step = steps
    list = []
    for i = 1:step


        S = (state >= bin) .* state # the spiking neurons
        nS = (state < bin) .* state # the complimentary matrix

        conv = convolve2(nS + r * S, kernel, UInt32(0), UInt32(0)) / sum(kernel) # the spiking neuron have a 1.3fold greater influence over its neighbors


        state = e * (nS + k * S) + (1 - e) * conv # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron


        push!(list, state)
    end
    return list
end
S = CUDA.rand(100, 100)
##
A .>= 0.5
##

bin = 0.93
e = 0.66

r = 1.3
k = 0.0

b = 1.01
a = 0.909

outs = CUDA.zeros(n, n)
scheme = cu([1. 1 1; 1 0 1; 1 1 1])

##
outs_list, ewdistances, imgdistances = frames(n, F, r0, A, scheme, outs, steps=120, distances=true, frobenius=true)
# outs_list = frames(n, F, r0, A, scheme, outs, steps=150)
host_outs = Array.(outs_list)
##
# plot(sum.(ewdistances) ./n^2)
# plot(imgdistances)
##
length(host_outs)
##
opath = pwd()
opath
##
make_gif(host_outs, fps=8, path=opath)
