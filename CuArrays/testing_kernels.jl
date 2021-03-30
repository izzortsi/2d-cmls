##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##
include("aux_funs.jl")
using .Aux
##
CUDA.allowscalar(false)
##
function convolution(n, A, ckern, outs, kdim)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n
        ran = kdim ÷ 2
        # indices for the neighbors
        j_minus = mod1(j - ran, n)
        j_plus = mod1(j + ran, n)
        i_minus =  mod1(i - ran, n)
        i_plus = mod1(i + ran, n)

        for s in j_minus:j_plus, t in i_minus:i_plus
            s1 = s - j + (ran + 1)
            t1 = t - i + (ran + 1)
            outs[i, j] += (A[s, t] * ckern[s1, t1])
            # @cuprintln(s1, " ", t1)
        end

    end

    return nothing
end

"""
n is the size of the square n by n matrix that will be worked on
"""
function setup_kernel(n::Int64)
    dev = CuDevice(0)
    
    max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_threads_per_dim = sqrt(max_threads) / 2 |> Int
    
    numblocks = ceil(Int, n / max_threads_per_dim)
    threads = n ÷ numblocks
    
    return numblocks, threads
end

function setup_convolution(n::Int64)
    numblocks, threads = setup_kernel(n)
    conv(n, A, filter, outs, kdim) = @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, filter, outs, kdim)
    return conv
end

function loop_conv(niter, A, ckern, kdim)
    seq = [A]
    for i in 1:niter
        outs = CUDA.zeros(n, n)
        # @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, ckern, outs, kdim)
        # conv(n, A, ckern, outs, kdim)
        conv(n, seq[end], ckern, outs, kdim)
        push!(seq, outs)
        # A = outs
    end
    return seq
end

img = testimage("mandril");
img = Gray.(img);
# img = imrotate(imresize(img, ratio=2 / 3), π);
img = imresize(img, ratio=2 / 3)
##
img = OffsetArray(img, 1:342, 1:342)
img = convert(Array{Float64}, img);
A = cu(img)
##
n, = size(img)

##
##
# heatmap(A)
##
outs = CUDA.zeros(n, n)
##
# ckern = cu([0.5 0.5 0.5 0.5 0.5;
#             0.5 1. 1 1 0.5; 
#             1 1 1 1 1; 
#             0.5 1 1 1 0.5; 
#             0.5 0.5 0.5 0.5 0.5] ./ 18)
##
ckern = cu([1 1 1; 1 1 1; 1 1 1]) ./ 9
kdim, = size(ckern)

##
conv = setup_convolution(n)
##
# output = loop_conv(10, A, ckern, kdim)

##
# hostouts = Array(output)
##
# heatmap(hostouts, clims=(0, 1))
##
# conv(n, A, ckern, outs, 5)
##
output = loop_conv(100, A, ckern, kdim)
##

hostouts = Array.(output)

##

##
make_gif(hostouts, fps=10)
##
