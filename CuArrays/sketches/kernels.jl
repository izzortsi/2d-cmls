##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##

function convolution(n, A, filter, outs)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n

        # indices for the neighbors
        j_minus = mod1(j - 1, n)
        j_plus = mod1(j + 1, n)
        i_minus =  mod1(i - 1, n)
        i_plus = mod1(i + 1, n)

        # neighbors
        sup = A[i, j_minus]
        sleft = A[i_minus, j]
        sright = A[i_plus, j]
        sdown = A[i, j_plus]
        supleft = A[i_minus, j_minus]
        supright = A[i_plus, j_minus]
        sdownleft = A[i_minus, j_plus]
        sdownright = A[i_plus, j_plus]

        # coefficients from the corresponding coupling scheme
        cs_ij = filter
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        cf_upleft = cs_ij[1, 1]
        cf_downleft = cs_ij[1, 3]
        cf_upright = cs_ij[3, 1]
        cf_downright = cs_ij[3, 3]
        cf_center = cs_ij[2, 2]


        # calculating the convolution
        X = A[i,j]

        N = (
        sup * cf_up +
        sleft * cf_left + 
        sright * cf_right + 
        sdown * cf_down + 
        supleft * cf_upleft + 
        supright * cf_upright + 
        sdownleft * cf_downleft + 
        sdownright * cf_downright +
        X * cf_center
        )
        
        outs[i, j] = N
    end

    return nothing
end
##
function loop_filter(niter, A, filter)
    for i in 1:niter
        outs = CUDA.zeros(n, n)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, filter, outs)
        A = outs
    end
    return A
end

##

img = testimage("mandril");
img = Gray.(img);
img = imrotate(imresize(img, ratio=2 / 3), π);
##
img = OffsetArray(img, 1:344, 1:344)
img = convert(Array{Float64}, img);
A = cu(img)
##
n, = size(img)

##
dev = CuDevice(0)

max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads) / 2 |> Int

numblocks = ceil(Int, n / max_threads_per_dim)
threads = n ÷ numblocks
##
heatmap(A)
##
outs = CUDA.zeros(n, n)
filter = cu([1. 1 1; 1 1 1; 1 1 1] ./ 9)
##
@cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, filter, outs)
##

##
output = loop_filter(10, A, filter)

##
heatmap(A)
##
output = loop_filter(100, output, filter)
##
heatmap(output, clims=(0, 1))
##