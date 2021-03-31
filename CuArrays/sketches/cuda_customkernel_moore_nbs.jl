using CuArrays
using CUDAnative
using CUDAdrv
using LinearAlgebra
using Plots
using Dates



function custom_kernel(n, e, A, schemes, outs)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n

        #indices for the neighbors
        j_minus = mod1(j-1, n)
        j_plus = mod1(j+1, n)
        i_minus=  mod1(i-1, n)
        i_plus = mod1(i+1, n)

        #neighbors
        sup = A[i, j_minus]
        sleft = A[i_minus, j]
        sright = A[i_plus, j]
        sdown = A[i, j_plus]
        supleft = A[i_minus, j_minus]
        supright = A[i_plus, j_minus]
        sdownleft = A[i_minus, j_plus]
        sdownright = A[i_plus, j_plus]

        #coefficients from the corresponding coupling scheme
        cs_ij = schemes[i,j]
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        cf_upleft = cs_ij[1, 1]
        cf_downleft = cs_ij[1, 3]
        cf_upright = cs_ij[3, 1]
        cf_downright = cs_ij[3, 3]

        #calculating the convolution
        X = A[i,j]
        N = (sup*cf_up + sleft*cf_left + sright*cf_right + sdown*cf_down + supleft*cf_upleft + supright*cf_upright + sdownleft*cf_downleft + sdownright*cf_downright)/4

        outs[i, j] = (1-e)*X + e*N
        #avg = (cf_down + cf_right + cf_left + cf_up + cf_upright + cf_upleft + cf_downright + cf_downleft)/8
        #outs[i, j] = (1-avg)*X + avg*N


    end

    return nothing
end

function frames(n, e, A, schemes, outs; steps=8)

    frames_list = [A]

    for i in 1:steps
        outs_ =similar(outs)
        @cuda blocks=(numblocks, numblocks) threads=(threads, threads) custom_kernel(n, e, frames_list[end], schemes, outs_)
        push!(frames_list, outs_)

    end
    return frames_list
end

function make_gif(clist; path::String="~/Dropbox/Julia/2DCMLs/", contour::Bool=true, fps=2)
    steps=length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c= ColorGradient([:black, :white]), xaxis=true, yaxis=true,clims=(0., 1.))
        title!("frame $i")

    end every 1

    gif(anim, "$(path) $(Dates.Time(Dates.now())).gif", fps=fps)
end

n=200
n2=n^2
e = 0.5

A = CuArrays.rand(n,n)
outs = CuArrays.zeros(n,n)

schemes = cu(cudaconvert.([CuArrays.rand(3, 3) for i=1:n, j=1:n]));

dev = CuDevice(0)
max_threads = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads)/2 |> Int

numblocks = ceil(Int, n/max_threads_per_dim)
threads = n÷numblocks

@cuda blocks=(numblocks, numblocks) threads=(threads, threads) custom_kernel(n, e, A, schemes, outs)

outs_list = frames(n, e, A, schemes, outs, steps=80)
host_outs = Array.(outs_list)

make_gif(host_outs, fps=15)
