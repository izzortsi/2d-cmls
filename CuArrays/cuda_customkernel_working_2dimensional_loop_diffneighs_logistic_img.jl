using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors



function custom_kernel(n, r, A, schemes, outs)


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

        # coefficients from the corresponding coupling scheme
        cs_ij = schemes[i,j]
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        # @cuprintln(cf_up)

        X = A[i,j]
        N = (sup * cf_up + sleft * cf_left + sright * cf_right + sdown * cf_down) / 4
        outs[i, j] = r * X * (1 - X)

        schemes[i,j][2, 1] = r * cf_up * (1 - cf_up)
        schemes[i,j][1, 2] = r * cf_left * (1 - cf_left)
        schemes[i,j][3, 2] = r * cf_right * (1 - cf_right)
        schemes[i,j][2, 3] = r * cf_down * (1 - cf_down)

    end

    return nothing
end

function frames(n, r0, A, schemes, outs; steps=8)

    frames_list = [A]

    for i in 1:steps
        outs_ = similar(outs)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) custom_kernel(n, frames_list[end], schemes, outs_)
        push!(frames_list, outs_)
        # push!(frames_list, A)
    end
    return frames_list
end

function make_gif(clist; path::String="~", fps=2)
    steps = length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c=cgrad([:black, :white]), xaxis=true, yaxis=true, clims=(0., 1.))
        title!("frame $i")

    end every 1
    ct = "heatmap"
    gif(anim, "$(path)/$(Dates.Time(Dates.now())).gif", fps=fps)
end

img = testimage("cam");
img = Gray.(img);
img = imrotate(imresize(img, ratio=2 / 3), π);
img = convert(Array{Float64}, img);
A = cu(img)
A_ = A / 2


sum(sqrt.((A_ - A).^2)) / 344^2

n, = size(img)

outs = CuArrays.zeros(n, n)

schemes = cu(cudaconvert.([CuArrays.rand(3, 3) for i = 1:n, j = 1:n]));

dev = CuDevice(0)
max_threads = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads) / 2 |> Int

numblocks = ceil(Int, n / max_threads_per_dim)
threads = n ÷ numblocks

@cuda blocks = (numblocks, numblocks) threads = (threads, threads) custom_kernel(n, r, A, schemes, outs)

# host_outs = Array(outs)
# plot(host_outs)
outs_list = frames(n, r0, A, schemes, outs, steps=150)
host_outs = Array.(outs_list)

opath = pwd()

##
make_gif(host_outs, fps=8, path=opath)
