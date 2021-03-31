using CuArrays
using CUDAnative
using CUDAdrv
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors

F(r, x) = r * x * (1 - x)


function custom_kernel(n, F, r, A, scheme, outs)


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

        # coefficients from the coupling scheme
        cs_ij = scheme
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        # @cuprintln(cf_up)

        X = A[i,j]
        N = (sup * cf_up + sleft * cf_left + sright * cf_right + sdown * cf_down) / 4
        out = F(r, X)
        # outs[i, j] = 0.93*(r*X*(1-X)) + 0.07N
        outs[i, j] = out

    end

    return nothing
end

function frames(n, F, r0, A, scheme, outs; distances=false, steps=8)

    frames_list = [A]
    entrywise_distances = []
    img_distances = []

    for i in 1:steps
        outs_ = similar(outs)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) custom_kernel(n, F, r0, frames_list[end], scheme, outs_)
        push!(frames_list, outs_)

        if distances == true
            ewD = sqrt.((A - outs_).^2)
            push!(entrywise_distances, ewD)

            imgD = sum(ewD) / n^2
            push!(img_distances, imgD)
        end
    end

    if distances == true
        return frames_list, entrywise_distances, img_distances
    end

    return frames_list
end

function grid_search1d(n, A, scheme, range, outs, frames_per_step)

    frames_list = []
    params_list = []

    for i in range
        push!(params_list, i)
        frames_device = frames(n, F, i, A, scheme, outs; distances=false, steps=frames_per_step)
        push!(frames_list, Array.(frames_device))
        make_gif(frames_list[end], path=pwd() * "/gs$i", fps=4)
    end
    return frames_list, params_list
end


function make_gif(clist; path::String="~/Dropbox/Julia/2DCMLs/", contour::Bool=true, fps=2)
    steps = length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c=ColorGradient([:black, :white]), xaxis=true, yaxis=true, clims=(0., 1.))
        title!("frame $i")

    end every 1
    ct = "heatmap"
    gif(anim, "$(path) $(Dates.Time(Dates.now())).gif", fps=fps)
end

function make_gif_fast(clist; path::String="~/Dropbox/Julia/2DCMLs/", contour::Bool=true, fps=2)
    steps = length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c=ColorGradient([:black, :white]), xaxis=true, yaxis=true, clims=(0., 1.))
        title!("frame $i")

    end every 1
    ct = "heatmap"
    gif(anim, "$(path) $(Dates.Time(Dates.now())).gif", fps=fps)
end



img = testimage("cam");
img = Gray.(img);
img = imrotate(imresize(img, ratio=2 / 3), π);
img = convert(Array{Float64}, img);
A = cu(img)
# A_ = A/2
#
#
# sum(sqrt.((A_ - A).^2))/344^2

n, = size(img)
r0 = 3.87


outs = CuArrays.zeros(n, n)

scheme = CuArrays.rand(3, 3)

dev = CuDevice(0)
max_threads = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads) / 2 |> Int

numblocks = ceil(Int, n / max_threads_per_dim)
threads = n ÷ numblocks

@cuda blocks = (numblocks, numblocks) threads = (threads, threads) custom_kernel(n, F, r0, A, scheme, outs)

# host_outs = Array(outs)
# plot(host_outs)

# r0 = 4.01
r0 = 3.8795
outs_list, ewdistances, imgdistances = frames(n, F, r0, A, scheme, outs, steps=250, distances=true)
# outs_list = frames(n, F, r0, A, scheme, outs, steps=150)
host_outs = Array.(outs_list)

# plot(sum.(ewdistances) ./n^2)
plot(imgdistances)

##
opath = pwd()
make_gif(host_outs, fps=8, path=opath)