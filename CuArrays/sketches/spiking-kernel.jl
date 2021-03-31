##
using CUDA
using LinearAlgebra
using Plots
using Dates
using Images, TestImages, Colors
using OffsetArrays
##

F(r, x) = r * x * (1 - x)


function spiking_kernel(n, e, A, cscheme, outs)

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
        cs_ij = cscheme
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        cf_upleft = cs_ij[1, 1]
        cf_downleft = cs_ij[1, 3]
        cf_upright = cs_ij[3, 1]
        cf_downright = cs_ij[3, 3]
        

        #
        bin = 0.93
        e = 0.66
        r = 1.3
        k = 0.0


        # calculating the convolution
        X = A[i,j]
        if X >= bin:
            X = X*r

        N = (sup * cf_up + sleft * cf_left + sright * cf_right + sdown * cf_down + supleft * cf_upleft + supright * cf_upright + sdownleft * cf_downleft + sdownright * cf_downright) / 4

        outs[i, j] = (1 - e) * X + e * N
        # avg = (cf_down + cf_right + cf_left + cf_up + cf_upright + cf_upleft + cf_downright + cf_downleft)/8
        # outs[i, j] = (1-avg)*X + avg*N


        S = (config >= bin) .* config # the spiking neurons
        nS = (config < bin) .* config # the complimentary matrix

        conv = convolve2(nS + r * S, kernel, UInt32(0), UInt32(0)) / sum(nbhood) # the spiking neuron have a r-fold greater influence over its neighbors

        config = e * (nS + k * S) + (1 - e) * conv # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron


    end

    return nothing
end
##
function frames(n, F, r0, A, scheme, outs; distances=false, frobenius=false, steps=8)

    frames_list = [A]
    entrywise_distances = []
    img_distances = []

    for i in 1:steps
        outs_ = similar(outs)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) spiking_kernel(n, F, r0, frames_list[end], scheme, outs_)
        push!(frames_list, outs_)

        if distances == true
            if frobenius == true
                ewD = (A - outs_).^2
                push!(entrywise_distances, ewD)
                imgD = sqrt(sum(ewD))
                push!(img_distances, imgD)
            else
                ewD = abs.(A - outs_)
                push!(entrywise_distances, ewD)

                imgD = sum(ewD) / n^2
                push!(img_distances, imgD)
            end
        end
    end

    if distances == true
        return frames_list, entrywise_distances, img_distances
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

img = testimage("cam");
img = Gray.(img);
img = imrotate(imresize(img, ratio=2 / 3), π);

##
img = OffsetArray(img, 1:344, 1:344)
##
img = convert(Array{Float64}, img);
##
A = cu(img)
##
# A_ = A/2
#
# sqrt(sum((A_ - A).^2))
#
# sum(sqrt.((A_ - A).^2))/344^2
#
# sum(abs.((A_ - A)))/344^2

##
n, = size(img)


##
# schemes = cu(cudaconvert.([CUDA.rand(3, 3) for i=1:n, j=1:n]));
# schemes = cu(cudaconvert.([CUDA.fill(1/4, 3, 3) for i=1:n, j=1:n]));
##
dev = CuDevice(0)
##
max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads) / 2 |> Int

numblocks = ceil(Int, n / max_threads_per_dim)
threads = n ÷ numblocks

##

@cuda blocks = (numblocks, numblocks) threads = (threads, threads) spiking_kernel(n, F, r0, A, scheme, outs)
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
