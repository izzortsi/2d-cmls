using ArrayFire
using LinearAlgebra
using AbstractPlotting
using GLMakie
using Images, TestImages, Colors
using ColorSchemes
using Dates
using OffsetArrays

function make_initial_config(;initialization="rand")
    if initialization=="rand"
        return rand(dim, dim) |> AFArray
    else
        img = testimage("cam")
        img = Gray.(img)
        img = imrotate(imresize(img, ratio=2/3), π)
        n, m = size(img)
        img = OffsetArray(img, 1:n, 1:m)
        img = convert(Array{Float64}, img)
        return img |> AFArray
    end
end

function make_frames(config; kernel=[1. 1 1; 1 0 1; 1 1 1] |> AFArray, nbhood=[1. 1 1; 1 0 1; 1 1 1] |> AFArray)

    step = steps
    list = [config]
    for i = 1:step

        S = (config >= bin) .* config #the spiking neurons
        nS = (config < bin) .* config #the complimentary matrix

        conv = convolve2(nS + r*S, kernel, UInt32(0), UInt32(0))/sum(nbhood) #the spiking neuron have a r-fold greater influence over its neighbors

        config = e*(nS + k*S) + (1-e)*conv #(nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron

        push!(list, config)

    end
    return list
end


#program parameters {

out_path = "~/Dropbox/Julia/2DCMLs/"
initialization="rand" # "img" or "rand"

bin=0.93
e=0.66
b=1.01

r=1.3
k=0.0
a=0.909
dim=250
steps=700

# }

iconfig = make_initial_config(initialization="rand")
dim, = size(iconfig)

##alternative kernels
#kernel = [b*a b b*a; b 0 b; b*a b b*a] |> AFArray

nbhood = [1. 1 1; 1 0 1; 1 1 1] |> AFArray
kernel = [b*a b b*a; b e*b b; b*a b b*a] |> AFArray

aflist = make_frames(iconfig, kernel=kernel, nbhood=nbhood)

list = Array.(aflist)

finalize(aflist)

if initialization=="img"
    scene = heatmap(iconfig, show_axis=false, colorrange=(0,1), colormap=:grays, resolution=size(iconfig))

    record(scene, "output$(time()).mkv", range(1, stop=steps), framerate=10, sleep=false) do i
        heatmap!(list[i], show_axis=false, colorrange=(0,1), colormap=:grays)
    end
else
    scene = heatmap(iconfig, show_axis=false, colorrange=(0,1), colormap=ColorSchemes.sunset, resolution=size(iconfig))

    record(scene, "output$(time()).mkv", range(1, stop=steps), framerate=12, sleep=false) do i
        heatmap!(list[i], show_axis=false, colorrange=(0,1), colormap=ColorSchemes.sunset)
    end
end
