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
        img = imrotate(img, π/2)
        n, m = size(img)
        img = OffsetArray(img, 1:n, 1:m)
        img = convert(Array{Float64}, img)
        return img |> AFArray
    end
end

function make_frames(config::AFArray; kernel=[1. 1 1; 1 0 1; 1 1 1] |> AFArray, nbhood=[1. 1 1; 1 0 1; 1 1 1] |> AFArray)


    step = steps
    list = [config]
    for i = 1:step

        nbs = convolve2(config, nbhood, UInt32(0), UInt32(0))/sum(nbhood)
        mnbs = maximum(nbs)
        conv = convolve2(config, kernel, UInt32(0), UInt32(0))
        mconv = maximum(conv)
        #println("maxnbs: $(mnbs)")
        #println("maxconv: $(mconv)")

        A = (config < 0.5*r) .* config
        B = (config >= 0.5*r) .* config
        A *= r
        B = r*(1-B)

        #A = (config < a*mconv) .* config
        #B = (config >= a*mconv) .* config
        #A *= r
        #B = r*(1-B)
        config = A + B
        push!(list, config)
    end
    return list
end


#program parameters

out_path="~/Dropbox/Julia/2DCMLs/"
initialization="img" #or "img"

e=0.7
b=0.11111
r=1.4173
a=1/r
dim=200
steps=200


#kernel = [b*a b b*a; b 0 b; b*a b b*a] |> AFArray
#kernel = [b*a b b*a; b e*b b; b*a b b*a] |> AFArray

nbhood = [1. 1 1; 1 0 1; 1 1 1] |> AFArray
kernel = [b*a b b*a; b e*b b; b*a b b*a] |> AFArray


iconfig=make_initial_config(initialization="img")
dim, =size(iconfig)
aflist = make_frames(iconfig, kernel=kernel, nbhood=nbhood)
list = Array.(aflist)

finalize(aflist)

if initialization=="img"
    #scene = heatmap(iconfig, show_axis=false, colorrange=(0,1), colormap=:grays, resolution=size(iconfig))

    record(scene, "output$(time()).mkv", range(1, stop=steps), framerate=10, sleep=false) do i
        heatmap!(list[i], show_axis=false, colorrange=(0,1), colormap=:grays)
    end
else
    #scene = heatmap(iconfig, show_axis=false, colorrange=(0,1), colormap=ColorSchemes.sunset, resolution=size(iconfig))

    record(scene, "output$(time()).mkv", range(1, stop=steps), framerate=12, sleep=false) do i
        heatmap!(list[i], show_axis=false, colorrange=(0,1), colormap=ColorSchemes.sunset)
    end
end
