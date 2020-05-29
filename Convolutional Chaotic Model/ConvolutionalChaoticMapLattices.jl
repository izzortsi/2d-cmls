using ArrayFire
using LinearAlgebra
using Plots
using Images, TestImages, Colors
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

function make_gif(clist; path::String="~/Dropbox/Julia/2DCMLs/", contour::Bool=true, fps=2)

    anim = @animate for i = 1:steps

        heatmap(clist[i], xaxis=true, yaxis=true,clims=(0., 1.))
        title!("$(fps) fps, iter $(i)/$(steps); params: e=$(e), b=$(b), a=$(Float16(a))")
    end every 1
    ct="heatmap"
    gif(anim, "$(path)$(fps)fps r$(r) e$(e) a$(a) $(ct) dim$(dim) kernel $(kernel) $(Dates.Time(Dates.now())).gif", fps=fps)
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

make_gif(list, fps=10)
