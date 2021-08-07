module Aux
using Plots, LinearAlgebra, CUDA, Dates, GLMakie

export make_gif, harr, makie_record

function vn_neighborhood()
    return
end

function moore_neighborhood()
    return
end

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

function make_gif(clist; path::String="~", filename="$(Dates.Time(Dates.now()))", fps=2)
    steps = length(clist)
    anim = @animate for i = 1:steps

        heatmap(clist[i], c=cgrad([:black, :white]), xaxis=true, yaxis=true, clims=(0., 1.))
        title!("frame $i")

    end every 1
    ct = "heatmap"
    gif(anim, "$(path)/$(filename).gif", fps=fps)
end

harr = ((x -> heatmap(x, clims=(0, 1))) ∘ Array)

end


function set_path_and_params(params, chaotic_or_spiking)
    if chaotic_or_spiking == "spiking"
        endpoint_dir = "conv_spiking/"
    elseif chaotic_or_spiking == "chaotic"
        endpoint_dir = "chaotic/"
    end
    opath = pwd() * "/CuArrays/outputs/" * endpoint_dir
    mkpath(opath)
    ##
    filename = replace("$(Dates.Time(Dates.now()))", ":" => "_") 
    open(opath * filename * ".txt", "w") do io  
        for (key, val) in params
            println(io, "$key: $val")
        end
    end
    return opath*filename
end

"""
function makie_record(fig, node, framelist, params, niter, chaotic_or_spiking; fps=30)
"""
function makie_record(fig, node, framelist, params, niter, chaotic_or_spiking; fps=30)

    path = set_path_and_params(params, chaotic_or_spiking)

    GLMakie.record(fig, path * ".mp4", 1:niter; framerate = fps) do i
        node[] = framelist[i][:,:]
        sleep(1/fps)
    end
end

function no_offset(offset_array)
    n, m = size(offset_array)
    offset_array = OffsetArray(offset_array, 1:n, 1:m)
    array = convert(Array, offset_array)
    return array
end