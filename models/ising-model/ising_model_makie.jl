using GLMakie
using Dates
using OffsetArrays
using TestImages, Colors, Images
#%%

function make_initial_config(;initialization="rand")
    if initialization=="rand"
        return rand([-1.0, 1.0], dims...)
    else
        img = testimage("cam")
        img = Gray.(img)
        img = imrotate(img, π/2)
        n, m = size(img)
        img = OffsetArray(img, 1:n, 1:m)
        img = convert(Array{Float64}, img)
        return img
    end
end
#%%

function update_site(config, N, M, n, m, β)

    nbs_total = 0

    for i in n-1:n+1
        for j in m-1:m+1
            if i != j
                nbs_total += config[mod1(i, N), mod1(j, M)]
            end
        end
    end

    dE = 2 * config[n, m] * nbs_total

    if dE <= 0
        config[n, m] *= (-1.)
    elseif exp(-dE * β) > rand()
        config[n, m] *= (-1.)
    end
    return config
end
#%%



function update_config(config, N, M, β=0.4)

    for i in 1:N
        for j in 1:M
            update_site(config, N, M, i, j, β)
        end
    end

    return config

end

#%%


function make_frames(config, dims, steps = 120, β = 0.4)

    list = [config]

    for r in 1:steps

        update_config(config, dims..., β)

        push!(list, copy(config))

    end

    return list

end
#%%



# function make_gif(clist; path::String="~/Dropbox/Julia/2DCMLs/", contour::Bool=false, fps=2)

#     anim = @animate for i = 1:steps

#         heatmap(clist[i], c=:grays, xaxis=false, yaxis=false, legend = false, clims=(-1., 1))
#         #title!("$(fps) fps, iter $(i)/$(steps); params: e=$(e), b=$(b), a=$(Float16(a))")
#     end every 1

#     gif(anim, "$(path)/$(Dates.Time(Dates.now())).gif", fps=fps)
# end
# #%%

#%%



function record_from_list(steps, list, fps = 24)
    output_path = joinpath(@__DIR__, mkpath(string(Dates.Date(Dates.now()))))
    filename = "ising_" * replace(string(Dates.Time(Dates.now())), ":" => "_") * ".mp4"
    frame = Node(list[1])
    fig, hm = heatmap(frame)
    frames = 1:steps
    record(fig, joinpath(output_path, filename), frames; framerate = fps) do i
        frame[] = list[i][:,:]
        sleep(1/fps)
    end
end



#program parameters

steps = 140
dims = (500, 500)
out_path=pwd()
initialization="rand" #or "img"
#%%



iconfig=make_initial_config(initialization=initialization)
n,m =size(iconfig)
#%%


list = make_frames(iconfig, dims, steps, 0.17)
#%%



record_from_list(steps, list)

# %%

