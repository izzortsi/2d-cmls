using AbstractPlotting
using GLMakie

AbstractPlotting.inline!(false)

include("getnbs.jl")


function rec_up!(system, x, y, N, M)

    system[x, y] += 1

    if system[x, y] >= 4

        system[x, y] -= 4
        neighs = get_vn_neighborhood(x, y, N, M)

        for n in neighs
            rec_up!(system, n..., N, M)
        end
    end
    return system
end


function update(system, t, N, M)

    output = []

    i = 0

    while i < t

        x = rand(1:N)
        y = rand(1:M)

        system = rec_up!(system, x, y, N, M)

        push!(output, copy(system))

        i+=1

    end
    return output
end


#seed = [1 1 1 1; 1 2 2 1; 1 2 2 1; 1 1 1 1]
#seed = [2 1 1 2; 0 2 3 0; 0 3 2 0; 1 2 2 1]
seed = [2 1 3 1 2; 0 2 1 3 0; 3 0 1 0 3; 0 3 1 2 0; 1 2 3 2 1]
#seed = [1 0 0 1; 1 2 2 1; 1 2 2 1; 1 0 0 1]
#seed = [2 0 0 2; 1 0 0 1; 1 0 0 1; 2 0 0 2]

#heatmap(seed)

ntimes = 100


proto_system = repeat(seed, ntimes, ntimes)

proto_system[50:450, 50:450] .+=1
system=proto_system
maximum(system)

N, M = size(system)

#hm = heatmap(system, interpolate=false, show_axis=false, colorrange=(0,4), color=system, colormap=[:black, :green, :purple, :yellow, :red])

#cp = contour(system, show_axis=false, colorrange=(0,4), backgroundcolor=:black)

#cm = colorlegend(hm[end], raw=true, width=(4, 480), textsize=6)
#scene_final = vbox(hm, cm)

niter = 1000

outs = update(system, niter, N, M)

scene = heatmap(outs[1], show_axis=false, colorrange=(0,4), colormap=[:black, :green, :purple, :yellow, :red])
scenef = heatmap(outs[end], show_axis=false, colorrange=(0,4), colormap=[:black, :green, :purple, :yellow, :red])
#cm = colorlegend(scene[end], raw=true, width=(4, 480), textsize=6)
#scene_final = vbox(scene, cm)

record(scene, "output.mkv", range(1, stop=niter), sleep=false, framerate=80) do i
    heatmap!(outs[i], show_axis=false, colorrange=(0,4), colormap=[:black, :green, :purple, :yellow, :red])
end
