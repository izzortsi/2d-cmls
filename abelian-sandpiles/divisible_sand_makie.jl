using AbstractPlotting
using GLMakie

AbstractPlotting.inline!(false)

include("getnbs.jl")


function rec_up!(system, x, y, N, M, new_val)

    system[x, y] += new_val

    if system[x, y] >= 1.0

        system[x, y] -= 1.0

        #toadd = system[x, y]/2
        #system[x, y] /= 2.0
        neighs = get_vn_neighborhood(x, y, N, M)
        L = length(neighs)
        toadd = 1.0/L

        for n in neighs
            rec_up!(system, n..., N, M, toadd)
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
        new_val = rand()
        system = rec_up!(system, x, y, N, M, new_val)

        push!(output, copy(system))

        i+=1

    end
    return output
end


#seed = [1 1 1 1; 1 2 2 1; 1 2 2 1; 1 1 1 1]
#seed = [1 0 0 1; 1 2 2 1; 1 2 2 1; 1 0 0 1]
#seed = [2 0 0 2; 1 0 0 1; 1 0 0 1; 2 0 0 2]


seed = [2 1 1 2; 0 2 3 0; 0 3 2 0; 1 2 2 1] * rand()/2
ntimes = 100
proto_system = repeat(seed, ntimes, ntimes)
proto_system[50:350, 50:350] .+= 0.3
system=proto_system


#system = rand(N, M)
#system[75:325, 75:325] .+= 0.3
maximum(system)

N, M = size(system)

hm = heatmap(system, interpolate=false, colorrange=(0,2), show_axis=false)

#cp = contour(system, show_axis=false, colorrange=(0,4), backgroundcolor=:black)

#cm = colorlegend(hm[end], raw=true, width=(4, 480), textsize=6)
#scene_final = vbox(hm, cm)

niter = 1500

outs = update(system, niter, N, M)

scene = heatmap(outs[1], show_axis=false, colorrange=(0,2))
scene_f = heatmap(outs[end], show_axis=false, colorrange=(0,2))
#cm = colorlegend(scene[end], raw=true, width=(4, 480), textsize=6)
#scene_final = vbox(scene, cm)

record(scene, "output.mp4", range(1, stop=niter), sleep=false, framerate=80) do i
    heatmap!(outs[i], show_axis=false, colorrange=(0,2))
end
