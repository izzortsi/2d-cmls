using Plots

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


system=rand([0, 1, 2, 3], 50,50)
maximum(system)

N, M = size(system)

#hm = heatmap(system, interpolate=false, show_axis=false, colorrange=(0,4), color=system, colormap=[:black, :green, :purple, :yellow, :red])

#cp = contour(system, show_axis=false, colorrange=(0,4), backgroundcolor=:black)

#cm = colorlegend(hm[end], raw=true, width=(4, 480), textsize=6)
#scene_final = vbox(hm, cm)

niter = 3200

outs = update(system, niter, N, M)
heatmap(outs[1], clims = (0,4), color=:grays)

anim = @animate for f in outs[1:niter]
    heatmap(f, clims = (0,4), color=:grays)
end every 1

#gif(anim, fps=40, "ab_sandpile.gif")
gif(anim, fps=140, "ab_sandpile_long.gif")
