using LinearAlgebra
using Plots
include("getnbs.jl")

#rand([1,2,3,4])

#x
#y
#randvec[2]
#mod1(501, 500)
#test = [(1,2)]



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


seed = [2 1 1 2; 0 2 3 0; 0 3 2 0; 1 2 2 1]
#seed = [1 0 0 1; 1 2 2 1; 1 2 2 1; 1 0 0 1]
#seed = [2 0 0 2; 1 0 0 1; 1 0 0 1; 2 0 0 2]

#heatmap(seed)

ntimes = 100
proto_system = repeat(seed, ntimes, ntimes)

system=proto_system .+=1
maximum(system)

N, M = size(system)

Plots.heatmap(system)
#minimum(system)

outs = update(system, 1000, N, M)

#reduce(*, Int64.(maximum.(outs) .< 4))

#heatmap(outs[end])
#minimum(system)

heatmap(outs[1])
heatmap(outs[end])
#heatmap(outs[end-50])

anim = @animate for f in outs[1:1000]
    Plots.heatmap(f)
end every 1

gif(anim, fps=80, "ab_sandpile2.gif")
#gif(anim, fps=40)
