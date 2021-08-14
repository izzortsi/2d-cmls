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



#system = [rand([0,1,2,3]) for x in 1:gsize, y in 1:gsize]

#system = fill(1, gsize, gsize)

#system = [x==y ? 1 : 0 for x in 1:gsize, y in 1:gsize]

#seed = [1 1 1 1; 1 2 2 1; 1 2 2 1; 1 1 1 1]
#seed = [1 2 2 1; 1 3 3 1; 1 3 3 1; 1 2 2 1]
#seed = [1 0 0 1; 1 2 2 1; 1 2 2 1; 1 0 0 1]
#seed = [2 0 0 2; 1 0 0 1; 1 0 0 1; 2 0 0 2]
#seed = [0 1 1 2 2 1 1 0; 2 1 1 0 0 1 1 2; 2 1 1 0 0 1 1 2; 0 1 1 2 2 1 1 0]
#seed = [0 1 1 2 2 1 1 0; 2 3 3 0 0 3 3 2; 2 3 3 0 0 3 3 2; 0 1 1 2 2 1 1 0]
#seed = [0 2 1 3 2 3 1 2 0;2 2 3 3 2 3 3 2 2;1 3 2 2 1 2 2 3 1;
#        3 3 2 2 1 2 2 3 3;2 2 1 1 0 1 1 2 2;3 3 2 2 1 2 2 3 3;
#        1 3 2 2 1 2 2 3 1;2 2 3 3 2 3 3 2 2;0 2 1 3 2 3 1 2 0]
seed = [3 3 2 2 1 2 2 3 3; 3 3 2 2 1 2 2 3 3;
        2 2 3 3 1 3 3 2 2; 2 2 3 3 1 3 3 2 2;
        1 1 1 0 0 0 1 1 1;
        2 2 3 3 1 3 3 2 2; 2 2 3 3 1 3 3 2 2;
        3 3 2 2 1 2 2 3 3; 3 3 2 2 1 2 2 3 3]
heatmap(seed)
ntimes = 20
proto_system = repeat(seed, ntimes, ntimes)

N, M = size(proto_system)

#proto_system[N÷8:7*(N÷8), M÷8:7*(M÷8)] .+= 1
#proto_system[N÷4:3*(N÷4), M÷4:3*(M÷4)] .+= 1

system=proto_system

N, M = size(system)

#sto =rand([0, 1], N, M)
maximum(system)
#system += sto
mod4(X) = mod(X, 5)
system = mod4.(system)

maximum(system)

heatmap(system)
#minimum(system)

outs = update(system, 1000, N, M)

#reduce(*, Int64.(maximum.(outs) .< 4))

#heatmap(outs[end])
#minimum(system)

heatmap(outs[1])
heatmap(outs[end])
#heatmap(outs[end-50])

anim = @animate for f in outs[1:100]
    heatmap(f)
end every 1

#gif(anim, fps=40, "ab_sandpile.gif")
gif(anim, fps=8)
