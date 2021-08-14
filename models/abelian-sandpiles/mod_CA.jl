
using GLMakie
using Primes
GLMakie.inline!(false)

include("getnbs.jl")

max = 100
maxp = 20

N, M = 200, 200



#init_state = (x -> mod(x, 7)).(init_state)
#init_state

function state_seq(X, t)
    N, M = size(X)
    seq = []
    i = 0
    while i < t
        push!(seq, X)
        lim = rand(3:maxp)
        modulo = prevprime(lim)
        X = (x -> mod(x, modulo)).(X)
        #dx = rand(1:maxp, N, M)
        dx = rand(0:1, N, M)
        X += dx
        X .+= 1
        i += 1
    end
    return seq
end


#init_state = rand(1:100, N, M)

seed = [2 1 1 2; 0 2 3 0; 0 3 2 0; 1 2 2 1] * maxp
ntimes = 100
init_state = repeat(seed, ntimes, ntimes)


niter = 1000

outs = state_seq(init_state, niter)

scene = heatmap(outs[end], show_axis=false, colorrange=(0, maxp))

record(scene, "output_$(time()).mp4", range(1, stop=niter), sleep=false, framerate=30) do i
    heatmap!(outs[i], show_axis=false, colorrange=(0, maxp))
end
