using LinearAlgebra

gsize = 100
num_points = 10000
pos = rand(num_points, 2)*gsize
state = rand(num_points)
system = hcat(pos, state)

using Plots
scatter(pos[:, 1], pos[:, 2], marker_z = state, markerstrokealpha=0, markerstrokewidth=0, msize=1, size=(1200,800))

rand(1, 2)
new_point = rand(1, 2)*gsize

rownorm(mat) = sqrt.(mat[:,1].^2 .+ mat[:, 2].^2)

winner = argmin(rownorm(pos .- new_point))
system[winner,3]
function update(N)
    while i < N
        new_point = rand(1, 2)*gsize
        winidx = argmin(rownorm(pos .- new_point))
        add_state = rand()/2
        system[winidx, 3] += add_state
        if system[winidx, 3] >= 1.0
            #distribui pros vizinho


#fazer versão discreta
