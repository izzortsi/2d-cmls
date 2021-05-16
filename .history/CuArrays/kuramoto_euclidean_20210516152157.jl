# %%
using Random
using DifferentialEquations
#using GLMakie
using Plots
using Base.Cartesian
using LinearAlgebra
# %%
#using CUDA
#CUDA.allowscalar(false)
# %%

# %%

Random.seed!(1234)
# %%

# %%


n = 2^6
ω = randn(n, n) 
θ = randn(n, n) 
# %%
const N = n ^ 2
const K = 2
const Indices = CartesianIndices(θ)
const Positions = Tuple.(Indices)
# %%
aij = Positions[10, 10]
# %%

euclid_dists = (norm.(map(x-> x.-aij, Positions)))
# %%
euclid_dists[10, 10] = 4
# %%
euclid_dists .^= -2
# %%
ω .* θ 
# %%
ω * θ 
# %%
ω[1] * θ[1]

# %%

function kuramoto!(dθ, θ, p, t)
    for I in Indices
        x = Positions[I]        
        euclid_N = (norm.(map(y-> y.-x, Positions)))
        euclid_N[I] = 2*K
        euclid_N .^= -2
        dθ[I] = (ω[I] + K*sum((euclid_N .* sin.(θ .- θ[I])))
    end
end
# %%



# %%

tspan = (0.0,4.0)

prob = ODEProblem(kuramoto!, θ, tspan)
# %%

sol = solve(prob, adaptive=false, dt=0.1);   

# %%
# for (i, u) in enumerate(sol.u[end-50:end])
#     if i > 1
#         println(sum(u[i]-u[i-1]))
#     end
# end

n_frames = length(sol.t)

# %%

# %%
sol.u[end] == sol.u[end-1]
# %%
#nmax = maximum.(sol.u) |> maximum
#nmin = minimum.(sol.u) |> minimum
#i = 18
#fig = heatmap(sol.u[i])
# %%

for i in 1:n_frames
    fig = heatmap(sol.u[i])#, clims=(0, 2π))    
    savefig(fig, "frame$(i).png")
end
# %%


# %%
# fig, ax, hm = heatmap(sol.u[1])
# n_frames = length(sol.t)
# framerate = n_frames ÷ 7
# ax[1]
# %%


# record(fig, "test.mp4", framerate=framerate) do io
#     for i = 1:n_frames
#         heatmap!(sol.u[i])    
#         recordframe!(io)  # record a new frame
#     end
# end

# if i != 1
#     println(sum(sol.u[i] - sol.u[i-1]))
# end
# %%
# for i in 1:n_frames
#     fig, ax, hm = heatmap(sol.u[i])    
#     save("frame$(i).png", fig)
# end
# heatmap(sol.u[20])