# %%
using Random
using DifferentialEquations
#using GLMakie
using Plots
# %%
#using CUDA
#CUDA.allowscalar(false)
# %%

# %%

Random.seed!(0)
# %%

# %%


n = 50
const N = n ^ 2
const K = 2
ω = rand(n, n) * 2 * π 
dθ = rand(n, n) * 2 * π 

# %%

function kuramoto!(dθ, p, t)
    for i in 1:N
    dθ[i] = t*(ω[i] + (1/N)*sum(sin.(dθ .- dθ[i])))
    end
    dθ
end
# %%



# %%

tspan = (0.0,15.0)

prob = DiscreteProblem(kuramoto!, dθ, tspan)
# %%

sol = solve(prob, dt=0.5);   

# %%
# for (i, u) in enumerate(sol.u[end-50:end])
#     if i > 1
#         println(sum(u[i]-u[i-1]))
#     end
# end

n_frames = length(sol.t)

# %%

# %%

# %%
i = 18
fig = heatmap(sol.u[i])
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