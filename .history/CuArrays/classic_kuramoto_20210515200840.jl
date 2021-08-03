# %%
using Random
using DifferentialEquations
#using GLMakie
using Plots
# %%
#using CUDA
#CUDA.allowscalar(false)
# %%
using GPUifyLoops
# %%

Random.seed!(0)
# %%

n = 100
N = n ^ 2
K = 2
ω = rand(N) * 2 * π #|> cu
dθ = rand(N) * 2 * π # |> cu
# %%

function kuramoto!(dθ, p, t)
    
    for i in 1:N
        dθ[i] = ω[i] + ((K / N) * sum(sin.(dθ .- dθ[i])))
    end
    
    dθ
end
# %%
kuramoto!(dθ, nothing, nothing)
# %%

tspan = (0.0,15.0)

prob = ODEProblem(kuramoto!, dθ, tspan)
# %%

sol = solve(prob);   

# %%
# for (i, u) in enumerate(sol.u[end-50:end])
#     if i > 1
#         println(sum(u[i]-u[i-1]))
#     end
# end

heatmap(sol.u[50])

# %%

for i in 1:n_frames
    fig, ax, hm = heatmap(sol.u[i])    
    save("frame$(i).png", fig)
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