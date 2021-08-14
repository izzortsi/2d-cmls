# %%
using Random
using DifferentialEquations
using GLMakie

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
ω = rand(n, n) * 2 * π #|> cu
dθ = rand(n, n) * 2 * π # |> cu
# %%

function kuramoto!(dθ, p, t)
    
    @loop for i in (1:size(dθ, 1); threadIdx().x)
        dθ[i] = ω[i] + ((K / N) * sum(sin.(dθ .- dθ[i])))
    #        print((K / N) * np.sum(np.sin(θ_i - θ)), dθ[i])
        #println(i, " ", dθ[i])
    end
    @synchronize
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



# %%
fig, ax, hm = heatmap(sol.u[1])
n_frames = length(sol.t)
framerate = n_frames ÷ 7
ax[1]
# %%


# record(fig, "test.mp4", framerate=framerate) do io
#     for i = 1:n_frames
#        heatmap!(sol.u[i])    
#         recordframe!(io)  # record a new frame
#     end
# end

# if i != 1
#     println(sum(sol.u[i] - sol.u[i-1]))
# end
# %%
for i in 1:n_frames
    heatmap!(sol.u[i])    
    save("frame$(i).png", ax)
end
