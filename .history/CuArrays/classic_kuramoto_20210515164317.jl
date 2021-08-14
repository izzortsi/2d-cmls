# %%
using Random
using DifferentialEquations
# %%
Random.seed!(0)
# %%

n = 5  
N = n ^ 2
K = 1
ω = rand(n, n) * 2 * π
dθ = rand(n, n) * 2 * π
# %%

function kuramoto!(dθ, p, t)
    
    for i in 1:length(enumerate(dθ))
        dθ[i] = ω[i] - ((K / N) * sum(sin.(dθ[i] .- dθ)))
    #        print((K / N) * np.sum(np.sin(θ_i - θ)), dθ[i])
        #println(i, " ", dθ[i])
    end
    dθ
end
# %%


tspan = (0.0,2.0)
prob = ODEProblem(kuramoto!, dθ, tspan)
# %%

sol = solve(prob)   

# %%



# %%
using GLMakie
using AbstractPlotting.Colors

# %%

# %%

# %%
# %%

fig, ax, hm = heatmap(sol.u[1])
# %%

# %%

# %%
# animation settings
n_frames = length(sol.t)
framerate = 3

# %%
record(figure, "test.gif") do io
    for i = n_frames
        hm[1] = sol.u[i]
        recordframe!(io)  # record a new frame
    end
end