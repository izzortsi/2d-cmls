# %%
using Random
using DifferentialEquations
include("aux_funs.jl")
using .Aux
using Plots
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
n_frames = length(sol.t)


# %%

for i in n_frames
    println(typeof(sol.u[i]))
end
# %%


anim = @animate for i in n_frames
    heatmap(sol.u[i])
end

# %%

gif(anim, "classic_kuramoto.gif", fps = 3)

# %%

make_gif(sol.u)


# %%
plot(Plots.fakedata(50, 5), w = 3)

p = plot([sin, cos], zeros(0), leg = false, xlims = (0, 2π), ylims = (-1, 1))
anim = Animation()
for x = range(0, stop = 2π, length = 20)
    push!(p, x, Float64[sin(x), cos(x)])
    frame(anim)
end
# %%
gif(anim, "filename.gif", fps=15)
