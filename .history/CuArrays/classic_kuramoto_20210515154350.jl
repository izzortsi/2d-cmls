
using Random
using DifferentialEquations
include("aux_funs.jl")
using .Aux
# %%
# %%
Random.seed!(0)
# %%

n = 5  
N = n ^ 2
K = 1
ω = rand(n, n) * 2 * π
dθ = rand(n, n) * 2 * π
# %%
#i =10
# %%
#dθ[i] = ω[i] - ((K / N) * sum(sin.(dθ[i] .- dθ)))

# %%



function kuramoto!(dθ, p, t)
    
    for i in 1:length(enumerate(dθ))
        dθ[i] = ω[i] - ((K / N) * sum(sin.(dθ[i] .- dθ)))
    #        print((K / N) * np.sum(np.sin(θ_i - θ)), dθ[i])
        println(i, " ", dθ[i])
    end
    dθ
end
# %%


tspan = (0.0,2.0)
prob = ODEProblem(kuramoto!, dθ, tspan)
# %%
length(sol)
# %%

sol = solve(prob)   

# %%


# %%

# %%


anim = @animate for i in length(sol)
    heatmap(sol[i])
end

# %%

gif(anim, "classic_kuramoto.gif", fps = 3)