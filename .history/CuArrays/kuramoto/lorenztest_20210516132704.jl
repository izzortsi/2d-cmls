# %%
using DifferentialEquations
using Plots
# %%

# %%

function lorenz!(du,u,p,t)
    f1(u) = 10.0*(u[2]-u[1])
    f2(u) = u[1]*(28.0-u[3]) - u[2]
    f3(u) = u[1]*u[2] - (8/3)*u[3]
    list_funs = [f1, f2, f3]
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
   end

# %%
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob)   
# %%

plot(sol,vars=(1,2,3))

# %%
plot(sol,vars=(0,2))