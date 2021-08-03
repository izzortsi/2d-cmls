# %%
using Random
using DifferentialEquations
#using GLMakie
using Plots
# %%
using CUDA
CUDA.allowscalar(false)
# %%

# %%

#Random.seed!(0)
# %%

# %%


n = 10
const N = n ^ 2
const K = 2
ω = CUDA.rand(N) 
dθ = CUDA.rand(N) 
θ = copy(dθ)
# %%

# %%
function f_sin!(f, out, a, b)
    i = threadIdx().x
    out[i] = f(a[i], b)
    out[i] = CUDA.sin(out[i])
    return
  end

function kernel(a, o, da)
    i = threadIdx().x
    @cuda threads=length(a) dynamic=true f_sin!(-, da, a, a[i])
    sync_threads()
    da[i] = o[i] + K*sum(da)
    a[i] = da[i]
    return nothing
end
# %%
@cuda threads=length(θ) kernel(θ, ω, dθ)
# %%
#a = CUDA.rand(100)
#o = CUDA.rand(100)
#da = similar(a)
# %%

kura_gpu!(θ, ω, dθ) = @cuda threads=length(θ) kernel(θ, ω, dθ)
# %%
@cuda threads=length(θ) kernel(a, o, da)
# %%

kura_gpu!(θ, ω, dθ)
# %%

function kuramoto!(dθ, p, t)

    kura_gpu!(θ, ω, dθ)
    dθ
end
# %%
# %%
# %%

kuramoto!(dθ, nothing, 1)
# %%

tspan = (0.0,1.0)

prob = ODEProblem(kuramoto!, dθ, tspan)
# %%

sol = solve(prob);   

# %%
# for (i, u) in enumerate(sol.u[end-50:end])
#     if i > 1
#         println(sum(u[i]-u[i-1]))
#     end
# end

##
sols = sol.u .|> Array
# %%
sols = reshape.(sols..., (10, 10))

# %%

sol.t
##
n_frames = length(sol.t)

# %%

# %%

# %%

for i in 1:n_frames
    fig = heatmap(sol.u[i], clims=(0, 2π))    
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