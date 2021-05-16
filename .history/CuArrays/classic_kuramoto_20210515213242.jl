# %%
using Random
using DifferentialEquations
#using GLMakie
using Plots
# %%
using CUDA
#CUDA.allowscalar(false)
# %%

# %%

Random.seed!(0)
# %%

# %%
broadcast!

n = 50
N = n ^ 2
K = 2
ω = rand(n, n) * 2 * π |> cu
dθ = rand(n, n) * 2 * π |> cu
# %%
n = size(dθ, 1)
# %%
function kuramoto_gpu!(dθ, ω, t)
    n = size(dθ, 1)
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n
        val = ω[i, j] + dθ[i, j]
        summation = sum(dθ)
        sines = broadcast(CUDA.sin, dθ)
        #sines = CUDA.sin.(dθ)
        dθ[i, j] = val + summation


    end

    return nothing
end
# %%

"""
n is the size of the square n by n matrix that will be worked on
"""
function setup_kernel(n::Int64)
    dev = CuDevice(0)
    
    max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_threads_per_dim = sqrt(max_threads) / 2 |> Int
    
    numblocks = ceil(Int, n / max_threads_per_dim)
    threads = n ÷ numblocks
    
    return numblocks, threads
end
"""
returns a function `conv(n, A, ckern, outs, kdim)`, that performs a convolution of `ckern` over `A`;\\
    `A` is a `n` by `n` `CuArray` and `outs` must be similar to `A`;\\
    `ckern` is a `kdim` by `kdim` `CuArray`;
"""
function setup_call(n::Int64)
    numblocks, threads = setup_kernel(n)
    conv(A, p, t) = @cuda blocks = (numblocks, numblocks) threads = (threads, threads) kuramoto_gpu!(A, p, t)
    return conv
end
# %%


function kuramoto!(dθ, p, t)
    kura_gpu!(dθ, p, t)
    dθ
end
# %%
kura_gpu! = setup_call(n)
# %%

kuramoto!(dθ, ω, 1)
# %%

tspan = (0.0,5.0)

prob = ODEProblem(kuramoto!, dθ, tspan)
# %%

sol = solve(prob);   

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