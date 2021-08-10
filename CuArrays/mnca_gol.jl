#%%
using CUDA
using LinearAlgebra
using LazyGrids
using Random
using GLMakie
#%%
const SIZE = 1 << 9
const MID = SIZE ÷ 2
const O = SIZE - MID
Y, X = ndgrid(-O:(O-1), -O:(O-1))
# %%
include("convolution.jl")
using .Convolution
# %%

CUDA.allowscalar(false)
#%%
creature = Dict("name" => "Glider","R" => 13,"T"=>10,"m"=>0.15,"s"=>0.015,"b"=>[1],
  "cells" => [0 0 0 0 0;
              0 0 1 0 0;
              0 0 0 1 0;
              0 1 1 1 0;
              0 0 0 0 0]
            )
            
#%%
creature["cells"] |> cu # = hcat(creature["cells"]...)
#%%
R = creature["R"]
m = creature["m"]
s = creature["s"]
b = creature["b"]
T = creature["T"]

#%%
D(X, Y, R) = sqrt.((X/R) .^2 + (Y/R) .^2)
bell(x, m, s) = exp(-((x-m)/s)^2 / 2)
growth(U, m, s) = bell(U, m, s)*2-1
F(x) = cos(SIZE*x)^3 + sin((SIZE ÷2)*x + π/2)

conv = setup_convolution(SIZE)

# %%

conv

# %%
DHMatrix = Union{Matrix{Float64}, CuArray{Float32, 2}}
# %%
SIZE
# %%

mutable struct MNCA
    A::DHMatrix
    R::Int64
    μ::Float64
    σ::Float64
    β::Array{Float64}
    dt::Float64
    
    S::Matrix{Float64}
    K::Vector{DHMatrix}
    K_fft::Vector{Matrix{ComplexF64}}
    
    calc_kernels::Function
    δ::Function
    Φ::Vector{Function}
    update!::Function
    populate!::Function

    U::DHMatrix
    G::DHMatrix

    function MNCA(A, R, μ, σ, β, dt)

        function calc_kernel(M::MNCA)
            M.S = D(X, Y, R)
            M.K = Vector{DHMatrix}(undef, 0)
            K1 = [1. 1. 1.; 1. 1. 1.; 1. 1. 1.]
            #K1 = [1. 1. 1.; 1. 1. 1.; 1. 1. 1.]
            push!(M.K, K1 |> cu)
            K2 = copy(K1)
            push!(M.K, K2)
        end

        δ(U) = growth(U, μ, σ)

        M = new(A, R, μ, σ, β, dt);
        calc_kernel(M)

        M.calc_kernels = calc_kernel
        M.δ = δ

        return M
    end
end
#%%


A = zeros(SIZE, SIZE) |> cu
#%%


M = MNCA(A, 1, m, s, b, 1/5)

#%%
M.U = CUDA.similar(A)
M.G = CUDA.similar(A)
M.δ = x -> x/9

#%%

M.K[1]
# %%
U = CUDA.similar(A)

#%%
conv(M.A, M.K[1], M.U)

#%%
M.G .= (M.δ.(M.U) .* M.A)
# %%
M.A = ((M.U .== 2) .* M.A .+ (M.U .== 3))
# %%
view(M.A, 10:15, 10:15)
# %%
M.A[10:14, 10:14] = creature["cells"]
# %%
M.A
# %%
M.K[1] = cu([1. 1. 1.; 1. 0 1.; 1. 1. 1.])
# %%


function update1(M::MNCA)
    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ.(M.U) .* M.A)
    M.A = ((M.U .== 2) .* M.A .+ (M.U .== 3))
end
function update2(M::MNCA)
    # M.U[:,:] = real(ifft(M.K_fft[1] .* fft(M.A)))[:,:]
    # M.G[:,:] = (M.δ.(M.U) .* M.A)[:,:]
    # M.A[:, :] = Float64.((M.U .== 2) .* M.A + (M.U .== 3))[:, :]
    nothing
end

function update!(M::MNCA)
    for ϕ in M.Φ
        ϕ(M)
    end
end

#%%
M.Φ = Vector{Function}(undef, 0)
push!(M.Φ, update1)
push!(M.Φ, update2)
M.update! = update!
#%%
function populate(W, creature_cells, num_creatures)
    cx, cy = size(creature_cells)
    to_populate = rand(cx:SIZE-cx, (num_creatures, 2))
    for row in eachrow(to_populate)
        x, y = row
        W[(x-cx ÷ 2):(x+cx ÷ 2), (y-cx ÷ 2):(y+cx ÷ 2)] = creature_cells
    end
end
# %%
creature["cells"] = cu(creature["cells"])
#%%
M.A = A
populate(M.A, creature["cells"], 30)
#%%
M.populate! = () -> populate(M.A, creature["cells"], 1)
#%%


function panels(M::MNCA)

    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ.(M.U) .* M.A)

    fig = Figure()

    dnA = Node(M.A)
    hnA = lift(Array, dnA)
    heatmap(fig[1, 1], hnA)

    dnU = Node(M.U)
    hnU = lift(Array, dnU)
    heatmap(fig[1, 2], hnU)

    dnG = Node(M.G)
    hnG = lift(Array, dnG)
    heatmap(fig[2, 1], hnG)

    fig, dnA, hnA, dnU, hnU, dnG, hnG 
end


#%%
fig, dnA, hnA, dnU, hnU, dnG, hnG = panels(M)

#%%
fig#
#%%
M.update!(M)
#%%

function run(M::MNCA)

    #fig, nA, nU, nG = panels(A, M)

    fps = 60
    nframes = 360

    for i = 1:nframes
        M.Φ[1](M)
        dnA[] = M.A
        dnU[] = M.U
        dnG[] = M.G
        sleep(1/fps) # refreshes the display!
    end
end
#%%
run(M)
#%%

function record_run(fig, M; nframes = 600, fps=36)
    opath = pwd() * "/CuArrays/outputs/" * "lenia/"
    mkpath(opath)
    GLMakie.record(fig, opath * "mnorbia.mp4", 1:nframes; framerate = fps) do i
        M.Φ[1](M)
        M.Φ[2](M)
        nA[] = M.A
        nU[] = M.U
        nG[] = M.G
        sleep(1/fps) # refreshes the display!
    end
end

#%%

#record_run(fig, M)

#%%

include("sim_manager_min.jl")


#%%

simulate(M, fps=30)