#%%
using FFTW
using LinearAlgebra
using LazyGrids
using Random
using GLMakie
#%%
const SIZE = 1 << 9
const MID = SIZE ÷ 2
const O = SIZE - MID
Y, X = ndgrid(-O:(O-1), -O:(O-1))

#%%
orbium = Dict("name" => "Orbium","R" => 13,"T"=>10,"m"=>0.15,"s"=>0.015,"b"=>[1],
  "cells" => [[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
  [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
  [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
  [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
  [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
  [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
  [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
  [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
  [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
  [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
  [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
  [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
  [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
  [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
  [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
  [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
  [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
  [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
  [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
  [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
#%%
orbium["cells"] = hcat(orbium["cells"]...)
#%%
R = orbium["R"]
m = orbium["m"]
s = orbium["s"]
b = orbium["b"]
T = orbium["T"]

#%%
D(X, Y, R) = sqrt.((X/R) .^2 + (Y/R) .^2)
bell(x, m, s) = exp(-((x-m)/s)^2 / 2)
growth(U, m, s) = bell(U, m, s)*2-1
F(x) = cos(SIZE*x)^3 + sin((SIZE ÷2)*x + π/2)

# %%
mutable struct MNCA
    A::Matrix{Float64}
    R::Int64
    μ::Float64
    σ::Float64
    β::Array{Float64}
    dt::Float64
    
    S::Matrix{Float64}
    K::Vector{Matrix{Float64}}
    K_fft::Vector{Matrix{ComplexF64}}
    
    calc_kernels::Function
    δ::Function
    Φ::Vector{Function}
    update!::Function
    populate!::Function

    U::Matrix{Float64}
    G::Matrix{Float64}

    function MNCA(A, R, μ, σ, β, dt)

        function calc_kernel(M::MNCA)
            M.S = D(X, Y, R)
            M.K = Vector{Matrix{Float64}}(undef, 0)
            K1 = (M.S .< 1) .* bell.(M.S, 0.5, 0.15)
            push!(M.K, K1)
            K2 = clamp.((M.S .< 1) .* F.(bell.(M.S, 0.5, 0.15)), 0, 1)
            push!(M.K, K2)
            M.K_fft = Vector{Matrix{ComplexF64}}(undef, 0)
            push!(M.K_fft, fft(fftshift(M.K[1]/sum(M.K[1]))))
            push!(M.K_fft, fft(fftshift(M.K[2]/sum(M.K[2]))))
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


A = zeros(SIZE, SIZE)
#%%


M = MNCA(A, R, m, s, b, 1/5)

#%%
function update1(M::MNCA)
    M.U[:,:] = real(ifft(M.K_fft[1] .* fft(M.A)))[:,:]
    M.G[:,:] = M.δ.(M.U)[:,:]
    M.A[:, :] = clamp.(M.A + M.dt * M.G, 0, 1)[:, :]
end
function update2(M::MNCA)
    M.U[:,:] = real(ifft(M.K_fft[2] .* fft(M.A)))[:,:]
    M.G[:,:] = M.δ.(M.U)[:,:]
    M.A[:, :] = clamp.(M.A + M.dt^2 * M.G, 0, 1)[:, :]
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
        W[(x-cx ÷ 2):(x+cx ÷ 2-1), (y-cx ÷ 2):(y+cx ÷ 2 -1)] = creature_cells
    end
end

#%%
M.A = A
populate(M.A, orbium["cells"], 30)
#%%
M.populate! = () -> populate(M.A, orbium["cells"], 1)
#%%


function panels(M::MNCA)

    M.U = real(ifft(M.K_fft[1] .* fft(M.A)))
    U2 = real(ifft(M.K_fft[2] .* fft(M.A)))
    M.G = M.δ.(M.U)

    fig = Figure()

    nA = Node(M.A)
    heatmap(fig[1, 1], nA)

    nU = Node(M.U)
    heatmap(fig[1, 2], nU)

    nG = Node(M.G)
    heatmap(fig[2, 1], nG)

    nU2 = Node(U2)
    heatmap(fig[2, 2], U2)

    fig, nA, nU, nG, nU2
end


#%%
fig, nA, nU, nG, nU2 = panels(M)

#%%
fig
#%%

function run(M::MNCA)

    #fig, nA, nU, nG = panels(A, M)

    fps = 60
    nframes = 360

    for i = 1:nframes
        M.Φ[1](M)
        M.Φ[2](M)
        nA[] = M.A
        nU[] = M.U
        nG[] = M.G
        sleep(1/fps) # refreshes the display!
    end
end
#%%
#run(M)
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