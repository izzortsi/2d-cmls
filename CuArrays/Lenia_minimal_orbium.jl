#%%
using FFTW
using LinearAlgebra
using LazyGrids
using Random
#%%
using GLMakie
# %%
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
# %%
mutable struct Lenia
    
    R::Int64
    μ::Float64
    σ::Float64
    β::Array{Float64}
    dt::Float64
    
    S::Matrix{Float64}
    K::Matrix{Float64}
    K_fft::Matrix{ComplexF64}
    
    calc_kernel::Function
    δ::Function

    U::Matrix{Float64}
    G::Matrix{Float64}

    function Lenia(R, μ, σ, β, dt)

        function calc_kernel(L::Lenia)
            L.S = D(X, Y, R)
            L.K = (L.S .< 1) .* bell.(L.S, 0.5, 0.15)
            L.K_fft = fft(fftshift(L.K/sum(L.K)))
        end

        δ(U) = growth(U, μ, σ)

        L = new(R, μ, σ, β, dt);
        calc_kernel(L)

        L.calc_kernel = calc_kernel
        L.δ = δ

        return L
    end
end
#%%

D(X, Y, R) = sqrt.((X/R) .^2 + (Y/R) .^2)
bell(x, m, s) = exp(-((x-m)/s)^2 / 2)
growth(U, m, s) = bell(U, m, s)*2-1

#%%
function show_configuration(L::Lenia)
    
    fig = Figure()
    μ_k = 0.5
    σ_k = 0.15
    x1 = LinRange(0, 3*μ_k, 120)
    x2 = LinRange(0, 3*L.μ, 120)

    k_core(x) = (x .< 1) .* bell.(x, μ_k, σ_k)

    ax1 = Axis(fig[1, 1], title = "Kernel's core section", xlims=(0, 1.5))
    l11 = lines!(ax1, x1, k_core.(x1), color = :cyan)

    ax2 = Axis(fig[1, 2], title="δ function")
    l2 = lines!(ax2, x2, L.δ.(x2), color =:green)

    ax3, hm1 = heatmap(fig[2, 1], L.K, axis=(title = "Kernel's core",))
    ax4, hm2 = heatmap(fig[2, 2], real(L.K_fft), axis=(title = "Kernel's FFT",))#, axis=(title = "title", ))
    
    linkaxes!(ax3, ax4)
    hideydecorations!(ax4, grid = false)
    hidexdecorations!(ax4, grid = false)
    fig
end

#%%
function update!(A, L::Lenia)
    L.U[:,:] = real(ifft(L.K_fft .* fft(A)))[:,:]
    L.G[:,:] = L.δ.(L.U)[:,:]
    A[:, :] = clamp.(A + L.dt * L.G, 0, 1)[:, :]
end
#%%
function supdate!(A, L::Lenia)
    A_ = A .* (A .>= 1) * 0.9
    nA_ = A .* (A .< 1) * 1.05
    U1 = real(ifft(L.K_fft .* fft(A_)))
    U2 = real(ifft(L.K_fft .* fft(nA_)))
    L.U[:,:] = (U1+U2)[:,:]
    L.G[:,:] = L.δ.(L.U)[:,:]
    AdA = (A + L.dt * L.G)
    A[:, :] = clamp.(AdA, 0, 1)[:, :]

end

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
L = Lenia(R, m, s, b, 1/5)
#%%
show_configuration(L)
#%%
 
A = zeros(SIZE, SIZE)
#%%
populate(A, orbium["cells"], 30)
#%%

function panels(A, L::Lenia)

    L.U = real(ifft(L.K_fft .* fft(A)))
    L.G = L.δ.(L.U)

    fig = Figure()

    nA = Node(A)
    heatmap(fig[1, 1], nA)

    nU = Node(L.U)
    heatmap(fig[1, 2], nU)

    nG = Node(L.G)
    heatmap(fig[2, 1], nG)

    fig, nA, nU, nG
end


#%%
fig, nA, nU, nG = panels(A, L)

#%%
fig
#%%


function run(A, L::Lenia)

    #fig, nA, nU, nG = panels(A, L)

    fps = 60
    nframes = 360

    for i = 1:nframes
        supdate!(A, L)
        nA[] = A
        nU[] = L.U
        nG[] = L.G
        sleep(1/fps) # refreshes the display!
    end
end
#%%
#run(A, L)
#%%

function record_run(fig, A, L; nframes = 600, fps=36)
    opath = pwd() * "/CuArrays/outputs/" * "lenia/"
    mkpath(opath)
    GLMakie.record(fig, opath * "orbia.mp4", 1:nframes; framerate = fps) do i
        supdate!(A, L)
        nA[] = A
        nU[] = L.U
        nG[] = L.G
        sleep(1/fps) # refreshes the display!
    end
end

#%%

record_run(fig, A, L)

