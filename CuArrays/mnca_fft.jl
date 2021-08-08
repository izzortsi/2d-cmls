#%%
using FFTW
using LinearAlgebra
using LazyGrids
using Random
using GLMakie
using ImageFiltering
using Noise
#%%
const SIZE = 1 << 9
const MID = SIZE ÷ 2
const O = SIZE - MID
Y, X = ndgrid(-O:(O-1), -O:(O-1))

#%%
R = SIZE ÷ 2

#%%
D(X, Y, R) = sqrt.((X/R) .^2 + (Y/R) .^2)
bell(x, m, s) = exp(-((x-m)/s)^2 / 2)
growth(U, m, s) = bell(U, m, s)*2-1
#%%



DGRID = D(X, Y, R)
#%%
#heatmap(DGRID)
#%%
#noised_DGRID = Noise.poisson(DGRID, SIZE ÷2)

#%%
#heatmap(noised_DGRID)

#%%
#noised_DGRID = Noise.mult_gauss(DGRID, 0.600)#, clip=false, σ=0.1, μ=1)
#heatmap(noised_DGRID)
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

    U::Matrix{Float64}
    G::Matrix{Float64}

    function MNCA(R, μ, σ, β, dt)

        function calc_kernel(mnca::MNCA)
            mnca.S = D(X, Y, R)
            mnca.K = (m.S .< 1) .* bell.(L.S, 0.5, 0.15)
            mnca.K_fft = fft(fftshift(L.K/sum(L.K)))
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



#%%
function show_configuration(mnca::MNCA)
    
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
function update!(A, mnca::MNCA)
    L.U[:,:] = real(ifft(L.K_fft .* fft(A)))[:,:]
    L.G[:,:] = L.δ.(L.U)[:,:]
    A[:, :] = clamp.(A + L.dt * L.G, 0, 1)[:, :]
end
#%%
function supdate!(A, mnca::MNCA)
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

function panels(A, mnca::MNCA)

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


function run(A, mnca::MNCA)

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

