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
generic_kernel(S, β; μ = 0.5, σ = 0.15) = (S*β .< 1/β) .* bell.(S*β, μ, σ)
F(x) = cos(SIZE*x)^3 + sin((SIZE ÷2)*x + π/2)

# %%
mutable struct MNCA
    A::Matrix{Float64}
    R::Int64
    μ::Vector{Float64}
    σ::Vector{Float64}
    β::Array{Float64}
    dt::Float64
    
    S::Matrix{Float64}
    K::Vector{Matrix{Float64}}
    K_fft::Vector{Matrix{ComplexF64}}
    
    calc_kernels::Function
    δ::Vector{Function}
    Φ::Vector{Function}
    update!::Function
    populate!::Function

    U::Vector{Matrix{Float64}}
    G::Vector{Matrix{Float64}}

    function MNCA(A, R, μ, σ, β, dt)

        # function calc_kernel(M::MNCA)
        #     M.S = D(X, Y, M.R)
        #     M.K = Vector{Matrix{Float64}}(undef, 0)
        #     K1 = (M.S .< 1) .* bell.(M.S, 0.5, 0.15)
        #     push!(M.K, K1)
        #     K2 = clamp.((M.S .< 1) .* F.(bell.(M.S, 0.5, 0.15)), 0, 1)
        #     push!(M.K, K2)
        #     M.K_fft = Vector{Matrix{ComplexF64}}(undef, 0)
        #     push!(M.K_fft, fft(fftshift(M.K[1]/sum(M.K[1]))))
        #     push!(M.K_fft, fft(fftshift(M.K[2]/sum(M.K[2]))))
        # end

        # δ(U) = growth(U, μ, σ)

        M = new(A, R, μ, σ, β, dt);
        M.K = Vector{Matrix{Float64}}(undef, 0)
        M.K_fft = Vector{Matrix{ComplexF64}}(undef, 0)
        M.δ = Vector{Function}(undef, 0)
        # calc_kernel(M)

        # M.calc_kernels = calc_kernel
        # 

        return M
    end
end
#%%


A = zeros(SIZE, SIZE)
#%%


M = MNCA(A, 13, [0.15], [0.015], [1, 0.1], 1/T)
#%%
M.S = D(X, Y, M.R)
#%%

M.K = []
#%%


push!(M.K, generic_kernel(M.S, M.β[1]))
# (M.S .< 1) .* bell.(M.S, 0.5, 0.15))
#%%


#push!(M.K, (M.S*5 .< 1*5) .* bell.(M.S*5, 0.5, 0.15))
push!(M.K, generic_kernel(M.S, M.β[2], σ=0.4))
# (M.S*0.1 .< 1/0.1) .* bell.(M.S*0.1, 0.5, 0.4)
#%%

M.δ = [U -> growth.(U, M.μ[1], M.σ[1])]
#%%

push!(M.δ, U -> growth.(U, M.μ[1]/M.β[2], M.σ[1]/M.β[2]))

#%%
#K1 = (M.S .< 1) .* bell.(M.S, 0.5, 0.15)
F(x) =  ((cos(SIZE*x)^3 + sin((SIZE ÷2)*x + π/2))/2 -1)^2
#%%


K1 = (M.S .< 1) .* bell.(M.S, 0.11, 0.07)
#%%

#%%
M.K[2] = clamp.( F.(generic_kernel(M.S, M.β[2], σ=0.2, μ=0.7))/3, 0, 1)
#%%
M.δ[2] = U -> 0.2*growth.(U, 2+M.μ[1]/M.β[2], 5+10*M.σ[1]/M.β[2])
#%%



function plot_kernels(M)
    fig = Figure()
    ax1, hm1 = heatmap(fig[1, 1], M.K[1], colorrange=(0, 1))
    ax2, hm2 = heatmap(fig[1, 2], M.K[2], colorrange=(0, 1))
    ax3, hm3 = heatmap(fig[2, 1], M.δ[1].(M.S), colorrange=(0, 1))
    ax4, hm4 = heatmap(fig[2, 2], M.δ[2].(M.S), colorrange=(0, 1))
    fig
end
# heatmap(K1 + K2, colorrange=(0,1))
# heatmap(K1 + K2, colorrange=(0,1))
#%%
plot_kernels(M)
#%%
function calc_kernel(M::MNCA; R2=32, m2=0.48, s2=0.16)
    M.S = D(X, Y, M.R)
    M.K = Vector{Matrix{Float64}}(undef, 0)
    F(x) =  ((cos(SIZE*x)^3 + sin((SIZE ÷2)*x + π/2))/2 -1)^2
    K1 = generic_kernel(M.S, M.β[1])
    K2 = clamp.( F.(generic_kernel(M.S, M.β[2], σ=0.2, μ=0.7))/3, 0, 1)
    push!(M.K, K1)
    push!(M.K, K2)
    M.K_fft = Vector{Matrix{ComplexF64}}(undef, 0)
    push!(M.K_fft, fft(fftshift(M.K[1]/sum(M.K[1]))))
    push!(M.K_fft, fft(fftshift(M.K[2]/sum(M.K[2]))))
end
#%%

function update1(M::MNCA)
    M.U[1][:,:] = real(ifft(M.K_fft[1] .* fft(M.A)))[:,:]
    M.G[1][:,:] = M.δ[1](M.U[1])[:,:]
    M.A[:, :] = clamp.(M.A + M.dt * M.G[1], 0, 1)[:, :]
end
function update2(M::MNCA)
    M.U[2][:,:] = real(ifft(M.K_fft[2] .* fft(M.A)))[:,:]
    M.G[2][:,:] = M.δ[2](M.U[2])[:,:]
    M.A[:, :] = clamp.(M.A + M.dt^3 * M.G[2], 0, 1)[:, :]
end

function update!(M::MNCA)
    for ϕ in M.Φ
        ϕ(M)
    end
end

M.calc_kernels = calc_kernel
#%%
M.calc_kernels(M)
#%%

plot_kernels(M)
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

M.U = []
M.G = []

function panels(M::MNCA)
    
    push!(M.U, real(ifft(M.K_fft[1] .* fft(M.A))))
    push!(M.U, real(ifft(M.K_fft[2] .* fft(M.A))))

    push!(M.G, M.δ[1].(M.U[1]))
    push!(M.G, M.δ[2].(M.U[2]))

    fig = Figure(resolution=(960, 960))

    nA = Node(M.A)
    heatmap(fig[1, :], nA, colorrange=(0,1))

    nU1 = Node(M.U[1])
    heatmap(fig[2, 1], nU1, colorrange=(0,1))
    
    nU2 = Node(M.U[2])
    heatmap(fig[2, 2], nU2, colorrange=(0,1))

    nG1 = Node(M.G[1])
    heatmap(fig[3, 1], nG1, colorrange=(0,1))

    nG2 = Node(M.G[2])
    heatmap(fig[3, 2], nG2, colorrange=(0,1))

    # nU2 = Node(U2)
    # heatmap(fig[2, 2], U2, colorrange=(0,1))

    fig, nA, nU1, nU2, nG1, nG2
end


#%%
fig, nA, nU1, nU2, nG1, nG2 = panels(M)

#%%
fig
#%%
M.update!(M)
#%%


using Observables, GLMakie, Dates
# #%%
# fig = Figure()
# #%%
# fig.scene
# #%%
# events(fig).window_open[]
# #%%




function simulate(M; resolution=(1280, 720), fps = 24)
    


    fig = Figure(resolution=resolution)

    #modelobs = Observable(model.A)
    run_obs = Observable{Bool}(false)
    rec_obs = Observable{Bool}(false)

    #running_label = Label(fig[0, :], lift(x -> x ? "RUNNING" : "HALTED", run_obs))

    #ax1 = Axis(fig[1, 1])
    #heatmap!(ax1, modelobs)
    #fig, hm = heatmap(modelobs[].A)


    nA = Node(M.A)
    ax1, hm1 = heatmap(fig[1, 1], nA, colorrange=(0, 1))
    hidedecorations!(ax1)

    nU = Node((M.U[1] + M.U[2])/2)
    ax2, hm2 = heatmap(fig[1, 2], nU, colorrange=(0, 1))
    hidedecorations!(ax2)

    nG = Node((M.G[1] + M.G[2])/2)
    ax3, hm3 = heatmap(fig[2, 1], nG, colorrange=(0, 1))
    hidedecorations!(ax3)

    dims, = size(M.K[1])
    mid = dims ÷ 2
    r = M.R

    heatmap(fig[2,2][1,1], M.K[1][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))
    heatmap(fig[2,2][1,2], M.K[2][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))

    stream = VideoStream(fig.scene, framerate=fps)
    #fig

    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
           if event.key == Keyboard.s
                run_obs[] = !run_obs[]
                run_obs[] ? println("Simulation running. $(run_obs[])") : println("Simulation stopped.")

                @async while events(fig).window_open[] && run_obs[] 
                    # update observables in scene
                    M.update!(M)
                    nA[] = M.A[:,:]
                    nU[] = ((M.U[1] + M.U[2])/2)[:,:]
                    nG[] = ((M.G[1] + M.G[2])/2)[:,:]
                    sleep(1 / fps)
                end
            elseif event.key == Keyboard.a
                
                M.populate!()
                nA[] = M.A[:,:]

            elseif event.key == Keyboard.c
                
                A = zeros(SIZE, SIZE)
                M.A[:,:] = A[:,:]
                M.U[1][:, :] = A[:,:]
                M.U[2][:, :] = A[:,:]
                M.G[1][:, :] = A[:,:]
                M.G[2][:, :] = A[:,:]
                populate(M.A, orbium["cells"], 30)
                M.update!(M)

            elseif event.key == Keyboard.r
                if !rec_obs[]
                    # start recording
                    # start a new stream and set a new filename for the recording
                    stream = VideoStream(fig.scene, framerate=fps)
                    rec_obs[] = !rec_obs[]
                    println("Recording started.")
    
                    @async while events(fig).window_open[] && rec_obs[]
                        recordframe!(stream)
                        sleep(1 / fps)
                    end
    
                elseif rec_obs[]
                    # save stream and stop recording
                    
                    opath = pwd() * "/CuArrays/outputs/" * "lenia/"
                    mkpath(opath)
                    filename = replace("mn_orbia_$(Dates.Time(Dates.now()))", ":" => "_") *".mp4"

                    rec_obs[] = !rec_obs[]

                    save(opath * filename, stream)
                    println("Recording stopped. Files saved at $(opath * filename).")
                end
            end
        end
        # Let the event reach other listeners
        return Consume(false)
    end
    return fig
end




#%%

simulate(M, fps=30)