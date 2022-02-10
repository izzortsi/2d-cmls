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
  "cells" => cu([0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;
                0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
                1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
                1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
                1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
                1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.;
                1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
                1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
                1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
                0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
                0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;]))
            
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
DHMatrix = Union{Matrix{Float64}, CuArray{Float32, 2}, CuArray{Float64, 2}}
# %%

mutable struct MNCA
    A::DHMatrix
    R::Int64
    μ::Float64
    σ::Float64
    β::Array{Float64}
    dt::Float64
    
    bin::Float64
    r::Float64
    k::Float64
    e::Float64

    update_thresholds::Dict{String, Float64}

    S::Matrix{Float64}
    K::Vector{DHMatrix}
    K_fft::Vector{Matrix{ComplexF64}}
    
    calc_kernels::Function
    δ::Vector{Function}
    Φ::Vector{Function}
    update!::Function
    populate!::Function

    U::DHMatrix
    G::DHMatrix

    function MNCA(A, R, μ, σ, β, dt, bin, r, k, e, update_thresholds)

        function calc_kernel(M::MNCA)
            M.S = D(X, Y, R)
            M.K = Vector{DHMatrix}(undef, 0)

            K1= cu([1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;
                    1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.;])
            push!(M.K, K1)
            K2 = cu([0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;
                     0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
                     1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
                     1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
                     1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
                     1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.;
                     1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
                     1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
                     1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
                     0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
                     0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;])
            push!(M.K, K2)
        end

        M = new(A, R, μ, σ, β, dt, bin, r, k, e, update_thresholds);
        calc_kernel(M)

        M.U = CUDA.similar(A)
        M.G = CUDA.similar(A)
        M.δ = []
        push!(M.δ, x -> x/121)
        push!(M.δ, x -> x/64)
        M.calc_kernels = calc_kernel

        return M
    end
end
#%%
# K2 = [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;
# 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
# 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
# 1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
# 1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
# 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.;
# 1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 1.;
# 1. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1.;
# 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1.;
# 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.;
# 0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.;]
#%%
# heatmap(K2)


#%%
A = zeros(SIZE, SIZE) |> cu

bin=Float32(0.93) 
r=Float32(0.0) 
k=Float32(0.893)
e=Float32(0.1752)
update_thresholds = Dict{String, Float32}(["1"=>33.0, "2"=> 45.0, "3"=> 57.0]) 
#A = cu(bitrand(SIZE, SIZE)) * Float32(1.0)
M = MNCA(A, 1, m, s, b, 1/5, bin, r, k, e, update_thresholds)

#%%

#%%

function continuous_update(M::MNCA)
    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ[1].(M.U) .* M.A)
    function aux_update(U)
        u1 = (U .<= M.update_thresholds[1]) * M.r 
        u2 = (M.update_thresholds[1] .< U .<= M.update_thresholds[2]) * M.k
        u3 = (M.update_thresholds[2] .< U .<= M.update_thresholds[3]) .* M.A
        u4 = (U .> M.update_thresholds[3]) * M.e
        return u1 + u2 + u3 + u4
    end
    M.A .= clamp.(aux_update(M.U), 0, 1)
end
# %%

function update1(M::MNCA)
    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ[1].(M.U) .* M.A)
    function aux_update(U)
        u1 = (U .<= 33) * 0 
        u2 = (34 .<= U .<= 45) * 1
        u3 = (46 .<= U .<= 57) .* M.A
        u4 = (U .>= 58) * 0
        return u1 + u2 + u3 + u4
    end
    M.A .= aux_update(M.U) 
end
function update2(M::MNCA)
    conv(M.A, M.K[2], M.U)
    M.G .= (M.δ[2].(M.U) .* M.A)
    function aux_update(U)
        u1 = (U .<= 17) * 0
        u2 = (18 .<= U .<= 22) * 1
        u3 = (23 .<= U .<= 29) .* M.A
        u4 = (U .>= 30) * 0
        return u1 + u2 + u3 + u4 #clamp.(M.A + M.G, 0, 1)
    end
    M.A .= clamp.(M.A + aux_update(M.U) .÷1, 0, 1)
end

function continuous_update(M::MNCA)
    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ[1].(M.U) .* M.A)
    function aux_update(U)
        u1 = (U .<= 33) * M.r 
        u2 = (33 .< U .<= 45) * M.k
        u3 = (45 .< U .<= 57) .* M.A
        u4 = (U .> 57) * M.e
        return u1 + u2 + u3 + u4
    end
    M.A .= clamp.(aux_update(M.U), 0, 1)
end

function spiking_update(M::MNCA)
    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ[1].(M.U) .* M.A)
    function aux_update(U)
        u1 = (U .<= 33) * 0.1 
        u2 = (34 .<= U .<= 45) * 0.8
        u3 = (46 .<= U .<= 57) .* M.A
        u4 = (U .>= 58) * 0.1
        return u1 + u2 + u3 + u4
    end

    state = aux_update(M.U)
    S = (state .>= M.bin) .* state # the spiking neurons
    nS = (state .< M.bin) .* state # the complimentary matrix
    spike = nS .+ (M.r .* S)
    #println(typeof(spike))
    conv(spike, M.K[1], M.U) 
    M.U = M.U ./ Float32(9)
    #conv(n, spike, ckern, convolved, kdim) # the spiking neuron have a 1.3fold greater influence over its neighbors
    M.A = M.e .* (nS .+ (M.k .* S)) .+ (Float32(1) - M.e) .* M.U  # (nS + k*S) is the initial state but with the spiking neurons' states updated; (1-e)*conv is the influence the neighbors had over the neuron
    #e*(nS + k*S) + (1-e)*conv

end

function update!(M::MNCA)
    for ϕ in M.Φ
        ϕ(M)
    end
end

#%%
M.Φ = Vector{Function}(undef, 0)
#%%

push!(M.Φ, continuous_update)
# push!(M.Φ, update1)
#%%


#push!(M.Φ, update2)
#%%


M.update! = update!
#%%
function populate(W, creature_cells, num_creatures; is_random = false)
    cx, cy = size(creature_cells)
    to_populate = rand(cx:SIZE-cx, (num_creatures, 2))
    for row in eachrow(to_populate)
        x, y = row
        if is_random
            W[(x-cx ÷ 2):(x+cx ÷ 2), (y-cx ÷ 2):(y+cx ÷ 2)] = cu(bitrand(cx, cy) .|> Float32)
        else
            W[(x-cx ÷ 2):(x+cx ÷ 2), (y-cx ÷ 2):(y+cx ÷ 2)] = creature_cells
        end
    end
end
# %%
#creature["cells"] = cu(creature["cells"])
# %%

populate(M.A, creature["cells"], 50, is_random=true)
#%%
M.populate! = () -> populate(M.A, creature["cells"], 50)
#%%


function panels(M::MNCA)

    conv(M.A, M.K[1], M.U)
    M.G .= (M.δ[1].(M.U) .* M.A)

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
# fig#
#%%
M.update!(M)

#%%
include("sim_man_gol_mutation.jl")


#%%

simulate(M, fps=30)
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
#run(M)
#%%

function record_run(fig, M; nframes = 600, fps=36)
    opath = pwd() * "/CuArrays/outputs/" * "lenia/"
    mkpath(opath)
    GLMakie.record(fig, opath * "mnca_gol.mp4", 1:nframes; framerate = fps) do i
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
