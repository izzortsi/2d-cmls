# %%

using Random
using StatsBase
# %%

mutable struct MorphogeneticField{T<:Real, S<:Real}
    #types of morphic units
    #resonance rules
    ##como definir tipos dentro de uma estrutura? isso faz sentido?
    ##ou, então, como definir tipos associados a uma estrutura?
    ##acho que, pra isso, eu definiria o tipo MorphogeneticField, e tipos de unidades morficas como subtipos
    ##daí um morphic field específico seria uma struct, usando certos tipos e relaçoes entre eles
    Dimensions::Tuple{Int64, Int64}
    Types::Array{T}
    States::Array{S}
    CellsTypes::Array{T, 2}
    CellsStates::Array{S, 2}
end
# %%


function MorphogeneticField(dimensions::Tuple{Int64, Int64}, types::Array{T}, states::Array{S}) where {T<:Real, S<:Real}

    cellstypes = [rand(types) for x in 1:dimensions[1], y in 1:dimensions[2]]
    cellsstates = [rand(states) for x in 1:dimensions[1], y in 1:dimensions[2]]

    return MorphogeneticField(dimensions, types, states, cellstypes, cellsstates)

end

# %%


function get_modvn_neighborhood(x, y, N, M) #get the four Von Neumann neighbors of a point in a lattice toroidally

    x_left = mod1(x-1, N)
    x_right = mod1(x+1, N)
    y_up = mod1(y+1, M)
    y_down = mod1(y-1, M)

    neighbors = [(x_left, y), (x_right, y), (x, y_up), (x, y_down)]

    return neighbors

end


# %%

function update_states_vn!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    #states = deepcopy(mfield.CellsStates)

    for i in I
    
        tup_i = Tuple(i)
        nbhood = get_modvn_neighborhood(tup_i..., mfield.Dimensions...)
        nbs_types = [types[nb...] for nb in nbhood]
        nbstypes_mode = mode(nbs_types)
        mfield.CellsTypes[tup_i...] = nbstypes_mode

    end

end
# %%

function update_types_vn!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    states = deepcopy(mfield.CellsStates)

    for i in I
        
        tup_i = Tuple(i)

        if states[tup_i...] == 0
            tup_i = Tuple(i)
            nbhood = get_modvn_neighborhood(tup_i..., mfield.Dimensions...)
            nbs_types = [types[nb...] for nb in nbhood]
            nbstypes_mode = mode(nbs_types)
            mfield.CellsTypes[tup_i...] = nbstypes_mode
            mfield.CellsStates[tup_i...] = 1
        else
            mfield.CellsStates[tup_i...] = 0
        end

    end

end
# %%

function update_types_t!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    states = deepcopy(mfield.CellsStates)

    xdim = mfield.Dimensions[1] 
    ydim = mfield.Dimensions[2]

    for i in I
        
        tup_i = Tuple(i)

        if states[tup_i...] == 0

            x_0, y_0 = tup_i
            nbhood = []

            for i in x_0-1:x_0+1
                for j in y_0-1:y_0+1
                    if i != j
                        push!(nbhood, (mod1(i, xdim), mod1(j, ydim)))
                    end
                end
            end

            nbs_types = [types[nb...] for nb in nbhood]
            nbstypes_mode = mode(nbs_types)
            mfield.CellsTypes[tup_i...] = nbstypes_mode
            mfield.CellsStates[tup_i...] = 1
        else
            mfield.CellsStates[tup_i...] = 0
        end

    end

end
# %%

function update_types!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    states = deepcopy(mfield.CellsStates)

    for i in I
        
        tup_i = Tuple(i)

        if states[tup_i...] == 0

            x_0, y_0 = tup_i(i)
            nbhood = []

            for i in x_0-1:x_0+1
                for j in y_0-1:y_0+1
                    if i != j && i*j != 0 && i <= mfield.Dimensions[1] && j <= mfield.Dimensions[2]
                        push!(nbhood, (i, j))
                    end
                end
            end

            nbs_types = [types[nb...] for nb in nbhood]
            nbstypes_mode = mode(nbs_types)
            mfield.CellsTypes[tup_i...] = nbstypes_mode
            mfield.CellsStates[tup_i...] = 1
        else
            mfield.CellsStates[tup_i...] = 0
        end

    end

end
# %%

function update_states!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    #states = deepcopy(mfield.CellsStates)
    
    for i in I
        tup_i = Tuple(i)
        x_0, y_0 = tup_i
        nbhood = []

        for i in x_0-1:x_0+1
            for j in y_0-1:y_0+1
                if i != j && i*j != 0 && i <= mfield.Dimensions[1] && j <= mfield.Dimensions[2]
                    push!(nbhood, (i, j))
                end
            end
        end
        nbs_types = [types[nb...] for nb in nbhood]
        nbstypes_mode = mode(nbs_types)
        mfield.CellsTypes[tup_i...] = nbstypes_mode

    end

end

# %%

function update_states_t!(mfield::MorphogeneticField)
    
    I = CartesianIndices(mfield.CellsTypes)
    
    types = deepcopy(mfield.CellsTypes)
    #states = deepcopy(mfield.CellsStates)
    
    xdim = mfield.Dimensions[1] 
    ydim = mfield.Dimensions[2]

    for i in I
        tup_i = Tuple(i)
        x_0, y_0 = tup_i
        nbhood = []


        for i in x_0-1:x_0+1
            for j in y_0-1:y_0+1
                if i != j
                    push!(nbhood, (mod1(i, xdim), mod1(j, ydim)))
                end
            end
        end
        
        nbs_types = [types[nb...] for nb in nbhood]
        nbstypes_mode = mode(nbs_types)
        mfield.CellsTypes[tup_i...] = nbstypes_mode

    end

end
# %%

mfield = MorphogeneticField((200, 200), [0, 1, 2], [0, 1])

using Plots
#@time update_states!(mfield)
#@time update_types!(mfield)
#heatmap(mfield.CellsTypes)
#
#@gif for i in 1:50
#    update_states!(mfield)
#    heatmap!(mfield.CellsTypes)
#end


anim = @animate for i ∈ 1:50
    update_states!(mfield)
    Plots.heatmap!(mfield.CellsTypes)
end
# %%
gif(anim, "mfield_fps10.gif", fps = 10)

# %%
# run(`ffmpeg -v 0 -i $(anim.dir)/%06d.png -vf palettegen -y palette.gif`)
# # %%
# fps = 6
# loop = 0
# fn = "mfield.gif"
# run(`ffmpeg -v 0 -framerate $fps -loop $loop -i $(anim.dir)/%06d.png -i palette.gif -lavfi paletteuse -y $fn`)
# # %%



