using Random
using StatsBase

struct MorphicField{T<:Real, S<:Real}
    #types of morphic units
    #resonance rules
    ##como definir tipos dentro de uma estrutura? isso faz sentido?
    ##ou, então, como definir tipos associados a uma estrutura?
    ##acho que, pra isso, eu definiria o tipo MorphicField, e tipos de unidades morficas como subtipos
    ##daí um morphic field específico seria uma struct, usando certos tipos e relaçoes entre eles
    Dimensions::Tuple{Int64, Int64}
    Types::Array{T}
    States::Array{S}
    Field::Array{Tuple{T, S}, 2}
end

function MorphicField(dimensions::Tuple{Int64, Int64}, types::Array{T}, states::Array{S}) where {T<:Real, S<:Real}

    field = [(rand(types), rand(states)) for x in 1:dimensions[1], y in 1:dimensions[2]]
    
    return MorphicField(dimensions, types, states, field)

end

function get_states(mfield::MorphicField)

    xlen = mfield.Dimensions[1]
    ylen = mfield.Dimensions[2]
    states = [mfield.Field[x, y][2] for x in 1:xlen, y in 1:ylen]
    
    return states

end

function get_modvn_neighborhood(x, y, N, M) #get the four Von Neumann neighbors of a point in a lattice toroidally

    x_left = mod1(x-1, N)
    x_right = mod1(x+1, N)
    y_up = mod1(y+1, M)
    y_down = mod1(y-1, M)

    neighbors = [(x_left, y), (x_right, y), (x, y_up), (x, y_down)]

    return neighbors

end

function update_states!(mfield::MorphicField)
    
    I = CartesianIndices(mfield.Field)

    states = get_states(mfield)

    for i in I
    
        tup_i = Tuple(i)
        cell = mfield.Field[tup_i...]
        nbhood = get_modvn_neighborhood(tup_i..., mfield.Dimensions...)
        nbs_states = [states[nb...] for nb in nbhood]
        nbstates_mode = mode(nbs_states)
        mfield.Field[tup_i...] = (cell[1], nbstates_mode)

    end
    return states
end



mfield = MorphicField((100, 100), [0, 1, 2], [0, 1])
states = get_states(mfield)

mfield

using Plots


anim = @animate for i ∈ 1:50
    states = update_states!(mfield)
    heatmap!(states)
end
# %%
gif(anim, "mfield2_fps10.gif", fps = 10)
# heatmap(states)