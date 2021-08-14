function get_modvn_neighborhood(x, y, N, M)
    x_left = mod1(x-1, N)
    x_right = mod1(x+1, N)
    y_up = mod1(y+1, M)
    y_down = mod1(y-1, M)
    neighbors = [(x_left, y), (x_right, y), (x, y_up), (x, y_down)]
    return neighbors
end

function get_vn_neighborhood(x, y, N, M)

    if x == 1
        if y == 1
            nbs = [(x+1, y), (x, y+1)]
            return nbs
        elseif y == M
            nbs = [(x+1, y), (x, y-1)]
            return nbs
        else
            nbs = [(x+1, y), (x, y-1), (x, y+1)]
            return nbs
        end

    elseif x == N
        if y == 1
            nbs = [(x-1, y), (x, y+1)]
            return nbs
        elseif y == M
            nbs = [(x-1, y), (x, y-1)]
            return nbs
        else
            nbs = [(x-1, y), (x, y-1), (x, y+1)]
            return nbs
        end

    else
        if y == 1
            nbs = [(x-1, y), (x+1, y), (x, y+1)]
            return nbs
        elseif y == M
            nbs = [(x+1, y), (x-1, y), (x, y-1)]
            return nbs
        else
            nbs = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            return nbs
        end
    end
end
