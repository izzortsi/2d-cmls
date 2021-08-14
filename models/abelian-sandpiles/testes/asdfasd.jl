N, M = 10, 10

X = rand(N,M)

function foo(X, x, y)

    X[x, y] += 1

    if X[x, y] >= 5

        X[x, y] -= 5

    end

    return X

end

function get_from_index(X, xy)
    return X[xy[1], xy[2]]
end

function bar(X, x, y, N, M)

    X[x, y] += 1

    if X[x, y] >= 5

        X[x, y] -= 5

        nbs = get_vn_neighborhood(x, y, N, M)

        for nb in nbs
            bar(X, nb..., N, M)
        end
    end

    return X

end

x, y = 3, 2
X[x, y]

nbs = get_vn_neighborhood(x, y, N, M)

nbsvals = (xy -> get_from_index(X, xy)).(nbs)

X[x, y]

bar(X, x, y, N, M)

nbsvals = (xy -> get_from_index(X, xy)).(nbs)

X[x, y]

oX = copy(X)

nX = bar(X, x, y, N, M)

X == nX
X == oX
