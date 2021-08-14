using CuArrays
using CUDAnative
using CUDAdrv

include("getnbs.jl")

function get_unstable(N, X, U)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:N, j in indy:stridey:N

        inval1 = X[i, j]
        c=1
        if inval1 >= 4
            U[c] = (i, j)
            c += 1
        end
    end

    return nothing

end

function update_unstable(N, X, U)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:N, j in indy:stridey:N
        unstind = U[i]
        if unstind != (0,0)
            X[unstind...] -= 4
            #nbs = get_vn_neighborhood(unstind..., N, N) #n tá definida pra usar CuArray e nao posso passar array normal pro kernel da gpu
            #for nb in nbs
            #    if
        end
    end
end

N = 500
(1, 1) == (1, 0)
X = rand(0:4, N, N) |> CuArray
U = CuArray{Tuple{Int64, Int64}, 1}(undef, 500)
#U = CuArray{Int64, 1}(undef, 500)

dev = CuDevice(0)
max_threads = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
max_threads_per_dim = sqrt(max_threads)/2 |> Int

numblocks = ceil(Int, N/max_threads_per_dim)

threads = N÷numblocks



@cuda blocks=(numblocks, numblocks) threads=(threads, threads) get_unstable(N, X, U)
U
@cuda blocks=(numblocks, numblocks) threads=(threads, threads) update_unstable(N, X, U)
