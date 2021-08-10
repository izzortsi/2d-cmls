module Convolution

using CUDA
export setup_convolution, setup_extended_convolution, loop_conv

function extended_convolution(n, A, ckern, kfuns, outs, kdim)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n
        ran = kdim ÷ 2
        # indices for the neighbors
        i_minus =  mod1(i - ran, n)
        i_plus = mod1(i + ran, n)
        j_minus = mod1(j - ran, n)
        j_plus = mod1(j + ran, n)

        for s in i_minus:i_plus, t in j_minus:j_plus

            s1 = s - i + (ran + 1)
            t1 = t - j + (ran + 1)
            #outs[i, j] += (A[s, t] * ckern[s1, t1])/sqrt(((i-s1)^2 + (j-s2)^2))
            outs[i, j] += kfuns[s1, t1]((A[s, t]), ckern[s1, t1])
        end

    end

    return nothing
end


function convolution(A, ckern, outs)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y
    n, = CUDA.size(A)
    kdim, = CUDA.size(ckern)
    for i in indx:stridex:n, j in indy:stridey:n
        ran = kdim ÷ 2
        # indices for the neighbors
        i_minus =  mod1(i - ran, n)
        i_plus = mod1(i + ran, n)
        j_minus = mod1(j - ran, n)
        j_plus = mod1(j + ran, n)

        for s in i_minus:i_plus, t in j_minus:j_plus
            s1 = s - i + (ran + 1)
            t1 = t - j + (ran + 1)
            outs[i, j] += (A[s, t] * ckern[s1, t1])
        end

    end

    return nothing
end

"""
n is the size of the square n by n matrix that will be worked on
"""
function setup_kernel(n::Int64)
    dev = CuDevice(0)
    
    max_threads = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_threads_per_dim = sqrt(max_threads) / 2 |> Int
    
    numblocks = ceil(Int, n / max_threads_per_dim)
    threads = n ÷ numblocks
    
    return numblocks, threads
end
"""
returns a function `conv(n, A, ckern, outs, kdim)`, that performs a convolution of `ckern` over `A`;\\
    `A` is a `n` by `n` `CuArray` and `outs` must be similar to `A`;\\
    `ckern` is a `kdim` by `kdim` `CuArray`;
"""
function setup_extended_convolution(n::Int64)
    numblocks, threads = setup_kernel(n)
    econv(n, A, ckern, kfuns, outs, kdim) = @cuda blocks = (numblocks, numblocks) threads = (threads, threads) extended_convolution(n, A, ckern, kfuns, outs, kdim)
    econv
end

function setup_convolution(n::Int64)
    numblocks, threads = setup_kernel(n)
    conv(A, ckern, outs) = @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(A, ckern, outs)
    conv
end

function loop_conv(niter, A, ckern, kdim)
    for i in 1:niter
        outs = CUDA.zeros(n, n)
        # @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, ckern, outs, kdim)
        conv(n, A, ckern, outs, kdim)
        A = outs
    end
    return A
end

end