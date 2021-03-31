function convolution2(n, A, filter, outs)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n

        # indices for the neighbors
        j_minus = mod1(j - 1, n)
        j_plus = mod1(j + 1, n)
        i_minus =  mod1(i - 1, n)
        i_plus = mod1(i + 1, n)

        # neighbors
        sup = A[i, j_minus]
        sleft = A[i_minus, j]
        sright = A[i_plus, j]
        sdown = A[i, j_plus]
        supleft = A[i_minus, j_minus]
        supright = A[i_plus, j_minus]
        sdownleft = A[i_minus, j_plus]
        sdownright = A[i_plus, j_plus]

        # coefficients from the corresponding coupling scheme
        cs_ij = filter
        cf_up = cs_ij[2, 1]
        cf_left = cs_ij[1, 2]
        cf_right = cs_ij[3, 2]
        cf_down = cs_ij[2, 3]
        cf_upleft = cs_ij[1, 1]
        cf_downleft = cs_ij[1, 3]
        cf_upright = cs_ij[3, 1]
        cf_downright = cs_ij[3, 3]
        cf_center = cs_ij[2, 2]


        # calculating the convolution
        X = A[i,j]

        N = (
        sup * cf_up +
        sleft * cf_left + 
        sright * cf_right + 
        sdown * cf_down + 
        supleft * cf_upleft + 
        supright * cf_upright + 
        sdownleft * cf_downleft + 
        sdownright * cf_downright +
        X * cf_center
        )
        
        outs[i, j] = N
    end

    return nothing
end

function loop_conv2(niter, A, filter)
    for i in 1:niter
        outs = CUDA.zeros(n, n)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution2(n, A, filter, outs)
        A = outs
    end
    return A
end

# generalized version
function convolution(n, A, filter, outs, kdim)

    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stridex = blockDim().x * gridDim().x

    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stridey = blockDim().y * gridDim().y

    for i in indx:stridex:n, j in indy:stridey:n
        ran = kdim ÷ 2
        # indices for the neighbors
        j_minus = mod1(j - ran, n)
        j_plus = mod1(j + ran, n)
        i_minus =  mod1(i - ran, n)
        i_plus = mod1(i + ran, n)

        for s in j_minus:j_plus, t in i_minus:i_plus
            s1 = s - j + (ran + 1)
            t1 = t - i + (ran + 1)
            outs[i, j] += (A[s, t] * filter[s1, t1])
        end

    end

    return nothing
end

function loop_conv(niter, A, filter, kdim)
    for i in 1:niter
        outs = CUDA.zeros(n, n)
        @cuda blocks = (numblocks, numblocks) threads = (threads, threads) convolution(n, A, filter, outs, kdim)
        A = outs
    end
    return A
end

