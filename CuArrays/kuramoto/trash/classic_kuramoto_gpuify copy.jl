
using KernelAbstractions, CUDA, CUDAKernels
# %%

a = CUDA.rand(100)


# %%

@kernel function kernel(a)
	I = @index(Global)
	a[I] = a[I] + mapreduce(sin, +,  a)
end
# %%

f = kernel(CUDADevice(), 256)
a = CUDA.rand(100)
# %%

event = f(a, ndrange=size(a))

# %%
wait(event)

# %%

@kernel function kernel4(a)
	I = @index(Global)
    map(x->2*x, a)
end
# %%

f = kernel4(CUDADevice(), 256)

# %%

event = f(a, ndrange=size(a))

# %%
wait(event)