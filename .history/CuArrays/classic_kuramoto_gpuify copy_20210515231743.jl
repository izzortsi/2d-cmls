
using KernelAbstractions, CUDA, CUDAKernels
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

@kernel function kernel1(a)
	I = @index(Global)
	a[I] = a[I] + mapreduce(sin, +,  a)
end
# %%

f = kernel1(CUDADevice(), 256)
a = CUDA.rand(100)
# %%

event = f(a, ndrange=size(a))

# %%
wait(event)