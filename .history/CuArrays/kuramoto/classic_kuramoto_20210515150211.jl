
using Random
using DifferentialEquations
# %%
Random.seed!(0)

n = 50  # BE WARY OF n^2 COUPLED ODEs TO BE SOLVED
N = n ^ 2
K = 1
ω = rand(N) * 2 * π
dθ = rand(N) * 2 * π
# %%


function kuramoto!(t, dθ)
    
    for i, dθ in enumerate(dθ)
        dθ[i] = ω[i] - ((K / N) * np.sum(np.sin(θ_i - θ)))
    #        print((K / N) * np.sum(np.sin(θ_i - θ)), dθ[i])
    return dθ


integrator = Integrators["ForwardEuler"]()
# %%

ts, θs = integrator.solve(F, 0, 15, θ, 0.1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%

fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
im = ax.imshow(θs[0])  # , vmin=0, vmax=2 * np.pi)
# fig.colorbar(im)


def init_plot():
    return ax.images


def update(num, θs, ax):
    ax.images[0].set_data(θs[num])
    return ax.images


anim = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_TS,
    fargs=(θs, ax),
    interval=5,
    blit=True,
)
# %%
file_path = os.path.join(KURAMOTO_OUTS, f"classic_kuramoto_K={K:.4f}.mp4")
anim.save(file_path, fps=6)
