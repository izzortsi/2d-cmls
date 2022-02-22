# %%

using WGLMakie
scatter(rand(4))

# %%
function dom_handler(session, request)
    return DOM.div(
        DOM.h1("Some Makie Plots:"),
        meshscatter(1:4, color=1:4),
        meshscatter(1:4, color=rand(RGBAf, 4)),
        meshscatter(1:4, color=rand(RGBf, 4)),
        meshscatter(1:4, color=:red),
        meshscatter(rand(Point3f, 10), color=rand(RGBf, 10)),
        meshscatter(rand(Point3f, 10), marker=Pyramid(Point3f(0), 1f0, 1f0)),
    )
end
isdefined(Main, :app) && close(app)
app = JSServe.Server(dom_handler, "127.0.0.1", 8082)
# %%
