using AbstractPlotting
using GLMakie
AbstractPlotting.inline!(false)

scene = Scene()
f(t, v, s) = (sin(v + t) * s, cos(v + t) * s)
time_node = Node(0.0)
p1 = scatter!(scene, lift(t-> f.(t, range(0, stop = 2pi, length = 50), 1), time_node))[end]
p2 = scatter!(scene, lift(t-> f.(t * 2.0, range(0, stop = 2pi, length = 50), 1.5), time_node))[end]
points = lift(p1[1], p2[1]) do pos1, pos2
    map((a, b)-> (a, b), pos1, pos2)
end
linesegments!(scene, points)
N = 150
record(scene, "output.mp4", range(0, stop = 10, length = N)) do i
    time_node[] = i
end
