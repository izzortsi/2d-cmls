# %%

using Agents, Colors, Dates, DrWatson, JSON, Observables, Random, ImageCore
using GLMakie
using Dates

# using ImageCore
# using Makie, Makie.AbstractPlotting, Makie.AbstractPlotting.MakieLayout
# import Base.to_color
# %%
# function to_color(p::Particle)
#     return p.activation
# end
    

# %%

import Statistics: mean

import Base.length
function Base.length(ids::Base.Iterators.Filter)
    return length([id for id in ids])
end
##
include("model_functions.jl")
include("visualization_functions.jl")


cmap = colormap("RdBu", mid=0.5)


mdata = [avg_nbsize, avg_activation]
mlabels = ["average num neighbors", "average acivation"]

params_intervals = Dict(
    :iradius => 0.1:0.1:8.0,
    :cohere_factor => 0.1:0.01:0.6, 
    :separation => 0.1:0.1:8.0, 
    :separate_factor => 0.1:0.01:0.6, 
    :match_factor => 0.005:0.001:0.1
    )



params = Dict(
    :n_particles => 5000, 
    :speed => 1.3, 
    :separation => 0.7, 
    :iradius => 1.4, 
    :cohere_factor => 0.23, 
    :separate_factor => 0.15, 
    :match_factor => 0.03,
    :min_nb => 0., 
    :max_nb => 1.
    )

params = Dict(
    :n_particles => 1000, 
    :speed => 1.3, 
    :separation => 0.8, 
    :iradius => 1.7, 
    :cohere_factor => 0.25, 
    :separate_factor => 0.18, 
    :match_factor => 0.084,
    :min_nb => 0., 
    :max_nb => 1.
    )


##
function makie_abm(model, ac="#765db4", as=1, am=:circle, scheduler=model.scheduler; initial_params=model.properties, params_intervals=nothing, resolution=(1280, 720), fps=24, savepath="abm_recording.mp4")
    
    # TODO salvar recording junto com os parametros
    # TODO 
    # date = Date(now())
    date = Date(now())
    superfolder = "simulations/simulations_$date"
    timestamp_format = "HH"
    hour = Dates.format(now(), timestamp_format)
    prepath = "$superfolder/$(hour)h/"

    ids = scheduler(model)
    fig = Figure()
    ax1 = Axis(fig[1, 1])
    fig[1, 2] = buttongrid = GridLayout(tellwidth = false)
    # fig[3, 2] = slidergrid = GridLayout(tellwidth = false)
    # scene, layout = layoutscene(resolution=resolution)
    # model-related observables
    modelobs = Observable(model)
    colors = ac isa Function ? Observable(to_color.([ac(model[i]) for i in ids])) : to_color(ac)
    sizes  = as isa Function ? Observable([as(model[i]) for i in ids]) : as
    markers = am isa Function ? Observable([am(model[i]) for i in ids]) : am
    pos = Observable([model[i].pos for i in ids])
    # criar observers pras propriedades que tão sujeitas a randomização (com os valores iniciais)
    
    props_obs = Dict()
    props_labels = []
    
    if params_intervals !== nothing
        for (key, val) in params_intervals
            value = modelobs[].properties[key]
            props_obs[key] = Observable(value)
            plabel = lift(x -> "$(String(key)): $(x[])", props_obs[key])

            push!(props_labels, plabel)
        end
    end


    run_obs = Observable{Bool}(false)
    rec_obs = Observable{Bool}(false)
    
    buttonlabels = [props_labels ; [lift(x -> x ? "RUNNING" : "HALTED", run_obs), lift(x -> x ? "RECORDING" : "STOPPED", rec_obs)]]  
    buttons = buttongrid[1:length(buttonlabels), 1] = [Button(fig, label = l) for l in buttonlabels]
    # running_label = LText(scene, lift(x -> x ? "RUNNING" : "HALTED", run_obs))
    # recording_label = LText(scene, lift(x -> x ? "RECORDING" : "STOPPED", rec_obs))

    # ax1 = fig[1, 1] = LAxis(scene, tellheight=true, tellwidht=true)
    infos = GridLayout(tellheight=false, tellwidth=true)
    
    infos[1:length(buttonlabels), 1] = buttons

    fig[1, 2] = infos

    scatter!(ax1, pos;
    color=colors, markersize=sizes, marker=markers, strokewidth=0.0)

    stream = VideoStream(fig.scene, framerate=fps)

    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
           if event.key == Keyboard.s
                run_obs[] = !run_obs[]
                run_obs[] ? println("Simulation running. $(run_obs[])") : println("Simulation stopped.")

                @async while events(fig).window_open[] && run_obs[]                 # update observables in fig.scene
                    model = modelobs[]
                    Agents.step!(model, agent_step!, model_step!)
                    ids = scheduler(model)
                    update_abm_plot!(pos, colors, sizes, markers, model, ids, ac, as, am)
                    
                    if !isopen(fig.scene)
                        if !rec_obs[]
                            break
                        else
                        
                            new_filepath, tstamp = namefile(prepath, savepath)
                            path = mkpath("$(@__DIR__)/$(prepath)sim$tstamp")

                            open("$path/params$tstamp.json", "w") do f 
                                write(f, JSON.json(modelobs[].properties))
                            end

                            save(new_filepath, stream)
                            println("Window closed while recording. Recording stopped. Files saved at $(prepath)sim$tstamp/.")
                            break
                        end
                    end

                    sleep(1 / fps)
                end
            end

        elseif event.key == Keyboard.r
            if !rec_obs[]
                # start recording
                # start a new stream and set a new filename for the recording
                # stream = VideoStream(fig.scene, framerate=fps)
                rec_obs[] = !rec_obs[]
                println("Recording started.")

                @async while events(fig).window_open[] && rec_obs[]
                    recordframe!(stream)
                    sleep(1 / fps)
                end

            elseif rec_obs[]
                # save stream and stop recording
                
                path = pwd()*"\\"
                filename = split(string(now()), ":")[1]

                open("$path/params_$filename.json", "w") do f 
                    write(f, JSON.json(modelobs[].properties))
                end

                save(path, stream)
                println("Recording stopped. Files saved at $(prepath)sim$tstamp/.")
                
                rec_obs[] = !rec_obs[]
            end

        elseif event.key == Keyboard.p
            if params_intervals !== nothing
                for (key, val) in params_intervals
                    new_val = rand(val)
                    modelobs[].properties[key] = new_val
                    props_obs[key][] = new_val
                end
            else
                println("Parameters intervals are needed for randomization.")
            end

        elseif event.key == Keyboard.v
            # save parameters
            new_filepath, tstamp = namefile(prepath, savepath)
            path = mkpath("$(@__DIR__)/$(prepath)params$tstamp")
            open("$path/params$tstamp.json", "w") do f 
                write(f, JSON.json(modelobs[].properties))
            end
            println("Parameters saved at file $(prepath)params$tstamp/params$tstamp.json")
        end

    end
    
    return fig, ids, colors, sizes, markers, pos, ac, as, am
end


function namefile(prepath, savepath)
    timestamp_format = "HH:MM:SS"
    tstamp = Dates.format(now(), timestamp_format)
    dot_idx = findlast(isequal('.'), savepath)
    new_filepath = prepath * "sim$tstamp/" * savepath[1:dot_idx - 1] * "$tstamp" * savepath[dot_idx:end]
    return new_filepath, tstamp
end

function update_abm_plot!(pos, colors, sizes, markers, model, ids, ac, as, am)
    
    if Agents.nagents(model) == 0
        @warn "The model has no agents, we can't plot anymore!"
        error("The model has no agents, we can't plot anymore!")
    end
    
    pos[] = [model[i].pos for i in ids]
    
    if ac isa Function; colors[] = to_color.([ac(model[i]) for i in ids]); end
    if as isa Function; sizes[] = [as(model[i]) for i in ids]; end
    if am isa Function; markers[] = [am(model[i]) for i in ids]; end
end


# pwd()*"\\"*split(string(now()), ":")[1]
# pwd()
# # %%
# out_path = pwd() * now()
fps = 20

model = initialize_model(dims=(50, 50), params=params)
e = model.space.extent
# %%


##
function scatter_abm(model, ac="#765db4", as=1, am=:circle, scheduler=model.scheduler, resolution=(1280, 720))

    ids = scheduler(model)
    colors = ac isa Function ? Observable(to_color.([ac(model[i]) for i in ids])) : to_color(ac)
    sizes  = as isa Function ? Observable([as(model[i]) for i in ids]) : as
    markers = am isa Function ? Observable([am(model[i]) for i in ids]) : am
    pos = Observable([model[i].pos for i in ids])

    scene = scatter(pos;
        color=colors, markersize=sizes, marker=markers, strokewidth=0.0, resolution=resolution)

    # display(scene)

    return scene, ids, colors, sizes, markers, pos, ac, as, am
end

function record_simulation(model::AgentBasedModel, interval::AbstractRange; framerate=30, ac="#765db4", as=1, am=:circle, scheduler=model.scheduler, resolution=(1280, 720))

    scene, ids, colors, sizes, markers, pos, ac, as, am = scatter_abm(model, ac, as, am, scheduler, resolution)

    record(scene, "abm_animation.mp4", interval; framerate=framerate) do t
        Agents.step!(model, agent_step!, model_step!, 10)
        update_abm_plot!(pos, colors, sizes, markers, model, ids, ac, as, am)
    end
end
##
# fig
record_simulation(model, 1:300; framerate=60)

# fig, p = makie_abm(model, ac, 0.6, params_intervals=params_intervals; fps=fps, resolution=(1200, 780))
