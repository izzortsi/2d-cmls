using Observables, GLMakie
#%%


function simulate(model, ac="#765db4", as=1, am=:circle, scheduler=model.scheduler; initial_params=model.properties, params_intervals=nothing, resolution=(1280, 720), fps=24, savepath="abm_recording.mp4")
    
    # TODO salvar recording junto com os parametros
    # TODO 

    date = Date(now())
    superfolder = "simulations/simulations_$date"
    timestamp_format = "HH"
    hour = Dates.format(now(), timestamp_format)
    prepath = "$superfolder/$(hour)h/"

    
    
    ids = scheduler(model)
    init_params = deepcopy(initial_params)

    scene, layout = layoutscene(resolution=resolution)

    # model-related observables
    modelobs = Observable(model)
    colors = ac isa Function ? Observable(to_color.([ac(model[i]) for i in ids])) : to_color(ac)
    sizes  = as isa Function ? Observable([as(model[i]) for i in ids]) : as
    markers = am isa Function ? Observable([am(model[i]) for i in ids]) : am
    pos = Observable([model[i].pos for i in ids])

    # criar observables pras propriedades que tão sujeitas a randomização (com os valores iniciais)
    props_obs = Dict()
    props_labels = []
    if params_intervals !== nothing
        for (key, val) in params_intervals
            value = modelobs[].properties[key]
            props_obs[key] = Observable(value)
            plabel = Label(scene, lift(x -> "$(String(key)): $(x[])", props_obs[key]))

            push!(props_labels, plabel)
        end
    end

    # interaction control observables

    run_obs = Observable{Bool}(false)
    rec_obs = Observable{Bool}(false)
    
    running_label = Label(scene, lift(x -> x ? "RUNNING" : "HALTED", run_obs))
    recording_label = Label(scene, lift(x -> x ? "RECORDING" : "STOPPED", rec_obs))

    ax1 = layout[1, 1] = Axis(scene, tellheight=true, tellwidht=true)
    infos = GridLayout(tellheight=false, tellwidth=true)
    infos[1, 1] = running_label
    infos[2, 1] = recording_label
    
    for (i, plabel) in enumerate(props_labels)
        infos[i + 2, 1] = plabel
    end
    
    layout[1, 2] = infos

    scatter!(ax1, pos;
    color=colors, markersize=sizes, marker=markers, strokewidth=0.0)

    stream = VideoStream(scene, framerate=fps)

    on(scene.events.keyboardbuttons) do button

        if button == Set(AbstractPlotting.Keyboard.Button[AbstractPlotting.Keyboard.s]) 

            run_obs[] = !run_obs[]
            run_obs[] ? println("Simulation running. $(run_obs[])") : println("Simulation stopped.")

            @async while run_obs[]
                # update observables in scene
                model = modelobs[]
                Agents.step!(model, agent_step!, model_step!, 1)
                ids = scheduler(model)
                update_abm_plot!(pos, colors, sizes, markers, model, ids, ac, as, am)
                
                if !isopen(scene)
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
            # end
        
        elseif button == Set(AbstractPlotting.Keyboard.Button[AbstractPlotting.Keyboard.r]) 

            if !rec_obs[]
                # start recording
                # start a new stream and set a new filename for the recording
                stream = VideoStream(scene, framerate=fps)
                rec_obs[] = !rec_obs[]
                println("Recording started.")

                @async while rec_obs[]
                    recordframe!(stream)
                    sleep(1 / fps)
                end

            elseif rec_obs[]
                # save stream and stop recording
                rec_obs[] = !rec_obs[]
                new_filepath, tstamp = namefile(prepath, savepath)
                path = mkpath("$(@__DIR__)/$(prepath)sim$tstamp")

                open("$path/params$tstamp.json", "w") do f 
                    write(f, JSON.json(modelobs[].properties))
                end

                save(new_filepath, stream)
                println("Recording stopped. Files saved at $(prepath)sim$tstamp/.")
            end

        elseif button == Set(AbstractPlotting.Keyboard.Button[AbstractPlotting.Keyboard.p])
            #randomize parameters within intervals given
            if params_intervals !== nothing
                for (key, val) in params_intervals
                    new_val = rand(val)
                    modelobs[].properties[key] = new_val
                    props_obs[key][] = new_val #each element in props_obs is an observable that tracks a property value
                end
            else
                println("Parameters intervals are needed for randomization.")
            end
        
        elseif button == Set(AbstractPlotting.Keyboard.Button[AbstractPlotting.Keyboard.i]) 
            #restart simulation with same initial conditions (what are initial conditions here, anyway?)
            #are the properties whose intervals are not given
            println("Restarting model.")
            e = model.space.extend
            modelobs[] = model = initialize_model(dims=e, params=init_params)
            #for (key, val) in init_params
            #    try
            #        e = model.space.extend
            #        model.properties[key] = val
            #    catch er
            #        println("Something went wrong while trying to reinitialize the model to givel initial conditions. Reinitializing it from default. Are you sure adequate initial parameters were given? ")
             
        elseif button == Set(AbstractPlotting.Keyboard.Button[AbstractPlotting.Keyboard.v])
            # save parameters
            new_filepath, tstamp = namefile(prepath, savepath)
            path = mkpath("$(@__DIR__)/$(prepath)params$tstamp")
            open("$path/params$tstamp.json", "w") do f 
                write(f, JSON.json(modelobs[].properties))
            end

            println("Parameters saved at file $(prepath)params$tstamp/params$tstamp.json")

        end
    end

    return scene, ids, colors, sizes, markers, pos, ac, as, am
end