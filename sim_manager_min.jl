using Observables, GLMakie
#%%


function simulate(model; resolution=(1280, 720), fps = 24)
    
    
    scene, layout = layoutscene(resolution=resolution)

    # model-related observables
    modelobs = Observable(model)
    run_obs = Observable{Bool}(false)
    running_label = Label(scene, lift(x -> x ? "RUNNING" : "HALTED", run_obs))

    ax1 = layout[1, 1] = Axis(scene, tellheight=true, tellwidht=true)
    infos = GridLayout(tellheight=false, tellwidth=true)
    infos[1, 1] = running_label
    fig, hm = heatmap(modelobs[].A)
    on(scene.events.keyboardbuttons) do button

        if button == Set(Keyboard.Button[Keyboard.s]) 

            run_obs[] = !run_obs[]
            run_obs[] ? println("Simulation running. $(run_obs[])") : println("Simulation stopped.")

            @async while run_obs[]
                # update observables in scene
                modelobs[].update!(modelobs[]) 
                sleep(1 / fps)
            end
        end
    end
    
    on(scene.events.mousebuttons) do buttons
        if ispressed(scene, Mouse.left)
            pos = to_world(scene, Point2f0(scene.events.mouseposition[]))
            clicks[] = push!(clicks[], pos)
        end
        return
     end

    return scene
end

