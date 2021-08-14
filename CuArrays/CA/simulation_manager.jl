using Observables, GLMakie, Dates
# #%%
# fig = Figure()
# #%%
# fig.scene
# #%%
# events(fig).window_open[]
# #%%




function simulate(M; resolution=(1280, 720), fps = 24)
    


    fig = Figure(resolution=resolution)

    #modelobs = Observable(model.A)
    run_obs = Observable{Bool}(false)
    rec_obs = Observable{Bool}(false)

    #running_label = Label(fig[0, :], lift(x -> x ? "RUNNING" : "HALTED", run_obs))

    #ax1 = Axis(fig[1, 1])
    #heatmap!(ax1, modelobs)
    #fig, hm = heatmap(modelobs[].A)


    nA = Node(M.A)
    ax1, hm1 = heatmap(fig[1, 1], nA, colorrange=(0, 1))
    hidedecorations!(ax1)

    nU = Node(M.U)
    ax2, hm2 = heatmap(fig[1, 2], nU, colorrange=(0, 1))
    hidedecorations!(ax2)

    nG = Node(M.G)
    ax3, hm3 = heatmap(fig[2, 1], nG, colorrange=(0, 1))
    hidedecorations!(ax3)

    dims, = size(M.K[1])
    mid = dims ÷ 2
    r = M.R

    heatmap(fig[2,2][1,1], M.K[1][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))
    heatmap(fig[2,2][1,2], M.K[2][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))

    stream = VideoStream(fig.scene, framerate=fps)
    #fig

    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
           if event.key == Keyboard.s
                run_obs[] = !run_obs[]
                run_obs[] ? println("Simulation running. $(run_obs[])") : println("Simulation stopped.")

                @async while events(fig).window_open[] && run_obs[] 
                    # update observables in scene
                    M.update!(M)
                    nA[] = M.A[:,:]
                    nU[] = M.U[:,:]
                    nG[] = M.G[:,:]
                    sleep(1 / fps)
                end
            elseif event.key == Keyboard.a
                
                M.populate!()
                nA[] = M.A[:,:]

            elseif event.key == Keyboard.c
                
                A = zeros(SIZE, SIZE)
                M.A[:,:] = A[:,:]
                M.U[:,:] = A[:,:]
                M.G[:,:] = A[:,:]
                populate(M.A, orbium["cells"], 30)
                M.update!(M)

            elseif event.key == Keyboard.r
                if !rec_obs[]
                    # start recording
                    # start a new stream and set a new filename for the recording
                    stream = VideoStream(fig.scene, framerate=fps)
                    rec_obs[] = !rec_obs[]
                    println("Recording started.")
    
                    @async while events(fig).window_open[] && rec_obs[]
                        recordframe!(stream)
                        sleep(1 / fps)
                    end
    
                elseif rec_obs[]
                    # save stream and stop recording
                    
                    opath = pwd() * "/CuArrays/outputs/" * "lenia/"
                    mkpath(opath)
                    filename = replace("mn_orbia_$(Dates.Time(Dates.now()))", ":" => "_") *".mp4"

                    rec_obs[] = !rec_obs[]

                    save(opath * filename, stream)
                    println("Recording stopped. Files saved at $(opath * filename).")
                end
            end
        end
        # Let the event reach other listeners
        return Consume(false)
    end
    return fig
end

