using Observables, Dates, GLMakie


function simulate(M; resolution=(1280, 720), fps = 24)
    


    fig = Figure(resolution=resolution)


    run_obs = Observable{Bool}(false)
    rec_obs = Observable{Bool}(false)

    dnA = Node(M.A)
    hnA = lift(Array, dnA)
    ax1, hm1 = heatmap(fig[1, 1], hnA, colorrange=(0, 1))
    hidedecorations!(ax1)

    dnU = Node(M.U)
    hnU = lift(Array, dnU)
    ax2, hm2 = heatmap(fig[1, 2], hnU, colorrange=(0, 1))
    hidedecorations!(ax2)

    dnG = Node(M.G)
    hnG = lift(Array, dnG)
    ax3, hm3 = heatmap(fig[2, 1], hnG, colorrange=(0, 1))
    hidedecorations!(ax3)

    # dims, = size(M.K[1])
    # mid = dims ÷ 2
    # r = M.R

    # heatmap(fig[2,2][1,1], M.K[1][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))
    # heatmap(fig[2,2][1,2], M.K[2][mid-r:mid+r+2, mid-r:mid+r+2], colorrange=(0, 1))

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
                    dnA[] = M.A
                    dnU[] = M.U
                    dnG[] = M.G
                    sleep(1 / fps)
                end
            elseif event.key == Keyboard.a
                
                M.populate!()
                dnA[] = M.A

            elseif event.key == Keyboard.c
                
                A = CUDA.zeros(SIZE, SIZE)
                M.A .= A
                M.U .= A
                M.G .= A
                populate(M.A, creature["cells"], 100)
                M.update!(M)
                dnA[] = M.A
                dnU[] = M.U
                dnG[] = M.G

            elseif event.key == Keyboard.v
                
                A = cu(bitrand(SIZE, SIZE)*1.0)
                B = CUDA.zeros(SIZE, SIZE)
                M.A .= A
                M.U .= A
                M.G .= A
                
                M.update!(M)
                dnA[] = M.A
                dnU[] = M.U
                dnG[] = M.G

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
