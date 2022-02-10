using Observables, Dates, GLMakie

const MUTATION_RATE = 0.01

function update_update(M::MNCA)
    function continuous_update(M::MNCA)
        conv(M.A, M.K[1], M.U)
        M.G .= (M.δ[1].(M.U) .* M.A)
        function aux_update(U)
            u1 = (U .<= 33) * M.r 
            u2 = (34 .<= U .<= 45) * M.k
            u3 = (46 .<= U .<= 57) .* M.A
            u4 = (U .>= 58) * M.e
            return u1 + u2 + u3 + u4
        end
        M.A .= clamp.(aux_update(M.U), 0, 1)
    end
    M.Φ[1] = continuous_update
end


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
    ax2, hm2 = heatmap(fig[1, 2], hnU)#, colorrange=(0, 1))
    hidedecorations!(ax2)

    dnG = Node(M.G)
    hnG = lift(Array, dnG)
    ax3, hm3 = heatmap(fig[2, 1], hnG, colorrange=(0, 1))
    hidedecorations!(ax3)

    fig[2, 2] = buttongrid = GridLayout(tellheight = false, tellwidth = false)

    # dims, = size(M.K[1])
    # mid = dims ÷ 2
    # r = M.R
    or = Observable(M.r)
    ok = Observable(M.k)
    oe = Observable(M.e)

    params = ["e"=> oe, "k" => ok, "r"=> or] |> Dict

    labels = []
    # props_obs[key] = Observable(value)
    
    for (key, val) in params
        plabel = lift(x -> "$(String(key)): $(x[])", params[key])
        push!(labels, plabel)
    end
    buttons = buttongrid[1:length(labels), 1] = [Button(fig, label = l) for l in labels]
    # running_label = LText(scene, lift(x -> x ? "RUNNING" : "HALTED", run_obs))
    # recording_label = LText(scene, lift(x -> x ? "RECORDING" : "STOPPED", rec_obs))

    # ax1 = fig[1, 1] = LAxis(scene, tellheight=true, tellwidht=true)
    # infos = GridLayout(tellheight=false, tellwidth=false)
    
    # infos[1:3, 1] = buttons
    # fig[2, 2] = infos
    # infos[2:5, 1] = GridLayout(tellwidth = false)

    # for (i, plabel) in enumerate(props_labels)
    #     infos[i + 2, 1] = plabel
    # end
    
    # layout[1, 2] = infos

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
                A = CUDA.rand(SIZE, SIZE)
                # A += cu(bitrand(SIZE, SIZE)*1.0)
                # A = clamp.(A, 0, 1)
                # B = CUDA.zeros(SIZE, SIZE)
                M.A .= A
                M.U .= A
                M.G .= A
                
                M.update!(M)
                dnA[] = M.A
                dnU[] = M.U
                dnG[] = M.G

            elseif event.key == Keyboard.d

                dr, dk, de = sort(rand(3)/10)
                M.r += rand([-1, 1])*dr
                M.k += rand([-1, 1])*dk
                M.e += rand([-1, 1])*de
                params["r"][] = M.r
                params["k"][] = M.k
                params["e"][] = M.e
                function continuous_update(M::MNCA)
                    conv(M.A, M.K[1], M.U)
                    M.G .= (M.δ[1].(M.U) .* M.A)
                    function aux_update(U)
                        u1 = (U .<= 33) * M.r 
                        u2 = (34 .<= U .<= 45) * M.k
                        u3 = (46 .<= U .<= 57) .* M.A
                        u4 = (U .>= 58) * M.e
                        return u1 + u2 + u3 + u4
                    end
                    M.A .= clamp.(aux_update(M.U), 0, 1)
                end
                M.Φ[1] = continuous_update
                # A += cu(bitrand(SIZE, SIZE)*1.0)
                # A = clamp.(A, 0, 1)
                # B = CUDA.zeros(SIZE, SIZE)
            elseif event.key == Keyboard.h
                # dr = M.r/20
                dr = MUTATION_RATE
                M.r += dr
                params["r"][] = M.r
                update_update(M)
            elseif event.key == Keyboard.b
                # dr = M.r/20
                dr = MUTATION_RATE
                M.r -= dr
                params["r"][] = M.r
                update_update(M)
            elseif event.key == Keyboard.j
                # dk = M.k/20
                dk = MUTATION_RATE
                M.k += dk
                params["k"][] = M.k
                update_update(M)
            elseif event.key == Keyboard.n
                # dk = M.k/20
                dk = MUTATION_RATE
                M.k -= dk
                params["k"][] = M.k
                update_update(M)

            elseif event.key == Keyboard.k
                # de = M.e/20
                de = MUTATION_RATE
                M.e += de
                params["e"][] = M.e
                update_update(M)
            elseif event.key == Keyboard.m
                # de = M.e/20
                de = MUTATION_RATE
                M.e -= de
                params["e"][] = M.e
                update_update(M)                                             


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
