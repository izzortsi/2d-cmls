
# export AbstractParticle, Particle, initialize_model, model_step!, agent_step!, make_gif, ac, avg_nbsize, avg_activation

##

abstract type AbstractParticle <: AbstractAgent end

##
mutable struct Particle <: AbstractParticle
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    speed::Float64
    mass::Float64
    activation::Float64
    interaction_radius::Float64
    nb_size::Float64
end

##
function initialize_model(;
    dims=(100, 100),
    params=Dict(
    :n_particles => 100, 
    :speed => 1.0, 
    :iradius => 4.0, 
    :min_nb => 0., 
    :max_nb => 1., 
    :cohere_factor => 0.25, 
    :separation => 4.0, 
    :separate_factor => 0.25, 
    :match_factor => 0.01)
)

    space2d = ContinuousSpace((2, dims[1]); periodic=true)

    model = ABM(Particle, space2d; scheduler=random_activation, properties=params, warn=false)
    id = 0
    for _ in 1:params[:n_particles]
        id += 1
        pos = Tuple(rand(2) .* dims[1] .- 1)
        vel = Tuple(rand(2) * 2 .- 1)
        mass = randexp() * 0.5 * exp(-0.5)
        speed = params[:speed]
        activation = 0 # rand()
        interaction_radius = params[:iradius]
        nb_size = 0
        particle = Particle(id, pos, vel, speed, mass, activation, interaction_radius, nb_size)
        add_agent!(particle, model)
    end
    # model.index!
    # index!(model)
    return model
end

##

function model_step!(model)
    nothing
end

function agent_step!(particle, model)
    # Obtain the ids of neighbors within the particle's visual distance
    ids = space_neighbors(particle, model, particle.interaction_radius)
    particle.nb_size = length(ids)

    particle.nb_size > model.max_nb ? model.max_nb = particle.nb_size : nothing
    particle.nb_size < model.min_nb ? model.min_nb = particle.nb_size : nothing
    med_nb = (model.min_nb + model.max_nb) / 2

    (particle.nb_size > med_nb) & (particle.activation < 0.94) ? particle.activation += 0.05 : nothing
    (particle.nb_size <= med_nb) & (particle.activation > 0.11) ? particle.activation -= 0.1 : nothing

    # Compute velocity based on rules defined above
    particle.vel =
        (
            particle.vel .+ cohere(particle, model, ids) .+ separate(particle, model, ids) .+
            match(particle, model, ids)
        ) ./ 2
    particle.vel = particle.vel ./ norm(particle.vel)
    # Move particle according to new velocity and speed
    move_agent!(particle, model, particle.speed)
end

function cohere(particle, model, ids)
    N = max(length(ids), 1)
    particles = model.agents
    coherence = (0.0, 0.0)
    for id in ids
        coherence = coherence .+ get_heading(particles[id], particle) .* (1 + particles[id].activation)
    end
    return coherence ./ N .* model.cohere_factor
end

function separate(particle, model, ids)
    seperation_vec = (0.0, 0.0)
    N = max(length(ids), 1)
    particles = model.agents
    for id in ids
        neighbor = particles[id]
        if distance(particle, neighbor) < model.separation
            seperation_vec = seperation_vec .- get_heading(neighbor, particle)
        end
    end
    return seperation_vec ./ N .* model.separate_factor
end

function match(particle, model, ids)
    match_vector = (0.0, 0.0)
    N = max(length(ids), 1)
    particles = model.agents
    for id in ids
        match_vector = match_vector .+ particles[id].vel
    end
    return match_vector ./ N .* model.match_factor
end

distance(a1, a2) = sqrt(sum((a1.pos .- a2.pos).^2))
get_heading(a1, a2) = a1.pos .- a2.pos

ac(a) = (colorsigned(cmap[1], cmap[50], cmap[end]) ∘ scalesigned(0.0, 0.5, 1.0))(a.activation)
avg_nbsize(model) = mean(collect(a.nb_size for a in allagents(model)))
avg_activation(model) = mean(collect(a.activation for a in allagents(model)))