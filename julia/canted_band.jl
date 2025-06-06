using CairoMakie
using LinearAlgebra

function get_kvectors(pt1::Vector{Float64}, pt2::Vector{Float64}; n::Int=100)
    # Function to calculate the k-vectors from two points in reciprocal space
    # pt1 and pt2 are 3D vectors (x, y, z) in reciprocal space
    # Returns the k-vectors as a tuple (kx, ky)
    kx = range(pt1[1], pt2[1], n)
    ky = range(pt1[2], pt2[2], n)
    k = transpose([kx,ky])
    return k
end

function group_kvectors(k...)
    # Function to group k-vectors into a 2D array
    # k is a vector of k-vectors
    # Returns a 2D array of k-vectors
    n = length(k)
    k_vectors = k[1]
    for i in 2:n
       k_vectors = [vcat(first..., second[2:end]...) for (first, second) in zip(k_vectors, k[i])]
    end
    return k_vectors
end

function get_path_index(k...)
    # Function to get the pat
    # index of the k-vectors
    n = length(k)
    lengths = zeros(n)
    path_index = ones(n+1)
    for i in 1:n
        lengths[i] = length(k[i][1])
    end
    path_index[end] = sum(lengths)
    for j in  1:n
        path_index[j+1] = path_index[j] + lengths[j]-1
    end
    return path_index
end

function canted_energy(k::Vector{Float64}; J=1.54, D=0.1, S=5/2, s=0.6)
    # Function to calculate the energy of the canted antiferromagnetic model
    # k is a 2D vector (kx, ky) in reciprocal space
    # J is the exchange interaction strength (meV)
    # D is the Dzyaloshinskii-Moriya interaction strength (meV)
    # S is the spin quantum number
    # s is the canting angle (radians)
    # Returns the energy of the system (meV)
    Bs = 6*J*S
    v = s^2

    a = 1 # set to per unit lattice constnat
    
    # lattice structure parameters
    n_n = [[0, 1], [-√3/2, -1/2], [√3/2, -1/2]] .* a
    next_n_n = [[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]] .* a
    
    # hopping parameters
    gamma = sum(exp.(1im*dot.(Ref(k), transpose(n_n))))
    gamma_sin = sum(sin.(dot.(Ref(k), transpose(next_n_n))))

    # useful parameters
    ϕ_k = 2*J*S*gamma
    λ_k = 4*D*S*s*gamma_sin
    ϕ_k_sq = ϕ_k*ϕ_k'
    Δ_k = √(λ_k^2 + v^2*ϕ_k_sq)
    φ_k = 1im*log(ϕ_k/abs(ϕ_k))

    # energy band +,-
    energy = [sqrt((Bs-Δ_k)^2 - (1-v)^2*ϕ_k_sq), 
               sqrt((Bs+Δ_k)^2 - (1-v)^2*ϕ_k_sq)]
    return real(energy)
end

Γ = [0.0, 0.0] # Gamma
K = 2*pi*[2/3,0]/√3 #K 
M = 2*pi*[1/2,1/(2*√3)]/√(3) # M
K′ = 2*pi*[1/3,1/√(3)]/√(3) # K'

k0 = get_kvectors(-1*K, Γ, n=100)
k1 = get_kvectors(Γ, K, n=101)
k2 = get_kvectors(K, M, n=51)
k3 = get_kvectors(M, K′, n=51)
k4 = get_kvectors(K′, Γ, n=101)

k_vectors = group_kvectors(k0, k1, k2, k3, k4)
path = sqrt.(k_vectors[1].^2 .+ k_vectors[2].^2)
path_index = get_path_index(k0, k1, k2, k3, k4)
k_labels = ["K′", "Γ", "K", "M", "K′", "Γ"]

k = [[x, y] for (x,y) in zip(k_vectors[1],k_vectors[2])]
energy = canted_energy.(k; J=1, s=0.5)
energy = reduce(hcat, energy)

# plots
f = Figure()
ax = Axis(f[1,1];
    title = "Canted Antiferromagnetic Model",
    xlabel = "k (1/Å)",
    ylabel = "Energy (meV)",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
)
lines!(ax, 1:length(path), energy[1,:], color = :blue, linewidth = 2)
lines!(ax, 1:length(path), energy[2,:], color = :red, linewidth = 2)
f