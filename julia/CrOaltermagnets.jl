using LinearAlgebra, ForwardDiff
using CairoMakie, GeometryBasics
import Meshes

function get_kvectors(pt1::Vector, pt2::Vector; n::Int=100)
    # Function to calculate the k-vectors from tw\o points in reciprocal space
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

function h₀(kx,ky; J=2, L=-3, L′=L/2, L′′=L/5) 
    return 4 * J - 2 * L - L′ - L′′
end

function γ₁(kx, ky; J=2, L=-3, L′=L/2, L′′=L/5)
    return  2 * J * cos(kx/2)
end 

function γ₂(kx, ky; J=2, L=-3, L′=L/2, L′′=L/5) 
    return  2 * L * cos((-kx + ky)/2) + L′ * exp(1im * (kx + ky)/2) + L′′ * exp(-1im * (kx + ky)/2)
end

function γ₃(kx, ky; J=2, L=-3, L′=L/2, L′′=L/5) 
    return  2 * J * cos(ky/2) 
end

function γ₄(kx, ky; J=2, L=-3, L′=L/2, L′′=L/5) 
    return  2 * L * cos((kx + ky)/2) + L′ * exp(1im * (-kx + ky)/2) + L′′ * exp(1im * (kx - ky)/2)
end

function altermagnetic_magnon_model(kx, ky; J=2, S=3/2, L=-3, L′=L/2, L′′=L/5)
    h_int = [h₀(kx, ky; J=J, L=L, L′=L′, L′′=L′′) γ₁(kx, ky; J=J, L=L, L′=L′, L′′=L′′);
             conj(γ₁(kx, ky; J=J, L=L, L′=L′, L′′=L′′)) h₀(kx, ky; J=J, L=L, L′=L′, L′′=L′′)]
    Δ = [γ₂(kx, ky; J=J, L=L, L′=L′, L′′=L′′) γ₃(kx, ky; J=J, L=L, L′=L′, L′′=L′′);
         γ₃(kx, ky; J=J, L=L, L′=L′, L′′=L′′) γ₄(kx, ky; J=J, L=L, L′=L′, L′′=L′′)]
    return S * [h_int Δ; conj(Δ) h_int]
end 

altermagnetic_magnon_model(k::Vector;  J=2, S=3/2, L=-3, L′=L/2, L′′=L/5) = altermagnetic_magnon_model(k[1], k[2]; J=J, S=S, L=L, L′=L′, L′′=L′′)

# Set the parameters
J = 2
S = 1
L = -3
L′ = L/2
L′′ = L/5

# high symmetry points
Γ = [0.0, 0.0] # Γ
M = [pi, pi] # M 
X = [0.0, pi] # X
Y = [pi, 0.0] # Y

# get k-vectors
k0 = get_kvectors(Γ, M, n=100)
k1 = get_kvectors(M, X, n=51)
k2 = get_kvectors(X, Γ, n=51)
k3 = get_kvectors(Γ, Y, n=51)

k_vectors = group_kvectors(k0, k1, k2, k3)
path = sqrt.(k_vectors[1].^2 .+ k_vectors[2].^2)
path_index = get_path_index(k0, k1, k2, k3)
k_labels = ["Γ", "M", "X", "Γ", "Y"]

k = [[x, y] for (x,y) in zip(k_vectors[1],k_vectors[2])]
τ₃ = [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
H_eff_path = [τ₃ * altermagnetic_magnon_model(k[i]; J=J, S=S, L=L, L′=L′, L′′=L′′) for i in 1:length(k)]
H_eff_path_eig = [eigen(H_eff_path[i]) for i in 1:length(k)]
eigenvalues_path = [H_eff_path_eig[i].values for i in 1:length(k)]
eigenvectors_path = [H_eff_path_eig[i].vectors for i in 1:length(k)]

real_eigenvalues_path = real.(eigenvalues_path)
E₁_real_path = getindex.(real_eigenvalues_path, 1)
E₂_real_path = getindex.(real_eigenvalues_path, 2)
E₃_real_path = getindex.(real_eigenvalues_path, 3)
E₄_real_path = getindex.(real_eigenvalues_path, 4)


σ₃ = Diagonal(vcat(ones(2), -ones(2)))

# Hamiltonian, eigenvalues and eigenvectors
H =  [altermagnetic_magnon_model(kx[i], ky[j]; J=J, S=S, L=L, L′=L′, L′′=L′′) for i in 1:Nx, j in 1:Ny]
H_eff = [τ₃ * altermagnetic_magnon_model(kx[i], ky[j]; J=J, S=S, L=L, L′=L′, L′′=L′′) for i in 1:Nx, j in 1:Ny]

eigenvalues = [τ₃ * eigvals(H_eff[i, j]) for i in 1:Nx, j in 1:Ny] # Apply the τ₃ transformation to the eigenvalues
eigenvectors = [eigvecs(H_eff[i, j]) for i in 1:Nx, j in 1:Ny]

real_eigenvalues = real.(eigenvalues)
E₁_real = getindex.(real_eigenvalues, 1)
E₂_real = getindex.(real_eigenvalues, 2)
E₃_real = getindex.(real_eigenvalues, 3)
E₄_real = getindex.(real_eigenvalues, 4)

# Plotting the eigenvalues
fig = Figure()
ax1 = Axis3(fig[1, 1], title="Altermagnet band structure", xlabel="kx", ylabel="ky")
surface!(ax1, kx, ky, E₄_real,  colormap=:winter, alpha=0.5)
surface!(ax1, kx, ky, E₃_real,  colormap=:dense, alpha=0.5)
#surface!(ax1, kx, ky, E₂_real,  colormap=:dense, alpha=0.5)
#surface!(ax1, kx, ky, E₁_real,  colormap=:viridis, alpha=0.5)
ax_heat = Axis(fig[2, 1], title="Parameters", aspect=DataAspect() , xlabel="kx", ylabel="ky")
hm = heatmap!(ax_heat, kx, ky, Ω, colormap=:viridis)
poly!(ax_heat, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[2, 2], hm, label="Energy (meV)")
#surface!(ax1, kx, ky, upper_band, colormap=:plasma)
#surface!(ax1, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
ax1.azimuth[] = π/4    # Horizontal rotation (radians)
ax1.elevation[] = π/16  # Vertical tilt (radians)
fig 

# plots path
f = Figure()
ax = Axis(f[1,1];
    title = "Altermagnet Band Structure",
    xlabel = "k (1/Å)",
    ylabel = "Energy (meV)",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yticks = [0, 5, 10, 15, 20, 25]
)
lines!(ax, 1:length(path), E₄_real_path , color = (:red, 0.5), linewidth = 2)
lines!(ax, 1:length(path), E₃_real_path, color = (:red, 0.5), linewidth = 2) #
lines!(ax, 1:length(path), -E₂_real_path, color = (:blue, 0.5), linewidth = 2) # negative band -> antiferromagnetic -> chirality -1
lines!(ax, 1:length(path), -E₁_real_path, color = (:blue, 0.5), linewidth = 2) # -1 chirality
f

test = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
norm = [v' * τ₃ * v for v in eachcol(U)]
norm 

