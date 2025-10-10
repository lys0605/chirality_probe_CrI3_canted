using ForwardDiff, LinearAlgebra
using CairoMakie, LaTeXStrings, GeometryBasics
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

A_11(kx, ky; J₁=1, J₂=0.1, S=3/2, a=1) = 4*J₁*S * (1- (J₂/J₁)*sin(ky*a/2)^2)
A_22(kx, ky; J₁=1, J₂=0.1, S=3/2, a=1) = 4*J₁*S * (1- (J₂/J₁)*sin(kx*a/2)^2)
A_12(kx, ky; J₁=1, J₂=0.1, D=0.5, S=3/2, s=0.5 , a=1) = -4*J₁*S * s * (s*cos(kx*a/2)*cos(ky*a/2) -1im*D/J₁ * sin(kx*a/2)*sin(ky*a/2))
A_21(kx, ky; J₁=1, J₂=0.1, D=0.5, S=3/2, s=0.5 , a=1) = -4*J₁*S * s * (s*cos(kx*a/2)*cos(ky*a/2) +1im*D/J₁ * sin(kx*a/2)*sin(ky*a/2))
B_11(kx, ky) = 0
B_22(kx, ky) = 0
B_12(kx, ky; J₁=1, J₂=0.1, S=3/2, c=√(3)/2 , a=1) = 4*J₁*S * c^2 * cos(kx*a/2) * cos(ky*a/2)

function minimal_checkerboard_altermangetic_model(kx, ky; J₁=1, J₂=0.1, D=0.5, S=3/2, s=0.5, c=√(3)/2, a=1)
    H = [A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) A_12(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a) 0 B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a);
        A_21(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a) A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a) 0;
        0 conj(B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a)) conj(A_11(-kx, -ky; J₁=J₁, J₂=J₂, S=S, a=a)) conj(A_12(-kx, -ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a));
        conj(B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a)) 0 conj(A_21(-kx, -ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a)) conj(A_22(-kx, -ky; J₁=J₁, J₂=J₂, S=S, a=a))]
    return H
end

function minimal_checkerboard_altermangetic_model(k_vec; J₁=1, J₂=0.1, D=0.5, S=3/2, s=0.5, c=√(3)/2, a=1)
    kx = k_vec[1]
    ky = k_vec[2]
    H = [A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) A_12(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a) 0 B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a);
        A_21(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a) A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a) 0;
        0 conj(B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a)) conj(A_11(-kx, -ky; J₁=J₁, J₂=J₂, S=S, a=a)) conj(A_12(-kx, -ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a));
        conj(B_12(kx ,ky; J₁=J₁, J₂=J₂, S=S, c=c, a=a)) 0 conj(A_21(-kx, -ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a)) conj(A_22(-kx, -ky; J₁=J₁, J₂=J₂, S=S, a=a))]
    return H
end
minimal_checkerboard_altermangetic_model(1,1; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)

a = 1 # set to per unit lattice constnat
S = 3/2
J₁ = 1
J₂ = -0.04
D = 0.5*J₁
Bₛ = 8*J₁*S
B = 0.125*Bₛ
s = B/Bₛ
c = cos(asin(s))
E₀ = 4*J₁*S 

# high-symmetry points
Γ = [0.0, 0.0]
M = [pi, pi]*a
X = [0.0, pi]*a 
Y = [pi, 0.0]*a

# get k-vectors
k0 = get_kvectors(Γ, X, n=51)
k1 = get_kvectors(X, M, n=51)
k2 = get_kvectors(M, Γ, n=100)
k3 = get_kvectors(Γ, Y, n=51)

k_vectors = group_kvectors(k0, k1, k2, k3)
path = sqrt.(k_vectors[1].^2 .+ k_vectors[2].^2)
path_index = get_path_index(k0, k1, k2, k3)
k_labels = ["Γ", "X", "M", "Γ", "Y"]

k = [[x, y] for (x,y) in zip(k_vectors[1],k_vectors[2])]
τ₃ = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
H_eff_path = [τ₃ * minimal_checkerboard_altermangetic_model(k[i]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(k)]
H_eff_path_eig = [eigen(H_eff_path[i]) for i in 1:length(k)]
eigenvalues_path = [H_eff_path_eig[i].values for i in 1:length(k)]
eigenvectors_path = [H_eff_path_eig[i].vectors for i in 1:length(k)]

real_eigenvalues_path = real.(eigenvalues_path)
E₁_real_path = getindex.(real_eigenvalues_path, 1)
E₂_real_path = getindex.(real_eigenvalues_path, 2)
E₃_real_path = getindex.(real_eigenvalues_path, 3)
E₄_real_path = getindex.(real_eigenvalues_path, 4)

# plots path
f = Figure()
ax = Axis(f[1,1];
    title = "Altermagnet Band Structure",
    xlabel = "k (1/Å)",
    ylabel = "Energy (meV)",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yticks = [0, 1, 2],
    #yautolimitmargin = (0, 0),
    xautolimitmargin = (0, 0)
)
line_upper = lines!(ax, 1:length(path), E₄_real_path./ E₀ , color = (:red, 0.5), linewidth = 1)
line_lower = lines!(ax, 1:length(path), E₃_real_path./ E₀, color = (:blue, 0.5), linewidth = 1) #
Legend(f[1,2], [line_upper, line_lower], ["ϵ₊/4JS", "ϵ₋/4JS"]; orientation = :vertical, title = "Bands", framevisible = false)
#lines!(ax, 1:length(path), (E₄_real_path .- E₃_real_path) ./ E₀, color = (:blue, 0.5), linewidth = 2)
#lines!(ax, 1:length(path), E₂_real_path, color = (:blue, 0.5), linewidth = 2) # julia sort from smallest to largest
#lines!(ax, 1:length(path), E₁_real_path, color = (:blue, 0.5), linewidth = 2) # 
f

test = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
norm = [v' * τ₃ * v for v in eachcol(U)]
norm 

eigenvalues_path
