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

E(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sqrt(
    (A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a)^2 - A_22(kx,  ky; J₁=J₁, J₂=J₂, S=S, a=a)^2)^2 - 
    4*B_12(kx, ky; J₁=J₁, J₂=J₂, S=S, c=c , a=a)^2 * (A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) - A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a))^2
    + 4*(A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a) + A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a))^2 * abs2(A_12(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a))
)

ω₊(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sqrt(
    A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a)^2 + A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a)^2 - 2*B_12(kx, ky; J₁=J₁, J₂=J₂, S=S, c=c , a=a)^2
    + 2*abs2(A_12(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a)) + E(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)
) / sqrt(2)
ω₋(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sqrt(abs(
    A_11(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a)^2 + A_22(kx, ky; J₁=J₁, J₂=J₂, S=S, a=a)^2 - 2*B_12(kx, ky; J₁=J₁, J₂=J₂, S=S, c=c , a=a)^2
    + 2*abs2(A_12(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s , a=a)) - E(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)
)) / sqrt(2)


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
J₂ = -0.05*J₁
D = 0.0*J₁
Bₛ = 8*J₁*S
B = 0.0*Bₛ
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
eigenvalues_path = [H_eff_path_eig[i].values for i in 1:length(k)]# sort eigenvalues from largest to smallest
eigenvectors_path = [H_eff_path_eig[i].vectors for i in 1:length(k)]

perm = [4; 3; 1 ;2]
eigenvalues_path = [eigenvalues_path[i][perm] for i in 1:length(k)] 
eigenvectors_path = [eigenvectors_path[i][:, perm] for i in 1:length(k)] # sort eigenvectors according to sorted eigenvalues

real_eigenvalues_path = real.(eigenvalues_path)
E₁_real_path = getindex.(real_eigenvalues_path, 1)
E₂_real_path = getindex.(real_eigenvalues_path, 2)
E₃_real_path = getindex.(real_eigenvalues_path, 3)
E₄_real_path = getindex.(real_eigenvalues_path, 4)
ω₊_values = [ω₊(k[i][1], k[i][2]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(k)]
ω₋_values = [ω₋(k[i][1], k[i][2]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(k)]

ω₊_values[51] + ω₋_values[51]
 
u1_path = getindex.(eigenvectors_path, :, 1)
u2_path = getindex.(eigenvectors_path, :, 2)
u3_path = getindex.(eigenvectors_path, :, 3)
u4_path = getindex.(eigenvectors_path, :, 4)

s1_z =  [abs(u1_path[i][1])^2 - abs(u1_path[i][2])^2 - abs(u1_path[i][3])^2 + abs(u1_path[i][4])^2 for i in 1:length(k)]
s2_z = [abs(u2_path[i][1])^2 - abs(u2_path[i][2])^2 - abs(u2_path[i][3])^2 + abs(u2_path[i][4])^2 for i in 1:length(k)]

line_1_J2_005_D_05_B_0125_positive = E₁_real_path
line_2_J2_005_D_05_B_0125_positive = E₂_real_path

line_1_J2_0_D_B_0 = E₁_real_path
line_2_J2_0_D_B_0 = E₂_real_path

line_1_J2_005_D_B_0 = E₁_real_path
line_2_J2_005_D_B_0 = E₂_real_path

line_1_J2_0_D_05_B_0125 = E₁_real_path
line_2_J2_0_D_05_B_0125 = E₂_real_path

line_1_J2_005_D_05_B_0125 = E₁_real_path
line_2_J2_005_D_05_B_0125 = E₂_real_path

# plots path
f = Figure(size=(800, 400), figure_padding=1)
ax = Axis(f[1,1];
    #title = L"\textbf{Altermagnet Band Structure } J_2=0",
    xlabel = L"k/π (1/Å)",
    ylabel = L"\text{Energy (meV)}",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    #yticks = [0, 1, 2],
    #yautolimitmargin = (0, 0),
    xautolimitmargin = (0, 0)
)
line_upper = lines!(ax, 1:length(path), s1_z, color = (:red, 0.5), linewidth = 1)
line_lower = lines!(ax, 1:length(path), s2_z, color = (:blue, 0.5), linewidth = 1) #
#Legend(f[1,2], [line_upper, line_lower], ["ϵ₊", "ϵ₋"]; orientation = :vertical, title = "Bands", framevisible = false)
# Label(f[1,1][1, 1, TopLeft()], "(a)",
#     fontsize= 20,
#     padding = (0, 20, 0 ,0),)

ax_2 = Axis(f[1,2];
    #title = L"\textbf{Altermagnet Band Structure } J_2=-0.05J_1",
    xlabel = L"k/π (1/Å)",
    ylabel = L"\text{Energy (meV)}",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    #yticks = [0, 1, 2],
    #yautolimitmargin = (0, 0),
    xautolimitmargin = (0, 0)
)
line_upper_2 = lines!(ax_2, 1:length(path), line_1_J2_005_D_B_0 , color = (:red, 0.5), linewidth = 1)
line_lower_2 = lines!(ax_2, 1:length(path), line_2_J2_005_D_B_0, color = (:blue, 0.5), linewidth = 1) #
#Legend(f[1,4], [line_upper_2, line_lower_2], ["ϵ₊", "ϵ₋"]; orientation = :vertical, title = "Bands", framevisible = false)
# Label(f[1,2][1, 1, TopLeft()], "(b)",
#     fontsize= 20,
#     padding = (-5, 20, 0, 0),)

ax_3 = Axis(f[2,1];
    #title = L"\textbf{Altermagnet Band Structure } J_2=0",
    xlabel = L"k/π (1/Å)",
    ylabel = L"\text{Energy (meV)}",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    #yticks = [0, 1, 2, 3, 4, 5, 6],
    #yautolimitmargin = (0, 0),
    xautolimitmargin = (0, 0)
)
line_upper_3 = lines!(ax_3, 1:length(path), line_1_J2_0_D_05_B_0125 , color = (:red, 0.5), linewidth = 1)
line_lower_3 = lines!(ax_3, 1:length(path), line_2_J2_0_D_05_B_0125 , color = (:blue, 0.5), linewidth = 1) #
#Legend(f[2,2], [line_upper_3, line_lower_3], ["ϵ₊", "ϵ₋"]; orientation = :vertical, title = "Bands", framevisible = false)
# Label(f[2,1][1, 1, TopLeft()], "(c)",
#     fontsize= 20,
#     padding = (0, 20, 0, 0),)

ax_4 = Axis(f[2,2];
    #title = L"\textbf{Altermagnet Band Structure } J_2=-0.05J_1",
    xlabel = L"k/π (1/Å)",
    ylabel = L"\text{Energy (meV)}",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yticks = [0, 1, 2, 3, 4, 5, 6],
    #yautolimitmargin = (0, 0),
    xautolimitmargin = (0, 0)
)
line_upper_4 = lines!(ax_4, 1:length(path), line_1_J2_005_D_05_B_0125, color = (:red, 0.5), linewidth = 1)
line_lower_4 = lines!(ax_4, 1:length(path), line_2_J2_005_D_05_B_0125 , color = (:blue, 0.5), linewidth = 1) #
Legend(f[1,3], [line_upper_2, line_lower_2], [L"E^+", L"E^{-}"]; orientation = :vertical, title = "Bands", framevisible = true)
# Label(f[2,2][1, 1, TopLeft()], "(d)",
#     fontsize= 20,
#     padding = (-5, 20, 0, 0),)
f

# ax_3 = Axis(f[2,1];
#     title = L"\textbf{Altermagnet magnon magnetic moment}",
#     xlabel = L"k (1/Å)",
#     ylabel = L"\text{<S^z>}",
#     xticks = (path_index, k_labels),
#     yminorticks = IntervalsBetween(5),
#     yminorticksvisible = true,
#     yticks = [0, 1, 2],
#     #yautolimitmargin = (0, 0),
#     xautolimitmargin = (0, 0)
# )
# line_s1_z = lines!(ax_3, 1:length(path), s1_z , color = (:red, 0.5), linewidth = 1)
# line_s2_z = lines!(ax_3, 1:length(path), s2_z , color = (:blue, 0.5), linewidth = 1) #
# Legend(f[1,4], [line_s1_z, line_s2_z], [L"<S_1^z> ", L"<S_2^z>"]; orientation = :vertical, title = "", framevisible = false)

save("julia/figures/topological_altermagnet_band_structures_checkerboard_notitle.png", f, px_per_unit=300/96)

# Derivatives of the Hamiltonian
H₁(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(kx -> minimal_checkerboard_altermangetic_model(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), kx)
H₂(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(ky -> minimal_checkerboard_altermangetic_model(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), ky)
H₁₁(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(kx -> H₁(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), kx)
H₁₂(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(ky -> H₁(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), ky)
H₂₁(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(kx -> H₂(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), kx)
H₂₂(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ForwardDiff.derivative(ky -> H₂(kx, ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), ky)

# Define the number of points in kx and ky
Nx = 100
Ny = 100

# Define the kx and ky ranges
kx = range(-π/a, π/a, length=Nx)
ky = range(-π/a, π/a, length=Ny)
dkx = step(kx)
dky = step(ky)

# parameters
a = 1 # set to per unit lattice constnat
S = 3/2
J₁ = 1
J₂ = -0.0*J₁
D = 0.5*J₁
Bₛ = 8*J₁*S
B = 0.125*Bₛ
s = B/Bₛ
c = cos(asin(s))
E₀ = 4*J₁*S 

J₂_values = [-0.5, -0.15, -0.05, -0.025, -0.005] .* J₁

H_AM =  [minimal_checkerboard_altermangetic_model(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
H_AM_eff = [τ₃ * minimal_checkerboard_altermangetic_model(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
H_AM_eff_eig = [eigen(H_AM_eff[i, j]) for i in 1:Nx, j in 1:Ny] # eigen systems
H_AM_eff_eigenvalues = [H_AM_eff_eig[i, j].values for i in 1:Nx, j in 1:Ny] # Apply the τ₃ transformation to the eigenvalues
H_AM_eff_eigenvectors = [H_AM_eff_eig[i, j].vectors for i in 1:Nx, j in 1:Ny]

perm = [4; 3; 1 ;2]
H_AM_eff_eigenvalues = [H_AM_eff_eigenvalues[i, j][perm] for i in 1:Nx, j in 1:Ny]
H_AM_eff_eigenvectors = [H_AM_eff_eigenvectors[i, j][:, perm] for i in 1:Nx, j in 1:Ny] # sort eigenvectors according to sorted eigenvalues
ϵ₁_values = real.(getindex.(H_AM_eff_eigenvalues, 1))
ϵ₂_values = real.(getindex.(H_AM_eff_eigenvalues, 2))
ϵ₃_values = real.(getindex.(H_AM_eff_eigenvalues, 3))
ϵ₄_values = real.(getindex.(H_AM_eff_eigenvalues, 4))
gap_12 = [(ϵ₁_values[i,j] - ϵ₂_values[i,j])^2 for i in 1:Nx, j in 1:Ny]
gap_13 = [(ϵ₁_values[i,j] - ϵ₃_values[i,j])^2 for i in 1:Nx, j in 1:Ny]
gap_14 = [(ϵ₁_values[i,j] - ϵ₄_values[i,j])^2 for i in 1:Nx, j in 1:Ny]

function pseudounitary_normalization(vec::Vector, τ)
    norm_factor = sqrt(abs(vec' * τ * vec))
    return vec / norm_factor
end

function pseudounitary_normalization(matrix::Matrix, τ)
    # normalizing column vectors
    no_rows, no_cols = size(matrix)
    normalized_vectors = [pseudounitary_normalization(matrix[:, i]::Vector, τ)  for i in 1:no_cols]
    return hcat(normalized_vectors...)
end
U_AM_k = [pseudounitary_normalization(H_AM_eff_eigenvectors[i, j], τ₃) for i in 1:Nx, j in 1:Ny]
U_AM_inv_k = [τ₃ * U_AM_k[i, j]' * τ₃ for i in 1:Nx, j in 1:Ny] # inverse of the pseudounitary matrix
U_AM_dag_k = [U_AM_k[i, j]' for i in 1:Nx, j in 1:Ny] # Hermitian conjugate of the pseudounitary matrix
u_L_1_k = [U_AM_inv_k[i, j][1, :] for i in 1:Nx, j in 1:Ny]
u_L_2_k = [U_AM_inv_k[i, j][2, :] for i in 1:Nx, j in 1:Ny]
u_L_3_k = [U_AM_inv_k[i, j][3, :] for i in 1:Nx, j in 1:Ny]
u_L_4_k = [U_AM_inv_k[i, j][4, :] for i in 1:Nx, j in 1:Ny]
u_R_1_k = [U_AM_k[i, j][:, 1] for i in 1:Nx, j in 1:Ny]
u_R_2_k = [U_AM_k[i, j][:, 2] for i in 1:Nx, j in 1:Ny]
u_R_3_k = [U_AM_k[i, j][:, 3] for i in 1:Nx, j in 1:Ny]
u_R_4_k = [U_AM_k[i, j][:, 4] for i in 1:Nx, j in 1:Ny]

L̃_11_AM = [U_AM_dag_k[i, j] * H₁₁(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_12_AM = [U_AM_dag_k[i, j] * H₁₂(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_21_AM = [U_AM_dag_k[i, j] * H₂₁(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_22_AM = [U_AM_dag_k[i, j] * H₂₂(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]

χ_AM = [2*imag.((L̃_11_AM[i,j] - L̃_22_AM[i,j]) .* conj.(L̃_12_AM[i,j])) for i in 1:Nx, j in 1:Ny]
χ_AM_FM = [χ_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,2] element for the FM case
χ_AM_2M = [χ_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny] # only [1,3] element for the 2M case
χ_AM_AFM = [χ_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny] # only [1,4] element for the AFM case

Ω_RCD_1 = (χ_AM_2M ./ (2 .* ϵ₁_values).^2 ) .+ (χ_AM_AFM ./ (ϵ₁_values .+ ϵ₂_values).^2)  .- (χ_AM_FM ./ (ϵ₁_values .- ϵ₂_values).^2 )
H₁_values = [H₁(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
H₂_values = [H₂(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
Ω₁_FM = [2 * imag( (transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₂_values[i,j])^2) for i in 1:Nx, j in 1:Ny]
Ω₁_2M = [2 * imag( (transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_3_k[i, j]) *  (transpose(u_L_3_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₃_values[i,j])^2) for i in 1:Nx, j in 1:Ny]
Ω₁_AFM = [2 * imag( (transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_4_k[i, j]) *  (transpose(u_L_4_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₄_values[i,j])^2) for i in 1:Nx, j in 1:Ny]
Ω₁ = Ω₁_FM .+ Ω₁_2M .+ Ω₁_AFM
Ω₁_normalized = Ω₁ ./ maximum(abs.(Ω₁))
g₁_11_FM = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₂_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_11_2M = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_3_k[i, j]) *  (transpose(u_L_3_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₃_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_11_AFM = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_4_k[i, j]) *  (transpose(u_L_4_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₄_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_11 = g₁_11_FM .+ g₁_11_2M .+ g₁_11_AFM
g₁_12_FM = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₂_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_12_2M = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_3_k[i, j]) *  (transpose(u_L_3_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₃_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_12_AFM = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_4_k[i, j]) *  (transpose(u_L_4_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₄_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_12 = g₁_12_FM .+ g₁_12_2M .+ g₁_12_AFM
g₁_21_FM = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₂_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_21_2M = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_3_k[i, j]) *  (transpose(u_L_3_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₃_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_21_AFM = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_4_k[i, j]) *  (transpose(u_L_4_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₄_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_21 = g₁_21_FM .+ g₁_21_2M .+ g₁_21_AFM
g₁_22_FM = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₂_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_22_2M = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_3_k[i, j]) *  (transpose(u_L_3_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₃_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_22_AFM = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_4_k[i, j]) *  (transpose(u_L_4_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ₁_values[i,j] - ϵ₄_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_22 = g₁_22_FM .+ g₁_22_2M .+ g₁_22_AFM
det_g₁ = g₁_11 .* g₁_22 .- g₁_12 .* g₁_21
tr_g₁ = g₁_11 .+ g₁_22
Δ_g = sqrt.( tr_g₁.^2 .- 4 .* det_g₁)

LD_AM = [2*real.((L̃_11_AM[i,j] + L̃_22_AM[i,j]) .* conj.(L̃_12_AM)[i,j]) for i in 1:Nx, j in 1:Ny]
LD_xx_yy_AM = [real.(conj.(L̃_11_AM[i,j]) .* L̃_11_AM[i,j] .- conj.(L̃_22_AM[i,j]) .* L̃_22_AM[i,j]) for i in 1:Nx, j in 1:Ny]

LD_AM_FM = [LD_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,2] element for the FM case
LD_AM_2M = [LD_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny] # only [1,3] element for the 2M case
LD_AM_AFM = [LD_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny] # only [1,4] element for the AFM case

LD_xx_yy_AM_FM = [LD_xx_yy_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,1] element for the FM case
LD_xx_yy_AM_2M = [LD_xx_yy_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny] # only [1,3] element for the 2M case
LD_xx_yy_AM_AFM = [LD_xx_yy_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny] # only [1,4] element for the AFM case

L̃_11_AM_FM = [L̃_11_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny]
L̃_11_AM_2M = [L̃_11_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny]
L̃_11_AM_AFM = [L̃_11_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny]

L̃_12_AM_FM = [L̃_12_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny]
L̃_12_AM_2M = [L̃_12_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny]
L̃_12_AM_AFM = [L̃_12_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny]

L̃_21_AM_FM = [L̃_21_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny]
L̃_21_AM_2M = [L̃_21_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny]
L̃_21_AM_AFM = [L̃_21_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny]

L̃_22_AM_FM = [L̃_22_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny]
L̃_22_AM_2M = [L̃_22_AM[i, j][1, 3] for i in 1:Nx, j in 1:Ny]
L̃_22_AM_AFM = [L̃_22_AM[i, j][1, 4] for i in 1:Nx, j in 1:Ny]

L̃_11_AM_FM_sq = [abs2(L̃_11_AM[i, j][1, 2]) for i in 1:Nx, j in 1:Ny]
L̃_11_AM_2M_sq = [abs2(L̃_11_AM[i, j][1, 3]) for i in 1:Nx, j in 1:Ny]
L̃_11_AM_AFM_sq = [abs2(L̃_11_AM[i, j][1, 4]) for i in 1:Nx, j in 1:Ny]

L̃_12_AM_FM_sq = [abs2(L̃_12_AM[i, j][1, 2]) for i in 1:Nx, j in 1:Ny]
L̃_12_AM_2M_sq = [abs2(L̃_12_AM[i, j][1, 3]) for i in 1:Nx, j in 1:Ny]
L̃_12_AM_AFM_sq = [abs2(L̃_12_AM[i, j][1, 4]) for i in 1:Nx, j in 1:Ny]

L̃_12_AM_2M_sq_low = [abs2(L̃_12_AM[i, j][2, 4]) for i in 1:Nx, j in 1:Ny]

L̃_21_AM_FM_sq = [abs2(L̃_21_AM[i, j][1, 2]) for i in 1:Nx, j in 1:Ny]
L̃_21_AM_2M_sq = [abs2(L̃_21_AM[i, j][1, 3]) for i in 1:Nx, j in 1:Ny]
L̃_21_AM_AFM_sq = [abs2(L̃_21_AM[i, j][1, 4]) for i in 1:Nx, j in 1:Ny]

L̃_22_AM_FM_sq = [abs2(L̃_22_AM[i, j][1, 2]) for i in 1:Nx, j in 1:Ny]
L̃_22_AM_2M_sq = [abs2(L̃_22_AM[i, j][1, 3]) for i in 1:Nx, j in 1:Ny]
L̃_22_AM_AFM_sq = [abs2(L̃_22_AM[i, j][1, 4]) for i in 1:Nx, j in 1:Ny]

total_intensity_AFM = L̃_11_AM_AFM_sq + L̃_22_AM_AFM_sq + 2 .* real.(conj.(L̃_12_AM_AFM) .* L̃_21_AM_AFM)
P_ne_se_AM_AFM = L̃_11_AM_AFM_sq .+ L̃_22_AM_AFM_sq .- 2 .* real.(conj.(L̃_11_AM_AFM) .* L̃_22_AM_AFM)
P_x_y_AM_AFM = L̃_22_AM_AFM_sq .+ L̃_11_AM_AFM_sq

# temporary 
Ω₁_J2_05 = Ω₁
Ω₁_J2_015 = Ω₁
Ω₁_J2_005 = Ω₁
Ω₁_J2_0025 = Ω₁
Ω₁_J2_0005 = Ω₁

g₁_11_topo = g₁_11
g₁_12_topo = g₁_12
g₁_21_topo = g₁_21
g₁_22_topo = g₁_22

g₁_11_AM = g₁_11
g₁_12_AM = g₁_12
g₁_21_AM = g₁_21
g₁_22_AM = g₁_22

LD_xx_yy_AM_AFM_positive = LD_xx_yy_AM_AFM
LD_xx_yy_AM_AFM_negative = LD_xx_yy_AM_AFM

# Chern
C₁ = sum(Ω₁) * dkx * dky / (2*pi)
# plot
function plot_heatmap_on_BZ(fig, kx, ky, Z; axis=[1,1], title="", titlesize=20, xlabel=L"k_x", ylabel=L"k_y", xlabelsize=18, ylabelsize=18, colormap=:viridis, coloarmaplabel="", xticklabelsize=16.0, yticklabelsize=16.0)
    ax = Axis(fig[axis[1], axis[2]], 
             title=title, titlesize=titlesize, aspect=DataAspect() , 
             xlabel=xlabel, ylabel=ylabel, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
             xticklabelsize=xticklabelsize, yticklabelsize=yticklabelsize)
    hm = heatmap!(ax, kx, ky, Z, colormap=colormap)
    colorbar_axis = axis+[0, 1]
    Colorbar(fig[colorbar_axis[1], colorbar_axis[2]], hm, label=coloarmaplabel, ticklabelsize=xticklabelsize)
    return ax, hm
end

f = Figure(size = (1200, 900))
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_FM; 
                                    axis=[1,1], title=L"\textbf{RCD (Interband component)} ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"χ")

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_2M; 
                                    axis=[1,3], title=L"\textbf{RCD (Two-magnon (same branch))} ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"χ") 

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_AFM; 
                                    axis=[1,5], title=L"\textbf{RCD (Two-magnon (different branch))}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"χ")

ax_heat_4, hm_4 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_FM; 
                                    axis=[2,1], title=L"\textbf{RLD (Interband component)} ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q")

ax_heat_5, hm_5 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_2M; 
                                    axis=[2,3], title=L"\textbf{RLD (Two-magnon (same branch))} ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q") 

ax_heat_6, hm_6 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_AFM; 
                                    axis=[2,5], title=L"\textbf{RLD (Two-magnon (different branch))}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q")

ax_heat_7, hm_7 = plot_heatmap_on_BZ(f, kx, ky, Ω₁; 
                                    axis=[3,1], title=L"\textbf{Berry curvature } Ω₊", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q")

ax_heat_8, hm_8 = plot_heatmap_on_BZ(f, kx, ky, g₁_12; 
                                    axis=[3,3], title=L"g_{+,xy}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q") 

ax_heat_9, hm_9 = plot_heatmap_on_BZ(f, kx, ky, tr_g₁; 
                                    axis=[3,5], title=L"tr(g_+) ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel=L"Q")
f

sum(χ_AM_FM.+χ_AM_2M.+χ_AM_AFM) * dkx * dky / (2*pi)^2
LD_AM_AFM[50,50]

save("julia/figures/altermagnet_RCD_RLD.png", f, px_per_unit = 300/96)

# 3D plot
f2 = Figure()
ax = Axis3(f2[1, 1], title="Magnon bands", azimuth=pi/4, elevation=pi/6, xlabel=L"k_x", ylabel=L"k_y", zlabel="Energy (meV)")
# surface!(kx, ky, )
surface!(kx, ky, ϵ₁_values .- ϵ₂_values)
f2


sum(Ω₁) * dkx * dky / (2*pi)

# Berry curvature
f = Figure(size = (1500, 250), figure_padding=0)
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, Ω₁_J2_05; 
                                    axis=[1,1], title=L"J_2=-0.5J_1", titlesize=24,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24,
                                    xticklabelsize=20, yticklabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="")
# Label(f[1,1][1, 1, TopLeft()], "(a)",
#                                     fontsize= 24,
#                                     padding = (-5, 20, 0, 0),)
Label(f[1,2][1, 1, Top()], L"\Omega_+",
                                    fontsize= 24,
                                    padding = (0, -10, 0, 0),)


ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, Ω₁_J2_015; 
                                    axis=[1,3], title=L"J_2=-0.15J_1", titlesize=24,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24,
                                    xticklabelsize=20, yticklabelsize=20,  
                                    colormap=:viridis, coloarmaplabel="")
# Label(f[1,3][1, 1, TopLeft()], "(b)",
#                                     fontsize= 24,
#                                     padding = (-5, 20, 0, 0),)
Label(f[1,4][1, 1, Top()], L"\Omega_+",
                                    fontsize= 24,
                                    padding = (0, -10, 0, 0),)

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, Ω₁_J2_005; 
                                    axis=[1,5], title=L"J_2=-0.05J_1", titlesize=24,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24,
                                    xticklabelsize=20, yticklabelsize=20,  
                                    colormap=:viridis, coloarmaplabel="")
# Label(f[1,5][1, 1, TopLeft()], "(c)",
#                                     fontsize= 24,
#                                     padding = (-5, 20, 0, 0),)
Label(f[1,6][1, 1, Top()], L"\Omega_+",
                                    fontsize= 24,
                                    padding = (0, -10, 0, 0),)

ax_heat_4, hm_4 = plot_heatmap_on_BZ(f, kx, ky, Ω₁_J2_0025; 
                                    axis=[1,7], title=L"J_2=-0.025J_1", titlesize=24,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24,
                                    xticklabelsize=20, yticklabelsize=20,  
                                    colormap=:viridis, coloarmaplabel="")
# Label(f[1,7][1, 1, TopLeft()], "(d)",
#                                     fontsize= 24,
#                                     padding = (-5, 20, 0, 0),)
Label(f[1,8][1, 1, Top()], L"\Omega_+",
                                    fontsize= 24,
                                    padding = (0, -10, 0, 0),)

ax_heat_5, hm_5 = plot_heatmap_on_BZ(f, kx, ky, Ω₁_J2_0005; 
                                    axis=[1,9], title=L"J_2=-0.005J_1", titlesize=24,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24,
                                    xticklabelsize=20, yticklabelsize=20,  
                                    colormap=:viridis, coloarmaplabel="")
# Label(f[1,9][1, 1, TopLeft()], "(e)",
#                                     fontsize= 24,
#                                     padding = (-5, 20, 0, 0),)
Label(f[1,10][1, 1, Top()], L"\Omega_+",
                                    fontsize= 24,
                                    padding = (0, -10, 0, 0),)
f

save("julia/figures/altermagnet_berry_curvature_J2_no_label.png", f, px_per_unit = 300/96)

g_RLD = (g₁_11_FM .* gap_12.+ g₁_11_2M .* gap_13 .+ g₁_11_AFM .* gap_14) .- (g₁_22_FM .* gap_12 .+ g₁_22_2M .* gap_13 .+ g₁_22_AFM .* gap_14)

# quantum metric
f = Figure(size = (800, 400), figure_padding=0)
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, g₁_11_topo; 
                                    axis=[1,1], title=L"g_{+,xx}", titlesize=20,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, g₁_12_topo; 
                                    axis=[1,3], title=L"g_{+,xy}", titlesize=20, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="")
                                    Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

Label(f[1,3][1, 1, TopLeft()], "(b)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, g₁_22_topo; 
                                    axis=[1,5], title=L" g_{+,yy}",  titlesize=20,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,5][1, 1, TopLeft()], "(c)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

ax_heat_4, hm_4 = plot_heatmap_on_BZ(f, kx, ky, (g₁_11_AM .+ g₁_22_AM) .* gap_12; 
                                    axis=[2,1], title=L"g_{+,xx}\Delta_{+\!\!-}^2", titlesize=20,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,1][1, 1, TopLeft()], "(d)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

ax_heat_5, hm_5 = plot_heatmap_on_BZ(f, kx, ky, g₁_12_AM .* gap_12; 
                                    axis=[2,3], title=L"g_{+,xy}\Delta_{+\!\!-}^2", titlesize=20, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,3][1, 1, TopLeft()], "(e)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)

ax_heat_6, hm_6 = plot_heatmap_on_BZ(f, kx, ky, g₁_22_AM .* gap_12; 
                                    axis=[2,5], title=L" g_{+,yy}\Delta_{+\!\!-}^2",  titlesize=20,
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=20, ylabelsize=20, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,5][1, 1, TopLeft()], "(f)",
                                    fontsize= 20,
                                    padding = (-5, 20, 0, 0),
)
f

save("julia/figures/topo_and_altermagnet_quantum_metrics.png", f, px_per_unit = 300/96)

# RCD & LCD by components
f = Figure(size = (1000, 800), figure_padding=0)

ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_FM; 
                                    axis=[1,1], title=L"\text{RCD } S^3_{12} ", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_2M; 
                                    axis=[1,3], title=L"\text{RCD }S^3_{13}", titlesize=24,  
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,3][1, 1, TopLeft()], "(b)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, χ_AM_AFM; 
                                    axis=[1,5], title=L"\text{RCD }S^3_{14}", titlesize=24,  
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,5][1, 1, TopLeft()], "(c)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_4, hm_4 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_FM; 
                                    axis=[2,1], title=L"\text{RLD }S^2_{12}",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,1][1, 1, TopLeft()], "(d)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_5, hm_5 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_2M; 
                                    axis=[2,3], title=L"\text{RLD }S^2_{13}",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
) 
Label(f[2,3][1, 1, TopLeft()], "(e)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_6, hm_6 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_AFM; 
                                    axis=[2,5], title=L"\text{RLD }S^3_{14}", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,5][1, 1, TopLeft()], "(f)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_7, hm_7 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_FM; 
                                    axis=[3,1], title=L"\text{RLD }S^1_{12}",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[3,1][1, 1, TopLeft()], "(g)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_8, hm_8 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_2M; 
                                    axis=[3,3], title=L"\text{RLD }S^1_{13}",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
) 
Label(f[3,3][1, 1, TopLeft()], "(h)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_9, hm_9 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_AFM; 
                                    axis=[3,5], title=L"\text{RLD }S^1_{14}", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[3,5][1, 1, TopLeft()], "(i)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

f

save("julia/figures/altermagnet_RCD_RLD_components.png", f, px_per_unit = 300/96)

# RLD in altermangetic limit
f = Figure(size = (600, 225), figure_padding=0)

ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_AFM_positive; 
                                    axis=[1,1], title=L"J_2=+0.05J_1",  titlesize=18, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 18,
                                    padding = (-5, 20, 0, 0),
)
ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_AFM_negative; 
                                    axis=[1,3], title=L"J_2=-0.05J_1",  titlesize=18, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,3][1, 1, TopLeft()], "(b)",
                                    fontsize= 18,
                                    padding = (-5, 20, 0, 0),
)
f

save("julia/figures/altermagnet_RLD_xx_yy_J2_flipped_sign.png", f, px_per_unit = 300/96)

# LMCs 
f = Figure(size = (1000, 800), figure_padding=0)

ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, L̃_11_AM_FM_sq; 
                                    axis=[1,1], title=L"|\tilde{L}_{xx,12}|^2", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, L̃_11_AM_2M_sq; 
                                    axis=[1,3], title=L"|\tilde{L}_{xx,13}|^2", titlesize=24,  
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,3][1, 1, TopLeft()], "(b)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, L̃_11_AM_AFM_sq; 
                                    axis=[1,5], title=L"|\tilde{L}_{xx,14}|^2", titlesize=24,  
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,5][1, 1, TopLeft()], "(c)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_4, hm_4 = plot_heatmap_on_BZ(f, kx, ky, L̃_12_AM_FM_sq; 
                                    axis=[2,1], title=L"|\tilde{L}_{xy,12}|^2",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,1][1, 1, TopLeft()], "(d)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)


ax_heat_5, hm_5 = plot_heatmap_on_BZ(f, kx, ky, L̃_12_AM_2M_sq; 
                                    axis=[2,3], title=L"|\tilde{L}_{xy,13}|^2",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
) 
Label(f[2,3][1, 1, TopLeft()], "(e)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_6, hm_6 = plot_heatmap_on_BZ(f, kx, ky, L̃_12_AM_AFM_sq; 
                                    axis=[2,5], title=L"|\tilde{L}_{xy,14}|^2", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[2,5][1, 1, TopLeft()], "(f)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_7, hm_7 = plot_heatmap_on_BZ(f, kx, ky, L̃_22_AM_FM_sq; 
                                    axis=[3,1], title=L"|\tilde{L}_{yy,12}|^2",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[3,1][1, 1, TopLeft()], "(g)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_8, hm_8 = plot_heatmap_on_BZ(f, kx, ky, L̃_22_AM_2M_sq; 
                                    axis=[3,3], title=L"|\tilde{L}_{yy,13}|^2",  titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
) 
Label(f[3,3][1, 1, TopLeft()], "(h)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_heat_9, hm_9 = plot_heatmap_on_BZ(f, kx, ky,  P_ne_se_AM_AFM; 
                                    axis=[3,5], title=L"|\tilde{L}_{yy,14}|^2", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:jet, coloarmaplabel="",
)
Label(f[3,5][1, 1, TopLeft()], "(i)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)
f

# frequency-resolved signal
function gaussian_1D(ω ; ω₀=10, σ=0.1)
    # Gaussian function for smoothing
    return 1/(sqrt(2*pi)*σ) * exp(-((ω-ω₀)^2) / (2 * σ^2))
end

function boltzman_factor(ε, T=2)
    # Boltzmann factor for thermal occupation;
    # ε is the energy in meV, T is the temperature in Kelvin
    k_b = 0.08617333262145 # meV/K, Boltzmann constant
    β = 1 / (k_b * T)
    return exp(-β * ε)
end

function bose_einstein_distribution(ε, T=2)
    # Bose-Einstein distribution for thermal occupation of bosonic particles;
    # ε is the energy in meV, T is the temperature in Kelvin
    k_b = 0.08617333262145 # meV/K, Boltzmann constant
    β = 1 / (k_b * T)
    return 1 / (exp(β * ε) - 1)
end

Z₁(kx, ky, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = bose_einstein_distribution(ω₊(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), T) + 1
Z₂(kx, ky, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = bose_einstein_distribution(ω₋(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), T) + 1

function partition(T; Nx=100, Ny=100, J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1)
    # Partition function for the two-band bosonic model
    # T is the temperature in Kelvin
    kx = range(-π/2, π/2, length=Nx)
    ky = range(-π/2, π/2, length=Ny)
    dkx = step(kx)
    dky = step(ky)
    Z₋ = [Z₁(kx[i], ky[j], T; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
    Z₊ = [Z₂(kx[i], ky[j], T; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
    lnZ₋ = log.(Z₋)
    lnZ₊ = log.(Z₊)
    lnZ = sum(lnZ₋)*dkx*dky/(2*pi)^2 + sum(lnZ₊)*dkx*dky/(2*pi)^2
    return exp(lnZ) 
end
findall(x -> abs(x-X[1]) < dkx , kx)

ω = range(0, 20, length=Int(2e2+1))
dω = step(ω)
δ(ω) = gaussian_1D(ω; ω₀=0, σ=1.5*dω)# Dirac delta function approximation
δ_approx = δ.(ω) 

T_neel = 60 # Néel temperature in Kelvin
temperatures = range(0, T_neel, length=13)
Z = partition.(temperatures; Nx=Nx, Ny=Ny,  J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)

Δ_FM(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ω₊(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)-ω₋(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)
Δ_2M_up(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = 2 * ω₊(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)
Δ_2M_low(kx, ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = 2 * ω₋(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)
Δ_AFM(kx,ky; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = ω₊(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) + ω₋(kx ,ky; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)

Δ_FM_values = [Δ_FM(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
Δ_2M_up_values = [Δ_2M_up(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
Δ_2M_low_values = [Δ_2M_low(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]
Δ_AFM_values = [Δ_AFM(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:Nx, j in 1:Ny]

boltzman_factor_lower(T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = [boltzman_factor(ω₊(kx[i], ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), T) for i in 1:Nx, j in 1:Ny]
boltzman_factor_upper(T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = [boltzman_factor(ω₋(kx[i] , ky[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a), T) for i in 1:Nx, j in 1:Ny]

Q_FM(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( (boltzman_factor_lower(T; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) ) .* LD_AM_FM .*  δ.(ω .- Δ_FM_values ) ./ partition(T; Nx=Nx, Ny=Ny, J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)) * dkx * dky / (2*pi)^2
Q_2M(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  LD_AM_2M .*  δ.(ω .- Δ_2M_values ) ) * dkx * dky / (2*pi)^2
Q_AFM(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  LD_AM_AFM .*  δ.(ω .- Δ_AFM_values ) ) * dkx * dky / (2*pi)^2

Q_xx_yy_FM(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( (boltzman_factor_lower(T; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) ) .* LD_xx_yy_AM_FM .*  δ.(ω .- Δ_FM_values) ./ partition(T; Nx=Nx, Ny=Ny, J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a)) * dkx * dky / (2*pi)^2
Q_xx_yy_2M(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  LD_xx_yy_AM_2M .*  δ.(ω .- Δ_2M_values) ) * dkx * dky / (2*pi)^2
Q_xx_yy_AFM(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( LD_xx_yy_AM_AFM .*  δ.(ω .- Δ_AFM_values) ) * dkx * dky / (2*pi)^2

Q_ne_we_AFM(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( P_ne_se_AM_AFM .*  δ.(ω .- Δ_AFM_values)  ) * dkx * dky / (2*pi)^2
Q_ne_we_AFM_M(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( P_ne_se_AM_AFM[100, 100] .*  δ.(ω .- Δ_AFM_values[100, 100]) ) * dkx * dky / (2*pi)^2
Q_ne_we_AFM_mid(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( P_ne_se_AM_AFM[50, 75] .*  δ(ω .- Δ_AFM_values[50, 75]) ) * dkx * dky / (2*pi)^2
Q_ne_we_AFM_X(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( P_ne_se_AM_AFM[50, 100] .*  δ.(ω .- Δ_AFM_values[50, 100]) ) * dkx * dky / (2*pi)^2
Q_ne_we_AFM_ΓX(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum( P_ne_se_AM_AFM[50, 50:end] .*  δ.(ω .- Δ_AFM_values[50, 50:end]) ) * dkx * dky / (2*pi)^2

L_xy_2M_up(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  L̃_12_AM_2M_sq .*  δ.(ω .- Δ_2M_up_values) ) * dkx * dky / (2*pi)^2
L_xy_2M_low(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  L̃_12_AM_2M_sq_low .*  δ.(ω .- Δ_2M_low_values) ) * dkx * dky / (2*pi)^2

AFM_dos(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  δ.(ω .- Δ_AFM_values) ) * dkx * dky / (2*pi)^2

I(ω, T; J₁=1, J₂=0, D=0.1, S=3/2, s=0.5, c=√(3)/2, a=1) = sum(  total_intensity_AFM .*  δ.(ω .- Δ_AFM_values) ) * dkx * dky / (2*pi)^2
# Q₀(ω) = sum( LD_AM_AFM .*  δ.(ω .- Δ_αβ)) * dkx * dky / (2*pi)^2

#Q_ω_T = [Q(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_xx_yy_AFM_ω_T = [Q_xx_yy_AFM(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
I_ω_T = [I(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_ne_we_ω_T = [Q_ne_we_AFM(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_ne_we_ω_T_M = [Q_ne_we_AFM_M(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_ne_we_ω_T_mid = [Q_ne_we_AFM_mid(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_ne_we_ω_T_X = [Q_ne_we_AFM_X(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_ne_we_ω_T_ΓX = [Q_ne_we_AFM_ΓX(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
#L_xy_2M_up_ω_T = [L_xy_2M_up(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
#L_xy_2M_low_ω_T = [L_xy_2M_low(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
AFM_dos_ω_T = [AFM_dos(ω[i], temperatures[j]; J₁=J₁, J₂=J₂, D=D, S=S, s=s, c=c, a=a) for i in 1:length(ω), j in 1:length(temperatures)]

# energy dos_line
energy_range = range(0, 10, length=Int(2e2+1))
δ(ω) = gaussian_1D(ω; ω₀=0, σ=2.5*step(energy_range))# Dirac delta function approximation
δ_approx_energy_low = [sum(δ.(energy_range[i] .- ω₋_values)) for i in 1:length(energy_range)]
δ_approx_energy_up = [sum(δ.(energy_range[i] .- ω₊_values)) for i in 1:length(energy_range)]

# temp
L_xy_2M_up_ω_T_topo = L_xy_2M_up_ω_T[:, 1]
L_xy_2M_low_ω_T_topo = L_xy_2M_low_ω_T[:, 1]

f = Figure(size=(1100, 800), figure_padding=0)
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, P_ne_se_AM_AFM; 
                                    axis=[1,1], title=L"P_{\textbf{k}, 14}^{\nearrow\searrow}", titlesize=24, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=24, ylabelsize=24, 
                                    colormap=:viridis, coloarmaplabel="",
)
Label(f[1,1][1, 1, TopLeft()], "(a)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)
poly!(ax_heat_1, Point2f[Γ, X,], color=(:white, 0), alpha=0.5, strokecolor=:white, strokewidth=3)
Label(f[1,1][2, 2, TopLeft()], "Γ",
                                    fontsize= 24,
                                    padding = (0, 40, 0, 0),
                                    color = :white,
)
Label(f[1,1][1, 2, Left()], "X",
                                    fontsize= 24,
                                    padding = (0, 40, 100, 0),
                                    color = :white,
)

ax = Axis(f[1,3];
    title = L"\textbf{Polarization-resolved scattering intesity $I_{14}^{\!\nearrow\!\!\searrow\!}(\omega)$}",
    xlabel = L"\hbar\omega/(E_X^+ + E_X^-)",
    ylabel = L"\text{Intensity (a.u.)}",
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0, 0),
    xlabelsize = 20,
    ylabelsize = 20,
    titlesize = 20,
)
##for i in 1:length(temperatures)
Q_ne_we_line = lines!(ax, ω ./ Δ_AFM_values[50, 100], Q_ne_we_ω_T[:, 1] ./ I_ω_T[:, 1], color = (:red, 13/13), linewidth = 1)
#dos_line = lines!(ax, ω./ Δ_AFM_values[50, 100], AFM_dos_ω_T[:, 1], color= (:blue, 13/13), linewidth = 1)
#end
Label(f[1,3][1, 1, TopLeft()], "(b)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)
#Legend(f[1,2], [upper_line, lower_line], [L"2\epsilon_{+}", L"2\epsilon_{-}"]; labelsize=20 ,orientation = :vertical, title = "Bands", framevisible = false)
colsize!(f.layout, 1, Fixed(300))

ax_dos = Axis(f[2,1:2];
    title = L"\textbf{2DOS}",
    xlabel = L"\hbar\omega/(E_X^+ + E_X^-)",
    ylabel = L"\text{ρ (a.u.)}",
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0, 0),
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize = 20,
)
dos_line = lines!(ax_dos, ω./ Δ_AFM_values[50, 100], AFM_dos_ω_T[:, 1], color= (:blue, 13/13), linewidth = 1)
Label(f[2,1][1, 1, TopLeft()], "(c)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)

ax_line = Axis(f[2,3];
    title = L"\textbf{$I_{14}^{\!\nearrow\!\!\searrow\!}(\omega)$ contributed from different $\mathbf{k}$-points}",
    xlabel = L"\hbar\omega/(E_X^+ + E_X^-)",
    ylabel = L"\text{Intensity (a.u.)}",
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0, 0),
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize = 20,
)
# energy_dos_line_low = lines!(ax_line, energy_range, δ_approx_energy_low[:,], color= (:blue, 13/13), linewidth = 1)
# energy_dos_line_up = lines!(ax_line, energy_range, δ_approx_energy_up[:,], color= (:red, 13/13), linewidth = 1)
Q_ne_we_M_line = lines!(ax_line, ω ./ Δ_AFM_values[50, 100], Q_ne_we_ω_T_M[:, 1] ./ I_ω_T[:, 1], color = (:green, 13/13), linewidth = 1)
Q_ne_we_mid_line = lines!(ax_line, ω ./ Δ_AFM_values[50, 100], Q_ne_we_ω_T_mid[:, 1] ./ I_ω_T[:, 1], color = (:red, 13/13), linewidth = 1)
Q_ne_we_X_line = lines!(ax_line, ω ./ Δ_AFM_values[50, 100], Q_ne_we_ω_T_X[:, 1] ./ I_ω_T[:, 1], color = (:blue, 13/13), linewidth = 1)
Q_ne_we_ΓX_line = lines!(ax_line, ω ./ Δ_AFM_values[50, 100], Q_ne_we_ω_T_ΓX[:, 1] ./ I_ω_T[:, 1], color = (:orange, 13/13), linewidth = 1)
axislegend(ax_line,[Q_ne_we_M_line, Q_ne_we_X_line, Q_ne_we_mid_line, Q_ne_we_ΓX_line], ["M", "X", "Δ", L"ΓX \text{ path}"]; orientation = :vertical, framevisible = true, labelsize=18)
Label(f[2,3][1, 1, TopLeft()], "(d)",
                                    fontsize= 24,
                                    padding = (-5, 25, 0, 0),
)
f

findmax(L_xy_2M_low_ω_T)
findmax(L_xy_2M_up_ω_T)
findmax(I_ω_T)
findmax(ω₋_values)[1] *2
ω[5]
maximum(AFM_dos_ω_T)

save("julia/figures/altermagnet_frequency_resolved_P_ne_sw_AFM.png", f, px_per_unit = 300/96)   