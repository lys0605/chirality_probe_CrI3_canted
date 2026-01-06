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

Aₖ(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = S * (2*J_nm*cos(2*kx*a)+2*J_d*cos(2*ky*a) - 2*(J_d + J_nm) + 4*J_ab + 2*K_z) + B_z
Dₖ(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = S * (2*J_nm*cos(2*ky*a)+2*J_d*cos(2*kx*a) - 2*(J_d + J_nm) + 4*J_ab + 2*K_z) - B_z
Bₖ(kx, ky; J_ab=1, S=3/2, a=1) = 2*S*J_ab*(cos(kx*a+ky*a) + cos(kx*a - ky*a))

a = 1
S = 3/2
J_ab = 1
J_nm = -0.5*J_ab
J_d = -0*J_ab
K_z = 0.01*J_ab
B_z = 0

H_p4mm(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = [Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) Bₖ(kx, ky; J_ab=J_ab, S=S, a=a)
Bₖ(kx, ky; J_ab=J_ab, S=S, a=a) Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)]


function H_p4mm(k_vec; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1)
    kx, ky = k_vec
    return [Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) Bₖ(kx, ky; J_ab=J_ab, S=S, a=a)
    Bₖ(kx, ky; J_ab=J_ab, S=S, a=a) Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)]
end

ω_α(kx ,ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = 0.5 * (Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) - Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) + sqrt((Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) + Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))^2 - 4*Bₖ(kx, ky; J_ab=J_ab, S=S, a=a)^2))
ω_β(kx ,ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = 0.5 * (-Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) + Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) + sqrt((Aₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) + Dₖ(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))^2 - 4*Bₖ(kx, ky; J_ab=J_ab, S=S, a=a)^2))

# high-symmetry points (a₁=2a x̂, a₂=2a ŷ; b₁=π/a x̂, b₂=π/a ŷ; hence M=(π/2a, π/2a), X=(0, π/2a), Y=(π/2a, 0))
Γ = [0.0, 0.0]
M = [pi, pi]/(2*a) 
X = [0.0, pi]/(2*a) 
Y = [pi, 0.0]/(2*a)

# symmetry lines
# get k-vectors ΓMXΓYMΓ
k0 = get_kvectors(Γ, M, n=100)
k1 = get_kvectors(M, X, n=51)
k2 = get_kvectors(X, Γ, n=51)
k3 = get_kvectors(Γ, Y, n=51)
k4 = get_kvectors(Y, M, n=51)
k5 = get_kvectors(M, Γ, n=101)

k_vectors = group_kvectors(k0, k1, k2, k3, k4, k5)
path = sqrt.(k_vectors[1].^2 .+ k_vectors[2].^2)
path_index = get_path_index(k0, k1, k2, k3, k4, k5)
k_labels = ["Γ", "M", "X", "Γ", "Y", "M", "Γ"]

k = [[x, y] for (x,y) in zip(k_vectors[1],k_vectors[2])]
σ₃ = [1 0 ; 0 -1]
H_eff_path = [σ₃ * H_p4mm(k[i]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(k)]
H_eff_path_eig = [eigen(H_eff_path[i]) for i in 1:length(k)]
eigenvalues_path = [real(H_eff_path_eig[i].values) for i in 1:length(k)] # eigenvalues are sorted from largest to smallest
eigenvectors_path = [H_eff_path_eig[i].vectors for i in 1:length(k)]
ω_α_path = [ω_α(k[i][1], k[i][2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(k)]
ω_β_path = [ω_β(k[i][1], k[i][2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(k)]

E₁_path = [-eigenvalues_path[i][1] for i in 1:length(k)]
E₂_path = [eigenvalues_path[i][2] for i in 1:length(k)]

u1_path = getindex.(eigenvectors_path, :, 1)
u2_path = getindex.(eigenvectors_path, :, 2)

s1_z =  [abs(u1_path[i][1])^2 - abs(u1_path[i][2])^2 for i in 1:length(k)]
s2_z = [abs(u2_path[i][1])^2  - abs(u2_path[i][2])^2 for i in 1:length(k)]

#plots path
f = Figure(size=(800, 400))
ax = Axis(f[1,1];
    title = L"\textbf{Altermagnet Band Structure p4mm} ",
    xlabel = L"k (1/Å)",
    ylabel = L"\text{Energy (meV)}",
    xticks = (path_index, k_labels),
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0, 0)
)
line_upper = lines!(ax, 1:length(path), E₁_path./1.5 , color = (:red, 0.5), linewidth = 1)
line_lower = lines!(ax, 1:length(path), E₂_path./1.5, color = (:blue, 0.5), linewidth = 1) #
Legend(f[1,2], [line_upper, line_lower], [L"ϵ_α ", L"ϵ_β"]; labelsize=20 ,orientation = :vertical, title = "Bands", framevisible = false)
f
# Label(f[1,1][1, 1, TopLeft()], "(a)",
#     fontsize= 20,
#     padding = (0, 10, 0 ,0),)

# Derivatives of the Hamiltonian
H₁(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(kx -> H_p4mm(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), kx)
H₂(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(ky -> H_p4mm(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), ky)
H₁₁(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(kx -> H₁(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), kx)
H₁₂(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(ky -> H₁(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), ky)
H₂₁(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(kx -> H₂(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), kx)
H₂₂(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = ForwardDiff.derivative(ky -> H₂(kx, ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), ky)

# Define the number of points in kx and ky
Nx = 200
Ny = 200

# Define the kx and ky ranges
kx = range(-π/(2*a), π/(2*a), length=Nx)
ky = range(-π/(2*a), π/(2*a), length=Ny)
dkx = step(kx)
dky = step(ky)

# parameters 
a = 1
S = 3/2
J_ab = 1
J_nm = -0.5*J_ab
J_d = -0.0*J_ab
K_z = 0.01*J_ab
B_z = 0

H_AM =  [H_p4mm(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
H_AM_eff = [σ₃ * H_p4mm(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
H_AM_eff_eig = [eigen(H_AM_eff[i, j]) for i in 1:Nx, j in 1:Ny] # eigen systems
H_AM_eff_eigenvalues = [H_AM_eff_eig[i, j].values for i in 1:Nx, j in 1:Ny] # Apply the τ₃ transformation to the eigenvalues
H_AM_eff_eigenvectors = [H_AM_eff_eig[i, j].vectors for i in 1:Nx, j in 1:Ny]

σ₁ = [0 1; 1 0]

# sorted from smalles to largest
ϵ_α_values = real.(getindex.(H_AM_eff_eigenvalues, 2))
ϵ_β_values = -real.(getindex.(H_AM_eff_eigenvalues, 1))
gap_αβ = [(ϵ_α_values[i,j] - ϵ_β_values[i,j])^2 for i in 1:Nx, j in 1:Ny]

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


# numerical 
U_AM_k = [pseudounitary_normalization(H_AM_eff_eigenvectors[i, j], σ₃) for i in 1:Nx, j in 1:Ny]
U_AM_k = [U_AM_k[i,j] * σ₁ for i in 1:Nx, j in 1:Ny] # swapping the columns to have the eigenvalues sorted from largest to smallest
U_AM_inv_k = [σ₃ * U_AM_k[i, j]' * σ₃ for i in 1:Nx, j in 1:Ny] # inverse of the pseudounitary matrix
U_AM_dag_k = [U_AM_k[i, j]' for i in 1:Nx, j in 1:Ny] # Hermitian conjugate of the pseudounitary matrix
u_L_1_k = [U_AM_inv_k[i, j][1, :] for i in 1:Nx, j in 1:Ny]
u_L_2_k = [U_AM_inv_k[i, j][2, :] for i in 1:Nx, j in 1:Ny]
u_R_1_k = [U_AM_k[i, j][:, 1] for i in 1:Nx, j in 1:Ny]
u_R_2_k = [U_AM_k[i, j][:, 2] for i in 1:Nx, j in 1:Ny]

L̃_11_AM = [U_AM_dag_k[i, j] * H₁₁(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_12_AM = [U_AM_dag_k[i, j] * H₁₂(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_21_AM = [U_AM_dag_k[i, j] * H₂₁(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]
L̃_22_AM = [U_AM_dag_k[i, j] * H₂₂(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) * U_AM_k[i, j] for i in 1:Nx, j in 1:Ny]

χ_AM = 2*imag.((L̃_11_AM - L̃_22_AM) .* conj.(L̃_12_AM))
χ_AM_AFM = [χ_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,4] element for the AFM case

H₁_values = [H₁(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
H₂_values = [H₂(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
Ω₁ = [2 * imag( (transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ_α_values[i,j] - ϵ_β_values[i,j])^2) for i in 1:Nx, j in 1:Ny]
Ω₂ = [2 * imag( (transpose(u_L_2_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) *  (transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_2_k[i, j]) / (ϵ_β_values[i,j] - ϵ_α_values[i,j])^2) for i in 1:Nx, j in 1:Ny]

g₁_11 = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ_α_values[i,j] - ϵ_β_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_12 = [-real((transpose(u_L_1_k[i, j]) * H₁_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ_α_values[i,j] - ϵ_β_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_21 = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₁_values[i, j] * u_R_1_k[i, j]) / (ϵ_α_values[i,j] - ϵ_β_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]
g₁_22 = [-real((transpose(u_L_1_k[i, j]) * H₂_values[i, j] * u_R_2_k[i, j]) *  (transpose(u_L_2_k[i, j]) * H₂_values[i, j] * u_R_1_k[i, j]) / (ϵ_α_values[i,j] - ϵ_β_values[i,j])^2)  for i in 1:Nx, j in 1:Ny]

det_g₁ = g₁_11 .* g₁_22 .- g₁_12 .* g₁_21
tr_g₁ = g₁_11 .+ g₁_22
Δ_g = sqrt.( tr_g₁.^2 .- 4 .* det_g₁)

LD_AM = [2*real.((L̃_11_AM[i, j] + L̃_22_AM[i, j]) .* conj.(L̃_12_AM[i,j])) for i in 1:Nx, j in 1:Ny]
LD_xx_yy_AM = [real.(conj.(L̃_11_AM[i, j]) .* L̃_11_AM[i, j] .- conj.(L̃_22_AM[i, j]) .* L̃_22_AM[i, j]) for i in 1:Nx, j in 1:Ny]

LD_AM_AFM = [LD_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case
LD_xx_yy_AM_AFM = [LD_xx_yy_AM[i, j][1, 2] for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case

L_11 = [real.(conj.(L̃_11_AM[i,j]) .* L̃_11_AM[i,j]) for i in 1:Nx, j in 1:Ny]
L_11_AFM = [abs(L_11[i, j][1, 2]) for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case
L_22 = [real.(conj.(L̃_22_AM[i,j]) .* L̃_22_AM[i,j]) for i in 1:Nx, j in 1:Ny]
L_22_AFM = [abs(L_22[i, j][1, 2]) for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case
L_12 = [real.(conj.(L̃_12_AM[i,j]) .* L̃_12_AM[i,j]) for i in 1:Nx, j in 1:Ny]
L_12_AFM = [abs(L_12[i, j][1, 2]) for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case
L_21 = [real.(conj.(L̃_21_AM[i,j]) .* L̃_21_AM[i,j]) for i in 1:Nx, j in 1:Ny]
L_21_AFM = [abs(L_21[i, j][1, 2]) for i in 1:Nx, j in 1:Ny] # only [1,2] element for the AFM case
total_intensity = L_11_AFM .+ L_22_AFM .+ 2*L_12_AFM
# plot
function plot_heatmap_on_BZ(fig, kx, ky, Z; axis=[1,1], title="", titlesize=20, xlabel=L"k_x", ylabel=L"k_y", xlabelsize=18, ylabelsize=18, colormap=:viridis, coloarmaplabel="")
    ax = Axis(fig[axis[1], axis[2]], title=title, titlesize=titlesize, aspect=DataAspect() , xlabel=xlabel, ylabel=ylabel, xlabelsize=xlabelsize, ylabelsize=ylabelsize)
    hm = heatmap!(ax, kx, ky, Z, colormap=colormap)
    colorbar_axis = axis+[0, 1]
    Colorbar(fig[colorbar_axis[1], colorbar_axis[2]], hm, label=coloarmaplabel,)
    return ax, hm
end

# Geometric quantities
#  Energy bands
f = Figure(size = (900, 225))
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, ϵ_α_values; 
                                    axis=[1,1], title=L"\textbf{Energy band} ϵ_α", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, ϵ_β_values; 
                                    axis=[1,3], title=L"\textbf{Energy band } ϵ_β", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, gap_αβ; 
                                    axis=[1,5], title=L"\textbf{Band gap } \Delta_{αβ}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
f

# Geometric quantities
f = Figure(size = (600, 450))
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, g₁_11; 
                                    axis=[1,1], title=L"\textbf{Quantum metric } g_{α,11}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, g₁_12; 
                                    axis=[1,3], title=L"\textbf{Quantum metric } g_{α,12}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, g₁_21; 
                                    axis=[2,1], title=L"\textbf{Quantum metric } g_{α,21}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, g₁_22; 
                                    axis=[2,3], title=L"\textbf{Quantum metric } g_{α,22}", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
f

# LD
f = Figure(size = (600, 450))
ax_heat_1, hm_1 = plot_heatmap_on_BZ(f, kx, ky, LD_AM_AFM ./ (L_11_AFM .+ L_22_AFM .+ 2*L_12_AFM) ; 
                                    axis=[1,1], title=L"\textbf{RLD (diagonal)} ",  titlesize=18, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:jet, coloarmaplabel=L"Q")

ax_heat_2, hm_2 = plot_heatmap_on_BZ(f, kx, ky, LD_xx_yy_AM_AFM ./ (L_11_AFM .+ L_22_AFM .+ 2*L_12_AFM) ; 
                                    axis=[1,3], title=L"\textbf{RLD (vertical-horizontal))}", titlesize=18, 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:jet, coloarmaplabel=L"Q")

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, g₁_12 .* gap_αβ; 
                                    axis=[2,1], title=L" -g_{α,12}\Delta_{αβ}^2 ", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:jet, coloarmaplabel="")

ax_heat_3, hm_3 = plot_heatmap_on_BZ(f, kx, ky, (g₁_11 .- g₁_22 ).* gap_αβ ; 
                                    axis=[2,3], title=L"(g_{α,11}-g_{α,22})\Delta_{αβ}^2", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:jet, coloarmaplabel="")
f

# frequency-resolved plot
function gaussian_1D(ω ; ω₀=10, σ=0.1)
    # Gaussian function for smoothing
    return exp(-((ω-ω₀)^2) / (2 * σ^2))
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

Z₁(kx, ky, T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = bose_einstein_distribution(ω_α(kx ,ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), T) + 1
Z₂(kx, ky, T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = bose_einstein_distribution(ω_β(kx ,ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), T) + 1

function partition(T; Nx=100, Ny=100, J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1)
    # Partition function for the two-band bosonic model
    # T is the temperature in Kelvin
    kx = range(-π/2, π/2, length=Nx)
    ky = range(-π/2, π/2, length=Ny)
    dkx = step(kx)
    dky = step(ky)
    Z₋ = [Z₁(kx[i], ky[j], T; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
    Z₊ = [Z₂(kx[i], ky[j], T; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
    lnZ₋ = log.(Z₋)
    lnZ₊ = log.(Z₊)
    lnZ = sum(lnZ₋)*dkx*dky/(2*pi)^2 + sum(lnZ₊)*dkx*dky/(2*pi)^2
    return exp(lnZ) 
end

δ(ω) = gaussian_1D(ω; ω₀=0, σ=5e-2)# Dirac delta function approximation
ω = range(0, 5, length=Int(1e2+1))
δ_approx = δ.(ω) 

T_neel = 60 # Néel temperature in Kelvin
temperatures = range(0, T_neel, length=13)
Z = partition.(temperatures; Nx=Nx, Ny=Ny, J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)
lower_band_population_Γ =  boltzman_factor.(Ref((ω_β(Γ[1] ,Γ[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z
lower_band_population_X = boltzman_factor.(Ref((ω_β(X[1] ,X[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z
lower_band_population_Y = boltzman_factor.(Ref((ω_β(Y[1] ,Y[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z
upper_band_population_Γ =  boltzman_factor.(Ref((ω_α(Γ[1] ,Γ[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z
upper_band_population_X = boltzman_factor.(Ref((ω_α(X[1] ,X[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z
upper_band_population_Y = boltzman_factor.(Ref((ω_α(Y[1] ,Y[2]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))), temperatures) ./ Z

# frequency-resolved thermal RCD 
a = 1
S = 3/2
J_ab = 1
J_nm = -0.5*J_ab
J_d = -0.1*J_ab
K_z = 0.01*J_ab
B_z = 0

Δ(kx, ky; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = real(ω_α(kx ,ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)+ω_β(kx ,ky; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a))

ω_αβ = [Δ(kx[i], ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:Nx, j in 1:Ny]
boltzman_factor_lower(T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = [boltzman_factor(ω_β(kx[i] ,ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), T) for i in 1:Nx, j in 1:Ny]
boltzman_factor_upper(T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = [boltzman_factor(ω_α(kx[i] ,ky[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a), T) for i in 1:Nx, j in 1:Ny]

Q(ω, T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = sum( (boltzman_factor_lower(T; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) ) .* LD_AM_AFM .*  δ.(ω .- ω_αβ) ./ partition(T; Nx=Nx, Ny=Ny, J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)) * dkx * dky / (2*pi)^2
Q_xx_yy(ω, T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = sum( (boltzman_factor_lower(T; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) ) .* LD_xx_yy_AM_AFM .*  δ.(ω .- ω_αβ) ./ partition(T; Nx=Nx, Ny=Ny, J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)) * dkx * dky / (2*pi)^2
I(ω, T; J_ab=1, J_nm=-0.5, J_d=0, K_z=0.01, B_z=0, S=3/2, a=1) = sum( (boltzman_factor_lower(T; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) ) .* total_intensity.*  δ.(ω .- ω_αβ) ./ partition(T; Nx=Nx, Ny=Ny, J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a)) * dkx * dky / (2*pi)^2
# Q₀(ω) = sum( LD_AM_AFM .*  δ.(ω .- Δ_αβ)) * dkx * dky / (2*pi)^2
Q_ω_T = [Q(ω[i], temperatures[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
Q_xx_yy_ω_T = [Q_xx_yy(ω[i], temperatures[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(ω), j in 1:length(temperatures)]
I_ω_T = [I(ω[i], temperatures[j]; J_ab=J_ab, J_nm=J_nm, J_d=J_d, K_z=K_z, B_z=B_z, S=S, a=a) for i in 1:length(ω), j in 1:length(temperatures)]

f = Figure(size=(800, 400))
ax = Axis(f[1,1];
    title = L"\textbf{Frequencyresolved RLD}",
    xlabel = L"\text{frequency } \omega",
    ylabel = L"\text{Intensity (a.u.)}",
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0, 0),
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize = 20,
)
for i in 1:length(temperatures)
    lines!(ax, ω, Q_xx_yy_ω_T[:, i] , color = (:blue, i/13), linewidth = 1)
end
# line_5T= lines!(ax, 1:length(ω), Q_ω_T[:, ] , color = (:red, 0.5), linewidth = 1)
# line_10T = lines!(ax, 1:length(ω), Q_ω_T[:, 3], color = (:blue, 0.5), linewidth = 1) #
# Legend(f[1,2], [line_5T, line_10T], [L"T=5K ", L"T=10K"]; labelsize=20 ,orientation = :vertical, title = "Bands", framevisible = false)
f

save("frequency_resolved_RLD_p4mm_AFM_Jd-0.1_Kz0.01_Tneel60.png", f)