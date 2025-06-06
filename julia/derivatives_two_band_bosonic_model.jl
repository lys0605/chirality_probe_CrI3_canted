using LinearAlgebra, ForwardDiff
using CairoMakie, GeometryBasics
import Meshes

function two_band_bosonic_model(kx, ky)
    # Construct the two-band model Hamiltonian by defined d₀, d₁, d₂, d₃
    
    σ₀ = [1 0; 0 1]
    σₓ = [0 1; 1 0]
    σᵧ = [0 -im; im 0]
    σ𝓏 = [1 0; 0 -1]

    H = d₀(kx, ky) * σ₀ + d₁(kx, ky) * σₓ + d₂(kx, ky) * σᵧ + d₃(kx, ky) * σ𝓏
    return σ𝓏 * H
end

a = 1 # set to per unit lattice constnat

# honeycomb
n_n = a.*[[0, 1], [-√3/2, -1/2], [√3/2, -1/2]]
next_n_n = a.*[[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]]

# custom your d₀, d₁, d₂, d₃ functions
function d₀(kx, ky; J=1.54, S=5/2, K=-0.00344, J₂=0)
    # d₀ function
    k = [kx, ky]
    next_n_n = a.*[[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]]
    γ₂ = sin(dot(k, next_n_n[1])) + sin(dot(k, next_n_n[2])) + sin(dot(k, next_n_n[3]))
    return  3 * J * S - K*(2*S-1) + 2 * J₂ * S * γ₂
end

function d₁(kx, ky; J=1.54, S=5/2)
    # d₁ function
    k = [kx, ky]
    n_n = a.*[[0, 1], [-√3/2, -1/2], [√3/2, -1/2]]
    Reγ = real.(exp(1im * dot(k, n_n[1])) + exp(1im * dot(k, n_n[2])) + exp(1im * dot(k, n_n[3])))
    return J * S * Reγ
end 

function d₂(kx, ky; J=1.54, S=5/2)
    # d₂ function
    k = [kx, ky]
    n_n = a.*[[0, 1], [-√3/2, -1/2], [√3/2, -1/2]]
    Imγ = -imag.(exp(1im * dot(k, n_n[1])) + exp(1im * dot(k, n_n[2])) + exp(1im * dot(k, n_n[3])))
    return J * S * Imγ
end

function d₃(kx, ky; D=0.36, S=5/2)
    # d₃ function
    k = [kx, ky]
    next_n_n = a.*[[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]]
    λ = sin(dot(k, next_n_n[1])) + sin(dot(k, next_n_n[2])) + sin(dot(k, next_n_n[3]))
    return 2 * D * S * λ
end

function berry_curvature_two_band_bosonic_formula(kx, ky)
    # Calculate the Berry curvature using the two-band model
    d = [d₁(kx, ky), d₂(kx, ky), d₀(kx, ky)]
    ∂d_∂kx = [∂d₁_∂kx(kx, ky), ∂d₂_∂kx(kx, ky), ∂d₀_∂kx(kx, ky)]
    ∂d_∂ky = [∂d₁_∂ky(kx, ky), ∂d₂_∂ky(kx, ky), ∂d₀_∂ky(kx, ky)]
    d_abs = sqrt(d[3]^2 - d[1]^2 - d[2]^2)

    triple_product = d[1] * (∂d_∂kx[2] * ∂d_∂ky[3] - ∂d_∂kx[3] * ∂d_∂ky[2]) +
                    d[2] * (∂d_∂kx[3] * ∂d_∂ky[1] - ∂d_∂kx[1] * ∂d_∂ky[3]) +
                    d[3] * (∂d_∂kx[1] * ∂d_∂ky[2] - ∂d_∂kx[2] * ∂d_∂ky[1])
    Ω = 0.5 * triple_product / d_abs^3
    return Ω
end


function berry_curvature_bosonic_component_formula(kx ,ky)
    # Calculate the Berry curvature component using the two-band model
    Hx_12 = u₋L(kx, ky) * H₁(kx, ky) * eigen_vector_2(kx, ky) # 1 for lower
    Hx_21 = u₊L(kx, ky) * H₁(kx, ky) * eigen_vector_1(kx, ky)
    Hy_12 = u₋L(kx, ky) * H₂(kx, ky) * eigen_vector_2(kx, ky)
    Hy_21 = u₊L(kx, ky) * H₂(kx, ky) * eigen_vector_1(kx, ky)
    Ω₁ = imag((Hx_12 * Hy_21 - Hy_12 * Hx_21)/ ((E₁(kx, ky) - E₂(kx, ky))^2))
    Ω₂ = imag((Hx_21 * Hy_12 - Hy_21 * Hx_12)/ ((E₁(kx, ky) - E₂(kx, ky))^2))
    return Ω₁
end

# General parameters for two band model
τ₃ = [1 0; 0 -1] # Pauli matrix across particle-hole space
d_abs(kx, ky) = sqrt(d₀(kx, ky)^2 - d₁(kx, ky)^2 - d₂(kx, ky)^2)
coshψ(kx, ky) = sqrt((d₀(kx, ky)+d_abs(kx, ky))/(2 * d_abs(kx, ky)))
sinhψ(kx, ky) = sqrt((d₀(kx, ky)-d_abs(kx, ky))/(2 * d_abs(kx, ky)))
phase_φ(kx, ky) = (d₁(kx, ky) - im * d₂(kx, ky)) / sqrt(d₁(kx, ky)^2 + d₂(kx, ky)^2)
eigen_vector_1(kx, ky) = [-coshψ(kx, ky) * phase_φ(kx, ky) , sinhψ(kx, ky)]
eigen_vector_2(kx, ky) = [sinhψ(kx, ky) * phase_φ(kx, ky) , -coshψ(kx, ky)]
U_k(kx, ky) = [eigen_vector_1(kx, ky) eigen_vector_2(kx, ky)]
U_k_inv(kx, ky) = τ₃ * transpose(conj(U_k(kx, ky))) * τ₃
u₋L(kx, ky) = transpose(U_k_inv(kx, ky)[1, :]) # <u₋ᴸ| 
u₊L(kx, ky) = transpose(U_k_inv(kx, ky)[2, :]) # <u₊ᴸ|


transpose(conj(eigen_vector_1(1, 1))) * eigen_vector_1(1, 1)
transpose(U_k_inv(1, 1)[1, :]) * U_k(1, 1)[:, 1]


# Derivatives of the Hamiltonian or your model
H₁(kx, ky) = ForwardDiff.derivative(kx -> τ₃ * two_band_bosonic_model(kx, ky), kx)
H₂(kx, ky) = ForwardDiff.derivative(ky -> τ₃ * two_band_bosonic_model(kx, ky), ky)
H₁₁(kx, ky) = ForwardDiff.derivative(kx -> H₁(kx, ky), kx)
H₁₂(kx, ky) = ForwardDiff.derivative(ky -> H₁(kx, ky), ky)
H₂₁(kx, ky) = ForwardDiff.derivative(kx -> H₂(kx, ky), kx)
H₂₂(kx, ky) = ForwardDiff.derivative(ky -> H₂(kx, ky), ky)

# Eigenvalues and Eigenvectors of the Hamiltonian; and their Derivatives
E₁(kx, ky) = eigvals(two_band_bosonic_model(kx, ky))[1] # lower 
E₂(kx, ky) = eigvals(two_band_bosonic_model(kx, ky))[2] # upper
u₁(kx, ky) = eigvecs(two_band_bosonic_model(kx, ky))[:, 1] # lower band eigenvector
u₂(kx, ky) = eigvecs(two_band_bosonic_model(kx, ky))[:, 2] # upper band eigenvector
∂E₁_∂kx(kx, ky) = ForwardDiff.derivative(kx -> E₁(kx, ky), kx)
∂E₁_∂ky(kx, ky) = ForwardDiff.derivative(ky -> E₁(kx, ky), ky)
∂E₂_∂kx(kx, ky) = ForwardDiff.derivative(kx -> E₂(kx, ky), kx)
∂E₂_∂ky(kx, ky) = ForwardDiff.derivative(ky -> E₂(kx, ky), ky)
∂u₁_∂kx(kx, ky) = ForwardDiff.derivative(kx -> u₁(kx, ky), kx)
∂u₁_∂ky(kx, ky) = ForwardDiff.derivative(ky -> u₁(kx, ky), ky)
∂u₂_∂kx(kx, ky) = ForwardDiff.derivative(kx -> u₂(kx, ky), kx)
∂u₂_∂ky(kx, ky) = ForwardDiff.derivative(ky -> u₂(kx, ky), ky)

# Define the number of points in kx and ky
Nx = 100
Ny = 100

# Define the kx and ky ranges
kx = range(-π, π, length=Nx)
ky = range(-π, π, length=Ny)
dkx = step(kx)
dky = step(ky)

# K,K' points
rot = [cos(2π/6) -sin(2π/6); sin(2π/6) cos(2π/6)]
K = 2π*[2/3,0]/√3 #K
K′ = rot * K

# Hamiltonian, eigenvalues and eigenvectors
H =  [two_band_bosonic_model(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
eigenvalues = [eigvals(H[i, j]) for i in 1:Nx, j in 1:Ny]
eigenvectors = [eigvecs(H[i, j]) for i in 1:Nx, j in 1:Ny]
lower_band = getindex.(eigenvalues, 1) # in python, use np.vectorize(lambda v:v[i])(bands) to get i-th bands ; or simply write loops
upper_band = getindex.(eigenvalues, 2)

# Derivatives of parameters
∂d₀_∂kx(kx, ky) = ForwardDiff.derivative(kx -> d₀(kx, ky), kx)
∂d₀_∂ky(kx, ky) = ForwardDiff.derivative(ky -> d₀(kx, ky), ky)
∂d₁_∂kx(kx, ky) = ForwardDiff.derivative(kx -> d₁(kx, ky), kx)
∂d₁_∂ky(kx, ky) = ForwardDiff.derivative(ky -> d₁(kx, ky), ky)
∂d₂_∂kx(kx, ky) = ForwardDiff.derivative(kx -> d₂(kx, ky), kx)
∂d₂_∂ky(kx, ky) = ForwardDiff.derivative(ky -> d₂(kx, ky), ky)
∂d₃_∂kx(kx, ky) = ForwardDiff.derivative(kx -> d₃(kx, ky), kx)
∂d₃_∂ky(kx, ky) = ForwardDiff.derivative(ky -> d₃(kx, ky), ky)

# parameters value
d3_values = [d₃(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]
d3_derivative_kx = [∂d₃_∂kx(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]
d3_derivative_ky = [∂d₃_∂ky(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]

# geometric quantities
Ω = [berry_curvature_bosonic_component_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
#Ω = [berry_curvature_two_band_bosonic_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
gap = [d_abs(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]

# Plotting the eigenvalues
fig = Figure()
ax1 = Axis3(fig[1, 1], title="Parameters", xlabel="kx", ylabel="ky")
surface!(ax1, kx, ky, real(upper_band),  colormap=:winter, alpha=0.5)
surface!(ax1, kx, ky, -real(lower_band),  colormap=:dense, alpha=0.5)
ax_heat = Axis(fig[2, 1], title="Parameters", aspect=DataAspect() , xlabel="kx", ylabel="ky")
hm = heatmap!(ax_heat, kx, ky, Ω, colormap=:viridis)
poly!(ax_heat, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[2, 2], hm, label="Energy (meV)")
#surface!(ax1, kx, ky, upper_band, colormap=:plasma)
#surface!(ax1, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
ax1.azimuth[] = π/4     # Horizontal rotation (radians)
ax1.elevation[] = π/16  # Vertical tilt (radians)
fig 

# Integration within first BZ 
honeycomb = Polygon(Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K])
honeycomb_mesh = Meshes.PolyArea(Tuple.(coordinates(honeycomb)))
bzmesh_points = [Meshes.Point(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
honeycomb_mask = bzmesh_points .∈ honeycomb_mesh

sum(real.(Ω)[honeycomb_mask])*dkx*dky/(2π)

findmax([Ω[i,j] == reduce(min, Ω) ? 1 : 0 for i in 1:Nx, j in 1:Ny])