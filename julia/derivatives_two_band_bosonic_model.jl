using LinearAlgebra, ForwardDiff
using CairoMakie, GeometryBasics
using LaTeXStrings
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

function quantum_metric_two_band_bosonic_formula(kx, ky)
    # Calculate the Berry curvature using the two-band model
    d = [d₁(kx, ky), d₂(kx, ky), d₀(kx, ky)]
    d_abs = sqrt(d[3]^2 - d[1]^2 - d[2]^2)
    ∂d_∂kx = [∂d₁_∂kx(kx, ky), ∂d₂_∂kx(kx, ky), ∂d₀_∂kx(kx, ky)]
    ∂d_∂ky = [∂d₁_∂ky(kx, ky), ∂d₂_∂ky(kx, ky), ∂d₀_∂ky(kx, ky)]
   
    dot_product_11 = -(∂d_∂kx[1]^2 + ∂d_∂kx[2]^2 - ∂d_∂kx[3]^2)
    dot_product_12 = -(∂d_∂kx[1]*∂d_∂ky[1] + ∂d_∂kx[2]*∂d_∂ky[2] - ∂d_∂kx[3]*∂d_∂ky[3])
    dot_product_21 = -(∂d_∂ky[1]*∂d_∂kx[1] + ∂d_∂ky[2]*∂d_∂kx[2] - ∂d_∂ky[3]*∂d_∂kx[3])
    dot_product_22 = -(∂d_∂ky[1]^2 + ∂d_∂ky[2]^2 - ∂d_∂ky[3]^2)

    connection_product_x = -(d[1]*∂d_∂kx[1] + d[2]*∂d_∂kx[2] - d[3]*∂d_∂kx[3])
    connection_product_y = -(d[1]*∂d_∂ky[1] + d[2]*∂d_∂ky[2] - d[3]*∂d_∂ky[3])
    connection_product_11 = connection_product_x^2 / d_abs^2
    connection_product_12 = connection_product_x * connection_product_y / d_abs^2
    connection_product_21 = connection_product_y * connection_product_x / d_abs^2
    connection_product_22 = connection_product_y^2 / d_abs^2

    g_11 = 0.25 * (dot_product_11 / d_abs^2 - connection_product_11 / d_abs^2)
    g_12 = 0.25 * (dot_product_12 / d_abs^2 - connection_product_12 / d_abs^2)
    g_21 = 0.25 * (dot_product_21 / d_abs^2 - connection_product_21 / d_abs^2)
    g_22 = 0.25 * (dot_product_22 / d_abs^2 - connection_product_22 / d_abs^2)
    return g_22
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
u₊L(kx, ky) = transpose(U_k_inv(kx, ky)[1, :]) # <u₊ᴸ| 
u₋L(kx, ky) = transpose(U_k_inv(kx, ky)[2, :]) # <u₋ᴸ|

# Derivatives of the Hamiltonian or your model
H₁(kx, ky) = ForwardDiff.derivative(kx -> two_band_bosonic_model(kx, ky), kx)
H₂(kx, ky) = ForwardDiff.derivative(ky -> two_band_bosonic_model(kx, ky), ky)
H₁₁(kx, ky) = ForwardDiff.derivative(kx -> H₁(kx, ky), kx)
H₁₂(kx, ky) = ForwardDiff.derivative(ky -> H₁(kx, ky), ky)
H₂₁(kx, ky) = ForwardDiff.derivative(kx -> H₂(kx, ky), kx)
H₂₂(kx, ky) = ForwardDiff.derivative(ky -> H₂(kx, ky), ky)
L̃_11(kx, ky) = transpose(conj(U_k(kx, ky))) * τ₃ * H₁₁(kx, ky) * U_k(kx, ky)
L̃_12(kx, ky) = transpose(conj(U_k(kx, ky))) * τ₃ * H₁₂(kx, ky) * U_k(kx, ky)
L̃_21(kx, ky) = transpose(conj(U_k(kx, ky))) * τ₃ * H₂₁(kx, ky) * U_k(kx, ky)
L̃_22(kx, ky) = transpose(conj(U_k(kx, ky))) * τ₃ * H₂₂(kx, ky) * U_k(kx, ky)

# Quantum geometric components in terms of derivatives of Hamiltonian
# 1 for upper; 2 for lower
Hx_12(kx,ky) = u₊L(kx, ky) * H₁(kx, ky) * eigen_vector_2(kx, ky) # ⟨u₊ᴸ| ∂H/∂kx |u₋ᴿ⟩
Hx_21(kx,ky) = u₋L(kx, ky) * H₁(kx, ky) * eigen_vector_1(kx, ky) # ⟨u₋ᴸ| ∂H/∂kx |u₊ᴿ⟩
Hy_12(kx,ky) = u₊L(kx, ky) * H₂(kx, ky) * eigen_vector_2(kx, ky) # ⟨u₊ᴸ| ∂H/∂ky |u₋ᴿ⟩
Hy_21(kx,ky) = u₋L(kx, ky) * H₂(kx, ky) * eigen_vector_1(kx, ky) # ⟨u₋ᴸ| ∂H/∂ky |u₊ᴿ⟩
quantum_metric_11(kx,ky) = real((Hx_12(kx, ky) * Hx_21(kx, ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2))
quantum_metric_12(kx,ky) = real((Hx_12(kx, ky) * Hy_21(kx, ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2))
quantum_metric_21(kx,ky) = real((Hy_12(kx, ky) * Hx_21(kx, ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2)) 
quantum_metric_22(kx,ky) = real((Hy_12(kx, ky) * Hy_21(kx, ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2))
Ω₁(kx,ky) = imag((Hx_12(kx,ky) * Hy_21(kx,ky) - Hy_12(kx,ky) * Hx_21(kx,ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2))
Ω₂(kx,ky) = imag((Hx_21(kx,ky) * Hy_12(kx,ky) - Hy_21(kx,ky) * Hx_12(kx,ky))/ ((E₁(kx, ky) - E₂(kx, ky))^2))

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
H_M =  [two_band_bosonic_model(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
eigenvalues = [eigvals(H_M[i, j]) for i in 1:Nx, j in 1:Ny]
eigenvectors = [eigvecs(H_M[i, j]) for i in 1:Nx, j in 1:Ny]
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
r_x(kx, ky) = ∂d₁_∂kx(kx, ky) + im * ∂d₂_∂kx(kx, ky)
r_y(kx, ky) = ∂d₁_∂ky(kx, ky) + im * ∂d₂_∂ky(kx, ky)

# Quantum geometric direct formula
function quantum_metric_direct(kx, ky)
    g_11 = -2*real(r_x(kx,ky) * r_x(kx,ky) * coshψ(kx,ky)^2 * sinhψ(kx,ky)^2 * phase_φ(kx,ky)^2) - real(r_x(kx,ky)*conj(r_x(kx,ky)))*(coshψ(kx,ky)^4 + sinhψ(kx,ky)^4)
    g_22 = -2*real(r_y(kx,ky)*r_y(kx,ky)*coshψ(kx,ky)^2*sinhψ(kx,ky)^2*phase_φ(kx,ky)^2) - real(r_y(kx,ky)*conj(r_y(kx,ky)))*(coshψ(kx,ky)^4 + sinhψ(kx,ky)^4)
    g_12 = -2*real(r_x(kx,ky)*r_y(kx,ky)*coshψ(kx,ky)^2*sinhψ(kx,ky)^2*phase_φ(kx,ky)^2) - real(r_x(kx,ky)*conj(r_y(kx,ky)))*(coshψ(kx,ky)^4 + sinhψ(kx,ky)^4)
    g_21 = -2*real(r_y(kx,ky)*r_x(kx,ky)*coshψ(kx,ky)^2*sinhψ(kx,ky)^2*phase_φ(kx,ky)^2) - real(r_y(kx,ky)*conj(r_x(kx,ky)))*(coshψ(kx,ky)^4 + sinhψ(kx,ky)^4)
    return real(g_21 / (E₁(kx, ky) - E₂(kx, ky))^2)
end

# parameters value
d3_values = [d₃(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]
d3_derivative_kx = [∂d₃_∂kx(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]
d3_derivative_ky = [∂d₃_∂ky(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]


# geometric quantities
Ω₊ = [Ω₁(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
Ω₋ = [Ω₂(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
r_11 = [L̃_11(kx[i], ky[j])[1,2]  for i in 1:Nx, j in 1:Ny]
r_12 = [L̃_12(kx[i], ky[j])[1,2]  for i in 1:Nx, j in 1:Ny]
r_21 = [L̃_21(kx[i], ky[j])[1,2]  for i in 1:Nx, j in 1:Ny]
r_22 = [L̃_22(kx[i], ky[j])[1,2]  for i in 1:Nx, j in 1:Ny]
curl = imag.(conj.(r_11 - r_22) .* r_12)
LD = real.(conj.(r_11 + r_22) .* r_12)
LD_xx_yy = real.(conj.(r_11) .* r_11 .- conj.(r_22) .* r_22)
gap = [d_abs(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
h_z = [d₃(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]

# quantum metric 
g_11 = [quantum_metric_11(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_12 = [quantum_metric_12(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_21 = [quantum_metric_21(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_22 = [quantum_metric_22(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]

g_formula_11 = [quantum_metric_two_band_bosonic_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_formula_12 = [quantum_metric_two_band_bosonic_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_formula_21 = [quantum_metric_two_band_bosonic_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_formula_22 = [quantum_metric_two_band_bosonic_formula(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]

g_direct_11 = [quantum_metric_direct(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_direct_12 = [quantum_metric_direct(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_direct_21 = [quantum_metric_direct(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
g_direct_22 = [quantum_metric_direct(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]

det_g = g_11.*g_22 .- g_12.*g_21
det_direct_g = g_direct_11.*g_direct_22 .- g_direct_12.*g_direct_21
det_formula_g = g_formula_11.*g_formula_22 .- g_formula_12.*g_formula_21

tr_g = 0.5 .* (g_11 .+ g_22)
tr_direct_g = 0.5 .* (g_direct_11 .+ g_direct_22)
tr_formula_g = 0.5 .* (g_formula_11 .+ g_formula_22)

g₊ = (tr_g .+ sqrt.(tr_g.^2 .- det_g)) ./ 2
g₋ = (tr_g .- sqrt.(tr_g.^2 .- det_g)) ./ 2

# Plotting the eigenvalues
fig = Figure()
# ax1 = Axis3(fig[1, 1], title="Energy bands", xlabel="kx", ylabel="ky")
# surface!(ax1, kx, ky, real(upper_band),  colormap=:winter, alpha=0.5)
# surface!(ax1, kx, ky, -real(lower_band),  colormap=:dense, alpha=0.5)
ax_heat_1 = Axis(fig[1, 1], title=L"\textbf{Quantum metric} $\frac{1}{2}\Delta_{+-}^2g_{xy}$", 
                aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y", 
                xlabelsize=18, ylabelsize=18)
hm_1 = heatmap!(ax_heat_1, kx, ky,  2 .* gap.^2 .* (g_12), colormap=:viridis)
poly!(ax_heat_1, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[1, 2,] , hm_1, label="")

ax_heat_2 = Axis(fig[1, 3], title=L"\textbf{Berry curvature} $\Omega_{xy}$", 
                aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
                xlabelsize=18, ylabelsize=18)
hm_2 = heatmap!(ax_heat_2, kx, ky, Ω₋, colormap=:viridis)
poly!(ax_heat_2, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[1, 4], hm_2, label="")

ax_heat_3 = Axis(fig[2, 1], title=L"\textbf{LD (Diagonal-anti-diagonal)} ", 
                aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
                xlabelsize=18, ylabelsize=18)
hm_3 = heatmap!(ax_heat_3, kx, ky, LD, colormap=:viridis)
poly!(ax_heat_3, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[2, 2], hm_3, label="")

ax_heat_4 = Axis(fig[2, 3], title=L"\textbf{Difference}", 
                aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
                xlabelsize=18, ylabelsize=18)
hm_4 = heatmap!(ax_heat_4, kx, ky, LD .- 2*(g_12).*gap.^2, colormap=:viridis)
poly!(ax_heat_4, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[2, 4], hm_4, label="")
#surface!(ax1, kx, ky, upper_band, colormap=:plasma)
#surface!(ax1, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
# ax1.azimuth[] = π/4     # Horizontal rotation (radians)
# ax1.elevation[] = π/16  # Vertical tilt (radians)
fig
save("julia/figures/quantum_metric_linear_dichroism_diagonal_anti-diagonal_collinear_afm_honeycomb.png", fig, px_per_unit = 300/96)

# Integration within first BZ 
honeycomb = Polygon(Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K])
honeycomb_mesh = Meshes.PolyArea(Tuple.(coordinates(honeycomb)))
bzmesh_points = [Meshes.Point(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
honeycomb_mask = bzmesh_points .∈ honeycomb_mesh

sum(real.(Ω)[honeycomb_mask])*dkx*dky/(2π)

findmax([Ω[i,j] == reduce(min, Ω) ? 1 : 0 for i in 1:Nx, j in 1:Ny])
