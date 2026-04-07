using LinearAlgebra, CairoMakie

const σ₀ = [1 0; 0 1]
const σₓ = [0 1; 1 0]
const σᵧ = ComplexF64[0 -im; im 0]
const σ𝓏 = [1 0; 0 -1]

function two_band_model(kx, ky)
    # Construct the two-band model Hamiltonian by defined d₀, d₁, d₂, d₃
    H = d₀(kx, ky) * σ₀ + d₁(kx, ky) * σₓ + d₂(kx, ky) * σᵧ + d₃(kx, ky) * σ𝓏
    return H
end

# custom your d₀, d₁, d₂, d₃ functions
d₀ = (kx, ky) -> 0.5 * (1 + cos(kx) + cos(ky))
d₁ = (kx, ky) -> 0.5 * (1 + cos(kx) - cos(ky))
d₂ = (kx, ky) -> 0.5 * (1 - cos(kx) + cos(ky))
d₃ = (kx, ky) -> 0.5 * (1 - cos(kx) - cos(ky))

# Define the kx and ky ranges
kx = range(-π, π, length=100)
ky = range(-π, π, length=100)

# Create your model
H =  [two_band_model(kx[i], ky[j]) for i in 1:length(kx), j in 1:length(ky)]

# Calculate the eigenvalues and eigenvectors
eigenvalues = [eigvals(H[i, j]) for i in 1:length(kx), j in 1:length(ky)]
eigenvectors = [eigvecs(H[i, j]) for i in 1:length(kx), j in 1:length(ky)]
lower_band = getindex.(eigenvalues, 1) # in python, use np.vectorize(lambda v:v[i])(bands) to get i-th bands ; or simply write loops
upper_band = getindex.(eigenvalues, 2)

# bosonic case
σ𝓏 = [1 0; 0 -1]
σH = (*).(Ref(σ𝓏), H) # element-wise multiplication
bosonic_eigenvalues = [eigvals(σH[i, j]) for i in 1:length(kx), j in 1:length(ky)]
bosonic_eigenvectors = [eigvecs(σH[i, j]) for i in 1:length(kx), j in 1:length(ky)]
bosonic_lower_band = getindex.(-1*bosonic_eigenvalues, 1)  
bosonic_upper_band = getindex.(-1*bosonic_eigenvalues, 2)

# Plotting the eigenvalues
fig = Figure()
ax1 = Axis3(fig[1, 1], title="Eigenvalues of Two-Band Model", xlabel="kx", ylabel="ky")
surface!(ax1, kx, ky, lower_band,  colormap=:viridis)
surface!(ax1, kx, ky, upper_band, colormap=:plasma)
surface!(ax1, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
ax1.azimuth[] = π/4     # Horizontal rotation (radians)
ax1.elevation[] = π/16   # Vertical tilt (radians)# Plotting the bosonic eigenvalues

# Plotting the bosonic eigenvalues
ax2 = Axis3(fig[1, 2], title="Bosonic Eigenvalues of Two-Band Model", xlabel="kx", ylabel="ky")
surface!(ax2, kx, ky, real.(bosonic_lower_band),  colormap=:viridis)
surface!(ax2, kx, ky, real.(bosonic_upper_band), colormap=:plasma)
surface!(ax2, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
ax2.azimuth[] = π/4    # Horizontal rotation (radians) 
ax2.elevation[] = π/16 # Vertical tilt (radians)
fig
