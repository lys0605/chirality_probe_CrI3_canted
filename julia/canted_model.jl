using ForwardDiff, LinearAlgebra
using CairoMakie, LaTeXStrings, GeometryBasics
import Meshes


# Canted AFM on honeycomb lattice
a = 1
n_n = [[0, 1], [-√3/2, -1/2], [√3/2, -1/2]] .* a
next_n_n = [[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]] .* a
τ₃ = [1 0 0 0; 
      0 1 0 0;
      0 0 -1 0;
      0 0 0 -1]
τ₀ = [1 0 0 0; 
      0 1 0 0;
      0 0 1 0;
      0 0 0 1]

ϕₖ(kx, ky; J=1, S=5/2) = 2*J*S*(exp(-1im*dot([kx,ky], n_n[1])) + exp(-1im*dot([kx,ky], n_n[2])) + exp(-1im*dot([kx,ky], n_n[3])))
λₖ(kx, ky; D=0.1, S=5/2, s=0.6) = 4*D*S*s*(sin(dot([kx,ky], next_n_n[1])) + sin(dot([kx,ky], next_n_n[2])) + sin(dot([kx,ky], next_n_n[3])))
Δₖ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt(λₖ(kx,ky; D, S, s)^2 + (s^4*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S)))))

H(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = [6*J*S+λₖ(kx,ky) -s^2*conj(ϕₖ(kx,ky)) 0 (1-s^2)*conj(ϕₖ(kx,ky));
                                       -s^2*ϕₖ(kx,ky) 6*J*S-λₖ(kx,ky) (1-s^2)*ϕₖ(kx,ky) 0;
                                       0 (1-s^2)*conj(ϕₖ(kx,ky)) 6*J*S-λₖ(kx,ky) -s^2*conj(ϕₖ(kx,ky));
                                       (1-s^2)*ϕₖ(kx,ky) 0 -s^2*ϕₖ(kx,ky) 6*J*S+λₖ(kx,ky)]
ϵ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₃(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₄(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))

# eigenvectors
sinhχ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s) - ϵ₁(kx,ky; J, D, S, s)) / (2*ϵ₁(kx,ky; J, D, S, s)))
sinhχ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s) - ϵ₂(kx,ky; J, D, S, s)) / (2*ϵ₂(kx,ky; J, D, S, s)))
coshχ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s) + ϵ₁(kx,ky; J, D, S, s)) / (2*ϵ₁(kx,ky; J, D, S, s)))
coshχ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s) + ϵ₂(kx,ky; J, D, S, s)) / (2*ϵ₂(kx,ky; J, D, S, s)))
cosψ(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt( (Δₖ(kx,ky; J, D, S, s)+λₖ(kx,ky; D, S, s)) / (2*Δₖ(kx,ky; J, D, S, s)))
sinψ(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt( (Δₖ(kx,ky; J, D, S, s)-λₖ(kx,ky; D, S, s)) / (2*Δₖ(kx,ky; J, D, S, s)))
φ(kx ,ky; J=1, S=5/2) = 1im*log(ϕₖ(kx,ky; J=J, S=S)/sqrt(ϕₖ(kx,ky; J=J, S=S)*conj(ϕₖ(kx,ky; J=J, S=S))))

u₁(kx, ky; J=1, D=0.1, S=5/2, s=0.6) =  [-coshχ₁( kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        coshχ₁(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2);
                                        -sinhχ₁(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        sinhχ₁(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2)]

u₂(kx, ky; J=1, D=0.1, S=5/2, s=0.6) =  [coshχ₂( kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        coshχ₂(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2);
                                        sinhχ₂(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        sinhχ₂(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2)]

u₃(kx, ky; J=1, D=0.1, S=5/2, s=0.6) =  [-sinhχ₁( kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        sinhχ₁(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2);
                                        -coshχ₁(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        coshχ₁(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2)]

u₄(kx, ky; J=1, D=0.1, S=5/2, s=0.6) =  [sinhχ₂( kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        sinhχ₂(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2);
                                        coshχ₂(kx,ky; J, D, S, s)*cosψ(kx,ky; J, D, S, s)*exp(1im*φ(kx,ky)/2);
                                        coshχ₂(kx,ky; J, D, S, s)*sinψ(kx,ky; J, D, S, s)*exp(-1im*φ(kx,ky)/2)]
Uₖ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = [u₁(kx, ky; J, D, S, s) u₂(kx, ky; J, D, S, s) u₃(kx, ky; J, D, S, s) u₄(kx, ky; J, D, S, s)]
Uₖ_inv(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = τ₃ * adjoint(Uₖ(kx, ky; J, D, S, s)) * τ₃
u₁ᴸ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = Uₖ_inv(kx, ky; J, D, S, s)[1,:]
u₂ᴸ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = Uₖ_inv(kx, ky; J, D, S, s)[2,:]
u₃ᴸ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = Uₖ_inv(kx, ky; J, D, S, s)[3,:]
u₄ᴸ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = Uₖ_inv(kx, ky; J, D, S, s)[4,:]

# derivatives of Hamiltonian
Hₓ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> H(kx, ky; J, D, S, s), kx)
Hᵧ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> H(kx, ky; J, D, S, s), ky)
H_xx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> Hₓ(kx, ky; J, D, S, s), kx)
H_yy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> Hᵧ(kx, ky; J, D, S, s), ky)
H_xy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> Hₓ(kx, ky; J, D, S, s), ky)
H_yx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> Hᵧ(kx, ky; J, D, S, s), kx)

# derivtives of eigenvector
∂u₁_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₁(kx, ky; J, D, S, s), kx)
∂u₁_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₁(kx, ky; J, D, S, s), ky)
∂u₂_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₂(kx, ky; J, D, S, s), kx)
∂u₂_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₂(kx, ky; J, D, S, s), ky)
∂u₃_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₃(kx, ky; J, D, S, s), kx)
∂u₃_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₃(kx, ky; J, D, S, s), ky)
∂u₄_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₄(kx, ky; J, D, S, s), kx)
∂u₄_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₄(kx, ky; J, D, S, s), ky)
∂u₁ᴸ_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₁ᴸ(kx, ky; J, D, S, s), kx)
∂u₁ᴸ_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₁ᴸ(kx, ky; J, D, S, s), ky)
∂u₂ᴸ_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₂ᴸ(kx, ky; J, D, S, s), kx)
∂u₂ᴸ_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₂ᴸ(kx, ky; J, D, S, s), ky)
∂u₃ᴸ_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₃ᴸ(kx, ky; J, D, S, s), kx)
∂u₃ᴸ_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₃ᴸ(kx, ky; J, D, S, s), ky)
∂u₄ᴸ_∂kx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(kx -> u₄ᴸ(kx, ky; J, D, S, s), kx)
∂u₄ᴸ_∂ky(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = ForwardDiff.derivative(ky -> u₄ᴸ(kx, ky; J, D, S, s), ky)


coshχ₁(1,1)^2-sinhχ₁(1,1)^2

# Quantum Geometry
Q₁_xy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₁ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₁(kx,ky;J, D, S, s)*transpose(u₁ᴸ(kx, ky; J, D, S, s))) * ∂u₁_∂ky(kx, ky; J, D, S, s)
Q₂_xy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₂ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₂(kx,ky;J, D, S, s)*transpose(u₂ᴸ(kx, ky; J, D, S, s))) * ∂u₂_∂ky(kx, ky; J, D, S, s)
Q₃_xy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₃ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₃(kx,ky;J, D, S, s)*transpose(u₃ᴸ(kx, ky; J, D, S, s))) * ∂u₃_∂ky(kx, ky; J, D, S, s)
Q₄_xy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₄ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₄(kx,ky;J, D, S, s)*transpose(u₄ᴸ(kx, ky; J, D, S, s))) * ∂u₄_∂ky(kx, ky; J, D, S, s)
Q₁_xx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₁ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₁(kx,ky;J, D, S, s)*transpose(u₁ᴸ(kx, ky; J, D, S, s))) * ∂u₁_∂kx(kx, ky; J, D, S, s)
Q₂_xx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₂ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₂(kx,ky;J, D, S, s)*transpose(u₂ᴸ(kx, ky; J, D, S, s))) * ∂u₂_∂kx(kx, ky; J, D, S, s)
Q₃_xx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₃ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₃(kx,ky;J, D, S, s)*transpose(u₃ᴸ(kx, ky; J, D, S, s))) * ∂u₃_∂kx(kx, ky; J, D, S, s)
Q₄_xx(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₄ᴸ_∂kx(kx, ky;  J, D, S, s)) * (τ₀- u₄(kx,ky;J, D, S, s)*transpose(u₄ᴸ(kx, ky; J, D, S, s))) * ∂u₄_∂kx(kx, ky; J, D, S, s)
Q₁_yy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₁ᴸ_∂ky(kx, ky;  J, D, S, s)) * (τ₀- u₁(kx,ky;J, D, S, s)*transpose(u₁ᴸ(kx, ky; J, D, S, s))) * ∂u₁_∂ky(kx, ky; J, D, S, s)
Q₂_yy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₂ᴸ_∂ky(kx, ky;  J, D, S, s)) * (τ₀- u₂(kx,ky;J, D, S, s)*transpose(u₂ᴸ(kx, ky; J, D, S, s))) * ∂u₂_∂ky(kx, ky; J, D, S, s)
Q₃_yy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₃ᴸ_∂ky(kx, ky;  J, D, S, s)) * (τ₀- u₃(kx,ky;J, D, S, s)*transpose(u₃ᴸ(kx, ky; J, D, S, s))) * ∂u₃_∂ky(kx, ky; J, D, S, s)
Q₄_yy(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = transpose(∂u₄ᴸ_∂ky(kx, ky;  J, D, S, s)) * (τ₀- u₄(kx,ky;J, D, S, s)*transpose(u₄ᴸ(kx, ky; J, D, S, s))) * ∂u₄_∂ky(kx, ky; J, D, S, s)

# quantum geometry, component-wise

# LMCSs

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

s=0.75
λ_values = [λₖ(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
ϵ₁_values = [real(ϵ₁(kx[i], ky[j]; s=s)) for i in 1:Nx, j in 1:Ny]
Q₁_values = [Q₁_xy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₂_values = [Q₂_xy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₃_values = [Q₃_xy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₄_values = [Q₄_xy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₁_xx_values = [Q₁_xx(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₂_xx_values = [Q₂_xx(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₃_xx_values = [Q₃_xx(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₄_xx_values = [Q₄_xx(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₁_yy_values = [Q₁_yy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₂_yy_values = [Q₂_yy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₃_yy_values = [Q₃_yy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]
Q₄_yy_values = [Q₄_yy(kx[i], ky[j]; s=s) for i in 1:Nx, j in 1:Ny]

# magnon bands
ϵ₁_bands = [real(ϵ₁(kx[i], ky[j])) for i in 1:Nx, j in 1:Ny]
ϵ₂_bands = [real(ϵ₂(kx[i], ky[j])) for i in 1:Nx, j in 1:Ny]
ϵ₃_bands = [real(ϵ₃(kx[i], ky[j])) for i in 1:Nx, j in 1:Ny]
ϵ₄_bands = [real(ϵ₄(kx[i], ky[j])) for i in 1:Nx, j in 1:Ny]
Δ_12 = ϵ₁_bands .- ϵ₂_bands
Δ_13 = ϵ₁_bands .- ϵ₃_bands
Δ_14 = ϵ₁_bands .- ϵ₄_bands
Δ_23 = ϵ₂_bands .- ϵ₃_bands
Δ_34 = ϵ₃_bands .- ϵ₄_bands

# plot
function plot_heatmap_on_BZ(fig, kx, ky, Z; axis=[1,1], title="", xlabel=L"k_x", ylabel=L"k_y", xlabelsize=18, ylabelsize=18, colormap=:viridis, coloarmaplabel="")
    ax = Axis(fig[axis[1], axis[2]], title=title, aspect=DataAspect() , xlabel=xlabel, ylabel=ylabel, xlabelsize=xlabelsize, ylabelsize=ylabelsize)
    hm = heatmap!(ax, kx, ky, Z, colormap=colormap)
    colorbar_axis = axis+[0, 1]
    Colorbar(fig[colorbar_axis[1], colorbar_axis[2]], hm, label=coloarmaplabel)
    return ax, hm
end

# function plot_surface_on_BZ(fig, kx, ky, Z; axis=[1,1], title="", xlabel=L"k_x", ylabel=L"k_y", xlabelsize=18, ylabelsize=18, colormap=:viridis, coloarmaplabel="")
#     ax = Axis3(fig[axis[1], axis[2]], title=title, xlabel=xlabel, ylabel=ylabel, xlabelsize=xlabelsize, ylabelsize=ylabelsize)
#     surface!(ax, kx, ky, Z, colormap=colormap,  alpha=0.5)
#     return ax
# end

fig = Figure(size=(900, 500))
ax_heat_1, hm_1 = plot_heatmap_on_BZ(fig, kx, ky, real.(Q₁_values); 
                                    axis=[1,1], title=L"\textbf{Quantum metric} $g_{+,xy}$", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_1, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)

ax_heat_2, hm_2 = plot_heatmap_on_BZ(fig, kx, ky, -2*imag.(Q₁_values); 
                                    axis=[1,3], title=L"\textbf{Berry curvature} $Ω_{+,xy}$", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_2, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)

ax_heat_3, hm_3 = plot_heatmap_on_BZ(fig, kx, ky, real.(Q₁_xx_values .- Q₁_yy_values); 
                                    axis=[1,5], title=L"\textbf{tr}(g_{+})", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_3, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)

ax_heat_4, hm_4 = plot_heatmap_on_BZ(fig, kx, ky, real.(Q₂_values); 
                                    axis=[2,1], title=L"\textbf{Quantum metric} $g_{-,xy}$", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_4, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)

ax_heat_5, hm_5 = plot_heatmap_on_BZ(fig, kx, ky,  -2*imag.(Q₂_values); 
                                    axis=[2,3], title=L"\textbf{Berry curvature} $Ω_{-,xy}$", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_5, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)

ax_heat_6, hm_6 = plot_heatmap_on_BZ(fig, kx, ky, real.(Q₂_xx_values .- Q₂_yy_values) ; 
                                    axis=[2,5], title=L"\textbf{tr}(g_{-})", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="")
poly!(ax_heat_6, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
fig

# ax_heat_3 = Axis(fig[2, 1], title=L"\textbf{LMCs} $L_{\mu\nu}$", 
#                 aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
#                 xlabelsize=18, ylabelsize=18)
# hm_3 = heatmap!(ax_heat_3, kx, ky, LD, colormap=:viridis)
# poly!(ax_heat_3, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
# Colorbar(fig[2, 2], hm_3, label="Energy (meV)")

# ax_heat_4 = Axis(fig[2, 3], title=L"2*(g_\mu\nu)\Delta_g^2 - L_{\mu\nu}", 
#                 aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
#                 xlabelsize=18, ylabelsize=18)
# hm_4 = heatmap!(ax_heat_4, kx, ky, LD_xx_yy .- 2*(g_11-g_22).*gap.^2, colormap=:viridis)
# poly!(ax_heat_4, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
# Colorbar(fig[2, 4], hm_4, label="Energy (meV)")
#surface!(ax1, kx, ky, upper_band, colormap=:plasma)
#surface!(ax1, kx, ky, zeros(100,100) , colormap=:darkterrain, alpha=0.1)
#ax1.azimuth[] = π/4     # Horizontal rotation (radians)
#ax1.elevation[] = π/16  # Vertical tilt (radians)
#fig 

# Integration within first BZ 
honeycomb = Polygon(Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K])
honeycomb_mesh = Meshes.PolyArea(Tuple.(coordinates(honeycomb)))
bzmesh_points = [Meshes.Point(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
honeycomb_mask = bzmesh_points .∈ honeycomb_mesh

sum(-2*imag.(Q₂_values[honeycomb_mask]))*dkx*dky/(2π)

findmax([Ω[i,j] == reduce(min, Ω) ? 1 : 0 for i in 1:Nx, j in 1:Ny])



            