using LinearAlgebra, ForwardDiff
using CairoMakie, GeometryBasics
import Meshes

# material: MnPS3; Néel temperature = 78K

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

Z₁(kx, ky, T) = bose_einstein_distribution(real(-E₁(kx, ky)), T) + 1
Z₂(kx, ky, T) = bose_einstein_distribution(real(E₂(kx, ky)), T) + 1

function partition(T; Nx=100, Ny=100)
    # Partition function for the two-band bosonic model
    # T is the temperature in Kelvin
    kx = range(-π, π, length=Nx)
    ky = range(-π, π, length=Ny)
    Z₋ = [Z₁(kx[i], ky[j], T) for i in 1:Nx, j in 1:Ny]
    Z₊ = [Z₂(kx[i], ky[j], T) for i in 1:Nx, j in 1:Ny]
    lnZ₋ = log.(Z₋)
    lnZ₊ = log.(Z₊)
    lnZ = honeycomb_BZ_integration(lnZ₋; Nx=Nx, Ny=Ny)/(2*pi)^2 + honeycomb_BZ_integration(lnZ₊; Nx=Nx, Ny=Ny)/(2*pi)^2
    return exp(lnZ) 
end

# LMCs in magnon basis
L̃₁₁(kx, ky) =  transpose(conj(U_k(kx, ky))) * H₁₁(kx, ky) * U_k(kx, ky)
L̃₁₂(kx, ky) =  transpose(conj(U_k(kx, ky))) * H₁₂(kx, ky) * U_k(kx, ky)
L̃₂₁(kx, ky) =  transpose(conj(U_k(kx, ky))) * H₂₁(kx, ky) * U_k(kx, ky)
L̃₂₂(kx, ky) =  transpose(conj(U_k(kx, ky))) * H₂₂(kx, ky) * U_k(kx, ky)

# H^(out)(in) polarization
L = 1
R = -1

Hᵣᴿᴸ(kx, ky) = 0.5 * (L̃₁₁(kx, ky) + R*L*L̃₂₂(kx, ky) + 1im*(L-R)*L̃₁₂(kx, ky))
Hᵣᴸᴸ(kx, ky) = 0.5 * (L̃₁₁(kx, ky) + L*L*L̃₂₂(kx, ky) + 1im*(L-L)*L̃₁₂(kx, ky))
Hᵣᴿᴿ(kx, ky) = 0.5 * (L̃₁₁(kx, ky) + R*R*L̃₂₂(kx, ky) + 1im*(R-R)*L̃₁₂(kx, ky))
Hᵣᴸᴿ(kx, ky) = 0.5 * (L̃₁₁(kx, ky) + L*R*L̃₂₂(kx, ky) + 1im*(R-L)*L̃₁₂(kx, ky))

# off-diagonal scattering amplitude -> AFM = two-magnon creation in different bands
tᴿᴸ(kx, ky) = Hᵣᴿᴸ(kx, ky)[1, 2]
tᴸᴸ(kx, ky) = Hᵣᴸᴸ(kx, ky)[1, 2]
tᴿᴿ(kx, ky) = Hᵣᴿᴿ(kx, ky)[1, 2]
tᴸᴿ(kx, ky) = Hᵣᴸᴿ(kx, ky)[1, 2]

# Raman circular dichroism (only [1,2] elements for creation of magnons)
χ₁(kx, ky) = abs(tᴿᴸ(kx, ky))^2 + abs(tᴸᴸ(kx, ky))^2 - abs(tᴿᴿ(kx, ky))^2 - abs(tᴸᴿ(kx, ky))^2 # standard definition
χ₂(kx, ky) = 2 * imag((L̃₁₁(kx, ky)[1, 2] - L̃₂₂(kx, ky)[1, 2]) * conj(L̃₁₂(kx, ky)[1, 2])) # LMCs definition

# Define the number of points in kx and ky
Nx = 100
Ny = 100

# Define the kx and ky ranges
kx = range(-π, π, length=Nx)
ky = range(-π, π, length=Ny)
dkx = step(kx)
dky = step(ky)

RCD₁ = [χ₁(kx[i], ky[j])  for i in 1:Nx, j in 1:Ny]
RCD₂ = [χ₂(kx[i], ky[j])/Δ(kx[i], ky[j])^2 for i in 1:Nx, j in 1:Ny]

honeycomb_BZ_integration(RCD₂; Nx=Nx, Ny=Ny)

# momentum-resolved plot
fig = Figure()
ax1 = Axis(fig[1, 1], title="RCD₂", aspect=DataAspect(), xlabel="kx", ylabel="ky")
hm1 = heatmap!(ax1, kx, ky, RCD₂,  colormap=:viridis)
poly!(ax1, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
Colorbar(fig[1, 2], hm1, label="Energy (meV)")
fig 

# frequency-resolved plot
δ(ω) = gaussian_1D(ω; ω₀=0, σ=5e-2)# Dirac delta function approximation
ω = range(0, 20, length=Int(1e2+1))
δ_approx = δ.(ω) 

T_neel = 78 # Néel temperature in Kelvin
temperatures = range(0, T_neel, length=100)
Z = partition.(temperatures; Nx=Nx, Ny=Ny)
lower_band_population_Γ =  boltzman_factor.(Ref(-real(E₁(0, 0))), temperatures) ./ Z
lower_band_population_K = boltzman_factor.(Ref(-real(E₁(K[1], K[2]))), temperatures) ./ Z
lower_band_population_K′ = boltzman_factor.(Ref(-real(E₁(K′[1], K′[2]))), temperatures) ./ Z
upper_band_population_Γ = boltzman_factor.(Ref(real(E₂(0, 0))), temperatures) ./ Z 
upper_band_population_K = boltzman_factor.(Ref(real(E₂(K[1], K[2]))), temperatures) ./ Z
upper_band_population_K′ = boltzman_factor.(Ref(real(E₂(K′[1], K′[2]))), temperatures) ./ Z

# frequency-resolved thermal RCD 
Δ(kx, ky) = real(-E₁(-kx, -ky)+E₂(kx, ky))
Δ₂(kx, ky) = real(-E₁(kx, ky)+E₂(kx, ky))
E12 = [Δ(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
boltzman_factor_lower(T) = [boltzman_factor(-real(E₁(kx[i], ky[j])), T) for i in 1:Nx, j in 1:Ny]
boltzman_factor_upper(T) = [boltzman_factor(real(E₂(kx[i], ky[j])), T) for i in 1:Nx, j in 1:Ny]

χ(ω, T) = honeycomb_BZ_integration( (boltzman_factor_lower(T) .+ boltzman_factor_upper(T) ) .* RCD₂ .*  δ.(ω .- E12) ./ partition(T; Nx=Nx, Ny=Ny) ; Nx=Nx, Ny=Ny)
χ₀(ω) = honeycomb_BZ_integration( RCD₂ .*  δ.(ω .- E12) ; Nx=Nx, Ny=Ny)
χ_ω_T = [χ(ω[i], temperatures[j]) for i in 1:length(ω), j in 1:length(temperatures)]
χ_ω = χ₀.(ω)

# plotting
fig = Figure()
ax_frequency = Axis(fig[1, 1], title="Intensity", xlabel="ω (meV)", ylabel="δ(ω) a.u.")
lines!(ax_frequency, ω, δ_approx , color=:black)
ax_partition = Axis(fig[1, 2], title="Partition Function", xlabel="T (K)", ylabel="Z(T)")
lines!(ax_partition, temperatures, Z, color=:blue)
ax_population_lower = Axis(fig[2, 1], title="Population lower", xlabel="T (K)", ylabel=L"e^{-\beta ε_{-}({\mathbf{k}})}/Z(T)")
lines!(ax_population_lower, temperatures, lower_band_population_Γ, color=:red, label="Γ")
lines!(ax_population_lower, temperatures, lower_band_population_K, color=:green, label="K")
lines!(ax_population_lower, temperatures, lower_band_population_K′, color=:blue, label="K′")
axislegend(ax_population_lower, position=:rc)
ax_population_upper = Axis(fig[2, 2], title="Population upper", xlabel="T (K)", ylabel=L"e^{-\beta ε_{+}({\mathbf{k}})}/Z(T)")
lines!(ax_population_upper, temperatures, upper_band_population_Γ, color=:red, label="Γ")
lines!(ax_population_upper, temperatures, upper_band_population_K, color=:green, label="K")
lines!(ax_population_upper, temperatures, upper_band_population_K′, color=:blue, label="K′")
axislegend(ax_population_upper, position=:rc)
ax_frequency_resolved = Axis(fig[3, 1], title="Frequency-resolved RCD", xlabel="ω (meV)", ylabel="χ(ω, T) (a.u.)")
lines!(ax_frequency_resolved, ω, χ_ω, color=:red, label="")
fig

fig2 = Figure()
ax_frequency_resolved_thermal = Axis(fig2[1, 1], title="Frequency-resolved RCD",xlabel="ω (meV)", ylabel="T (K)")
hm = heatmap!(ax_frequency_resolved_thermal, ω, temperatures, χ_ω_T, colormap=:seismic)
Colorbar(fig2[1, 2], hm, label="χ(ω, T) (a.u.)")
fig2

