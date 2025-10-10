using CairoMakie
using LinearAlgebra


# Define the parameters for the 1D chain model
t₁ = 0.1 # 1st n.n. hopping parameter (# mev)
δ = 0.01 # non-hermitian parameter
α = 0.0 # turn on the non-hermitian term
t₁L = t₁ - α*δ
t₁R = t₁ + α*δ

t₂ = 0 # 2nd n.n. hopping parameter (# mev) 
t₂L = t₂ - α*δ
t₂R = t₂ + α*δ

U = 0.0 # uniform on-site potential (# mev)

# Open boundary conditions imposed
N = 100 # number of sites in the 1D chain
on_site_open = fill(U, N)
nearest_neighbour_L_open = fill(t₁L, N-1)
nearest_neighbour_R_open = fill(t₁R, N-1)
next_nearest_neighbour_L_open = fill(t₂L, N-2)
next_nearest_neighbour_R_open = fill(t₂R, N-2)


H₀_open = diagm(0 => on_site_open, 1 => nearest_neighbour_L_open, -1 => nearest_neighbour_R_open,
                2 => next_nearest_neighbour_L_open, -2 => next_nearest_neighbour_R_open)
eigenvalues_open, eigenvectors_open = eigen(H₀_open)
eigenvector_open = eigenvectors_open ./ LinearAlgebra.norm(eachcol(eigenvectors_open)) # Normalize the first eigenvector

# Plot eigenvalues for open boundary conditions
fig_open = Figure()
ax_open = Axis(fig_open[1, 1], title="Eigenvalues for Open Boundary Conditions")
scatter!(ax_open, 1:N, real.(eigenvalues_open), color=:blue, label="Eigenvalues")
fig_open
# Plot eigenvectors for open boundary conditions
probabilities_open = abs2.(eigenvectors_open)
fig_open_vec = Figure()
ax_open_vec = Axis(fig_open_vec[1, 1], title="Eigenvectors for Open Boundary Conditions")
prob_hm_open = heatmap!(ax_open_vec, 1:N, 1:N, probabilities_open)
Colorbar(fig_open_vec[1, 2], prob_hm_open, label="Probability Density")
fig_open_vec

# Periodic boundary conditions imposed
function H₀_periodic(kx, ky)
    # Define the Hamiltonian for the 1D chain with periodic boundary conditions
    return [0 0; 0 0]
end
# Plot

eachcol(eigenvectors_open)[1]

normalize.(eachcol(eigenvectors_open))

maximum(probabilities_open)

# SSH (intracell and intercell)