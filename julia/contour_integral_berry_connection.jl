using QuadGK, LinearAlgebra, ForwardDiff
using CairoMakie

function linear_interpolate(p₁, p₂, N)
    steps = 1 / N
    return [p₁ .+ t * (p₂ - p₁) for t in 0:steps:1-steps]
end

function circular_loop(center, radius, N)
    angles = range(0, 2π, length=N)
    return [center .+ radius * [cos(angle), sin(angle)] for angle in angles]
end
# ill-defined in overlaps of coordinate patches
A₁(kx, ky) = [im * transpose(conj(u₁(kx, ky))) * ∂u₁_∂kx(kx, ky), 
              im * transpose(conj(u₁(kx, ky))) * ∂u₁_∂ky(kx, ky)]

A₁(v::Vector) = A₁(v[1], v[2])



# K,K' points
rotation_2D(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
R₃ = rotation_2D(2π/3)
R₆ = rotation_2D(2π/6)
K = 2π*[2/3,0]/√3 #K

honeycomb_boundary_path = [
    K,
    R₆*K,
    R₆^2*K,
    R₆^3*K,
    R₆^4*K,
    R₆^5*K,
    K
]

integral = 0.0 + 0.0im 

# create the path
N = 10000
full_path = vcat([
    linear_interpolate(honeycomb_boundary_path[i], honeycomb_boundary_path[i+1], N) for i in 1:length(honeycomb_boundary_path)-1
]...)
full_path = vcat(full_path, [K]) # close the path

# Discrete path integration along the honeycomb boundary
Berry_phase = sum([angle(transpose(conj(v₁(full_path[i]) )) * v₁(full_path[i+1])) for i in 1:length(full_path)-1])
C = Berry_phase/2π
