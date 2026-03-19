using Random # Standard library for random numbers
using Statistics # For mean/std
using LinearAlgebra

# 1. DEFINE TYPES
# Define structures to hold physical parameters.
# 'struct' is immutable (fast), 'mutable struct' is changeable (slower).
struct IsingParams
    L::Int          # Length of chain
    J::Float64      # Coupling constant
    T::Float64      # Temperature in k_B units
    steps::Int      # Monte Carlo steps
end

# 2. CORE LOGIC (The "Kernel")
# This function does the heavy lifting. In Julia, we can use loops freely!
# Unlike Python, Julia loops are as fast as C.
function metropolis_step!(spins::Vector{Int}, params::IsingParams)
    L = params.L
    beta = 1.0 / params.T
    
    # Iterate L times (one "Sweep")
    for _ in 1:L
        # Pick random site
        i = rand(1:L)
        
        # Periodic Boundary Conditions
        # In Julia, arrays are 1-indexed (like Math, unlike Python)
        left  = spins[mod1(i-1, L)]
        right = spins[mod1(i+1, L)]
        
        # Compute Energy Change (Delta E)
        # If we flip s[i], the change is 2 * s[i] * neighbors
        dE = 2.0 * params.J * spins[i] * (left + right)
        
        # Metropolis Acceptance Criterion
        # 1. If dE < 0, system lowers energy -> Accept (flip)
        # 2. If dE > 0, accept with prob exp(-beta * dE)
        if dE <= 0 || rand() < exp(-dE * beta)
            spins[i] *= -1 # Flip the spin
        end
    end
end

# 3. DRIVER FUNCTION
function run_simulation(params::IsingParams)
    # Initialize random spins (+1 or -1)
    spins = rand([-1, 1], params.L)
    
    magnetization = Float64[] # Empty array to store measurements
    
    for step in 1:params.steps
        metropolis_step!(spins, params)
        
        # Measurement: Average magnetization per spin
        # We perform measurement every 100 steps to reduce correlation
        if step % 100 == 0
            m = abs(sum(spins)) / params.L
            push!(magnetization, m)
        end
    end
    
    return mean(magnetization)
end

# --- RUNNING IT ---
# Define the physics
p = IsingParams(100, 1.0, 0.1, 50000) # L=100, J=1.0, T=2.0 k_B units, 50k steps

# Run (The first time you run this, Julia compiles it. Second run is instant.)
println("Compiling and Running...")
avg_mag = run_simulation(p)
println("Average Magnetization: $avg_mag")