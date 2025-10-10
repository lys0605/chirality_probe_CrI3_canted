using ForwardDiff, LinearAlgebra
using CairoMakie, LaTeXStrings, GeometryBasics
import Meshes


# Canted AFM on honeycomb lattice
a = 1
n_n = [[0, 1], [-√3/2, -1/2], [√3/2, -1/2]] .* a
next_n_n = [[-√3/2, -3/2], [√3, 0], [-√3/2, 3/2]] .* a

ϕₖ(kx, ky; J=1, S=5/2) = 2*J*S*(exp(1im*dot([kx,ky], n_n[1])) + exp(1im*dot([kx,ky], n_n[2])) + exp(1im*dot([kx,ky], n_n[3])))
λₖ(kx, ky; D=0.1, S=5/2, s=0.6) = 4*D*S*s*(sin(dot([kx,ky], next_n_n[1])) + sin(dot([kx,ky], next_n_n[2])) + sin(dot([kx,ky], next_n_n[3])))
Δₖ(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt(λₖ(kx,ky; D, S, s)^2 + (s^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S)))))

H(kx, ky; J=1, D=0.1, S=5/2, s=0.6) = [6*J*S+λₖ(kx,ky) -s^2*conj(ϕₖ(kx,ky)) 0 (1-s^2)*conj(ϕₖ(kx,ky));
                                       -s^2*ϕₖ(kx,ky) 6*J*S-λₖ(kx,ky) (1-s^2)*ϕₖ(kx,ky) 0;
                                       0 (1-s^2)*conj(ϕₖ(kx,ky)) 6*J*S-λₖ(kx,ky) -s^2*conj(ϕₖ(kx,ky));
                                       (1-s^2)*ϕₖ(kx,ky) 0 -s^2*ϕₖ(kx,ky) 6*J*S+λₖ(kx,ky)]
ϵ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₃(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -sqrt((6*J*S+Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))
ϵ₄(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -sqrt((6*J*S-Δₖ(kx,ky; J, D, S, s))^2 - (1-s^2)^2*(ϕₖ(kx,ky; J, S)*conj(ϕₖ(kx,ky; J, S))))

# eigenvectors
sinhχ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = (6*J*S+Δₖ(kx,ky; J, D, S, s) - ϵ₁(kx,ky; J, D, S, s)) / ((1-s^2)*abs(ϕₖ(kx,ky; J, S)))
sinhχ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = -(6*J*S-Δₖ(kx,ky; J, D, S, s) - ϵ₂(kx,ky; J, D, S, s)) / ((1-s^2)*abs(ϕₖ(kx,ky; J, S)))
coshχ₁(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt(1 + sinhχ₁(kx,ky; J, D, S, s)^2)
coshχ₂(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = sqrt(1 + sinhχ₂(kx,ky; J, D, S, s)^2)
cosψ(kx,ky; J=1, D=0.1, S=5/2, s=0.6) = λₖ(kx,ky; D, S, s) / Δₖ(kx,ky; J, D, S, s)
φ(kx ,ky) = -1im*log(ϕₖ(kx,ky; J=1, S=5/2)/sqrt(ϕₖ(kx,ky; J=1, S=5/2)*conj(ϕₖ(kx,ky; J=1, S=5/2))))
u₁(kx, ky; J=1, D=0.1, S=5/2, s=0.6) =  []

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

λ_values = [λₖ(kx[i], ky[j]) for i in 1:Nx, j in 1:Ny]
ϵ₁_values = [real(ϵ₁(kx[i], ky[j])) for i in 1:Nx, j in 1:Ny]

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

fig = Figure()
ax_heat_1, hm_1 = plot_heatmap_on_BZ(fig, kx, ky, ϵ₁_values; 
                                    axis=[1,1], title=L"\textbf{Quantum metric} $2g_{μν}\Delta_{g}^2$", 
                                    xlabel=L"k_x", ylabel=L"k_y", 
                                    xlabelsize=18, ylabelsize=18, 
                                    colormap=:viridis, coloarmaplabel="Energy (meV)")
poly!(ax_heat_1, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 *K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
fig

# ax_heat_2 = Axis(fig[1, 3], title=L"\textbf{Berry curvature} $\Omega_{xy}$", 
#                 aspect=DataAspect() , xlabel=L"k_x", ylabel=L"k_y",
#                 xlabelsize=18, ylabelsize=18)
# hm_2 = heatmap!(ax_heat_2, kx, ky, Ω₋, colormap=:viridis)
# poly!(ax_heat_2, Point2f[K, rot * K, rot^2 * K, rot^3 * K, rot^4 * K, rot^5 * K], color=(:black, 0), alpha=0.5, strokecolor=:black, strokewidth=1)
# Colorbar(fig[1, 4], hm_2, label="Energy (meV)")

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

sum(real.(Ω)[honeycomb_mask])*dkx*dky/(2π)

findmax([Ω[i,j] == reduce(min, Ω) ? 1 : 0 for i in 1:Nx, j in 1:Ny])



            