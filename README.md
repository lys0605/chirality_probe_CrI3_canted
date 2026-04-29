# Chirality Probe

Computational condensed-matter physics project for calculating and visualising
**optical chirality** in magnetically ordered materials via Raman Circular
Dichroism (RCD) and related spectroscopic quantities.

---

## Physical systems

| System | Description | Key scripts |
|--------|-------------|-------------|
| **CrI₃** | Ferromagnetic honeycomb van der Waals insulator | `CrI3_band.py`, `CrI3_curvature.py`, `CrI3_raman_scattering.py`, `CrI3_pump_probe.py` |
| **Canted AFM** | Canted antiferromagnet on honeycomb lattice | `canted_energy_band.py`, `canted_chern_number.py`, `canted_RCD.py`, `canted_curvature.py`, `canted_raman_cross_section.py` |
| **Pump–probe** | Time-resolved / finite-temperature RCD spectroscopy | `pump_probe.py`, `finite_temperature_pump_probe.py` |

---

## Module architecture

The codebase is split into **shared library modules** (lower layers) and
**thin calculation scripts** (upper layer).  Each script imports from the
library modules and is responsible only for setting parameters, running the
calculation, and producing figures.

```
╔══════════════════════════════════════════════════════════════════════╗
║                      CALCULATION SCRIPTS                             ║
║                                                                      ║
║  CrI3_band.py            canted_energy_band.py    pump_probe.py      ║
║  CrI3_curvature.py       canted_chern_number.py   finite_temp_pp.py  ║
║  CrI3_raman_scattering.py  canted_RCD.py          panel_plot.py      ║
║  CrI3_pump_probe.py      canted_curvature.py                         ║
║                          canted_raman_cross_section.py               ║
╠══════════════════════════════════════════════════════════════════════╣
║                        LIBRARY MODULES                               ║
║                                                                      ║
║  parameters.py      ─── material constants & physical constants      ║
║  │                       CrI3 {J1, J2, J3, D, Az, S}                 ║
║  │                       CANTED_AFM {J, D, S}                        ║
║  │                       k_B_meV, k_B_eV                             ║
║  │                                                                   ║
║  honeycomb_lattice.py ── lattice geometry & Brillouin zone           ║
║  │                       High-symmetry pts: GAMMA, K, K_PRIME,       ║
║  │                                          M_POINT                  ║
║  │                       k-mesh:       bzmesh()                      ║
║  │                       k-path:       get_kvectors(), get_path(),   ║
║  │                                     group_kvectors(),             ║
║  │                                     get_total_path()              ║
║  │                       BZ integrals: bz_integration_honeycomb()    ║
║  │                       BZ boundary:  honeycomb_bz()                ║
║  │                       Rotation:     rotation2D()                  ║
║  │                                                                   ║
║  math_utils.py      ─── pure mathematical utilities                  ║
║  │                       Im, Re            (complex helpers)         ║
║  │                       normalize()                                 ║
║  │                       gaussian_function()                         ║
║  │                       lorentzian_function()                       ║
║  │                       is_invertible(), print_matrix()             ║
║  │                                                                   ║
║  thermal.py         ─── temperature-dependent statistics             ║
║  │                       bose_einstein(E, T)       [meV / K]         ║
║  │                       boltzmann_factor(E, T)                      ║
║  │                       occupation_function(E, T)                   ║
║  │                                                                   ║
║  plot_utils.py      ─── visualisation helpers                        ║
║                          plot(), panel(), panel_unequal()            ║
║                          letter_annotation()                         ║
║                          plot_lines_with_colorbar()                  ║
║                          plot_frequency_resolved_RCD()               ║
║                          plot_frequency_temperature_resolved_RCD()   ║
╠══════════════════════════════════════════════════════════════════════╣
║                      CrI₃ MODEL MODULE                               ║
║                                                                      ║
║  CrI3_model.py      ─── core CrI3 FM magnon Hamiltonian              ║
║                          NN, NNN, NNNN  (bond-vector constants)      ║
║                          FM_eigs_exact(k, ...)  Bogoliubov solver    ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Dependency graph (arrows = "imports from")

```
parameters.py
      ↑
   thermal.py
      ↑
honeycomb_lattice.py    math_utils.py    plot_utils.py
         ↑                   ↑                ↑
         └───────────────────┴────────────────┘
                             ↑
                       CrI3_model.py   ← shared CrI3 Hamiltonian
                             ↑
                   [CrI3 calculation scripts]

                   [canted / pump-probe scripts]  ← no CrI3_model dep
```

> `mathfuntion.py` is kept as a backward-compatibility shim that
> re-exports everything from `math_utils.py`.

---

## Key physics computed

| Quantity | Symbol | Where |
|----------|--------|-------|
| Magnon band structure | ε±(k) | `CrI3_band.py`, `canted_energy_band.py` |
| Berry curvature | Ω(k) | `CrI3_curvature.py`, `canted_curvature.py` |
| Quantum geometric tensor | g(k) | `CrI3_curvature.py` |
| Chern number | C | `canted_chern_number.py` |
| Raman cross-section | σ(k) | `CrI3_raman_scattering.py`, `canted_raman_cross_section.py` |
| RCD spectrum | χ(ω) | `canted_RCD.py`, `pump_probe.py` |
| Finite-T RCD | χ(ω, T) | `finite_temperature_pump_probe.py` |

---

## Unit conventions

| Quantity | Unit |
|----------|------|
| Energies (CrI₃) | meV |
| Energies (canted AFM) | meV, normalised to J = 1 meV |
| Temperature | K |
| Boltzmann constant | 8.617 × 10⁻² meV K⁻¹ — see `thermal.py` and `parameters.py` |
| Lattice constant | a = 1 (dimensionless) |

---

## File index

```
── Library modules ──────────────────────────────────────────────────────
parameters.py                 material & physical constants
math_utils.py                 lineshapes, normalisation, complex helpers
thermal.py                    Bose–Einstein, Boltzmann weights
honeycomb_lattice.py          lattice geometry, BZ mesh, k-paths
plot_utils.py                 all plotting helpers
mathfuntion.py                backward-compat shim → re-exports math_utils

── CrI₃ scripts ─────────────────────────────────────────────────────────
CrI3_model.py                 bond-vector constants (NN/NNN/NNNN) and
                              FM_eigs_exact — shared by all CrI3 scripts
CrI3_band.py                  magnon band structure along high-sym path
CrI3_curvature.py             Berry curvature and quantum metric on BZ
CrI3_raman_scattering.py      Raman cross-section and RCD k-maps
CrI3_pump_probe.py            finite-T RCD spectra χ(ω,T) and χ(ω,D)

── Canted AFM scripts ───────────────────────────────────────────────────
canted_energy_band.py         magnon band structure
canted_chern_number.py        Berry curvature and Chern number vs (B, D)
canted_curvature.py           RCD Berry curvature on BZ
canted_RCD.py                 RCD amplitudes (exact & LMC methods)
canted_raman_cross_section.py Raman cross-section on BZ mesh

── Pump–probe / thermal scripts ─────────────────────────────────────────
pump_probe.py                 T = 0 frequency-resolved RCD
finite_temperature_pump_probe.py  finite-T RCD χ(ω, T)

── Figure assembly ──────────────────────────────────────────────────────
panel_plot.py                 assemble multi-panel publication figures
```

---

## Julia code

Higher-performance versions of the Berry curvature and Chern number
calculations are implemented in Julia (`.jl` files in the project root)
for parameter sweeps where the Python k-space loops are too slow.

---

## License

MIT License

Copyright (c) 2026 lys0605

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
