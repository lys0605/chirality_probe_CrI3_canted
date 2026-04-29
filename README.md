# Chirality Probe

> Numerical study of **Raman Circular Dichroism (RCD)** and magnon topology in magnetically ordered materials on the honeycomb lattice.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19884757.svg)](https://doi.org/10.5281/zenodo.19884757)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557c?logo=matplotlib)](https://matplotlib.org/)
[![SciencePlots](https://img.shields.io/badge/SciencePlots-2.0%2B-green)](https://github.com/garrettj403/SciencePlots)
[![GitHub last commit](https://img.shields.io/github/last-commit/lys0605/chirality-probe)](https://github.com/lys0605/chirality-probe/commits/main)

---

## Table of Contents

- [Overview](#overview)
- [Physical systems](#physical-systems)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Usage](#usage)
- [Key physics computed](#key-physics-computed)
- [Unit conventions](#unit-conventions)
- [Module architecture](#module-architecture)
- [Julia code](#julia-code)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the Python source code accompanying the manuscript on chirality probing in magnetic materials. It computes:

- Magnon band structures and Berry curvature maps over the Brillouin zone
- Raman scattering cross-sections and Raman Circular Dichroism (RCD) for circularly polarised light
- Finite-temperature RCD spectra as a function of temperature *T* and DMI strength *D*
- Chern numbers and quantum geometric tensor for canted antiferromagnets

---

## Physical systems

| System | Description | Key scripts |
|--------|-------------|-------------|
| **CrI₃** | Ferromagnetic honeycomb van der Waals insulator | `CrI3/` |
| **Canted AFM** | Canted antiferromagnet on honeycomb lattice | `canted/` |
| **Pump–probe / finite-T** | Time-resolved and finite-temperature RCD | `canted/pump_probe.py`, `canted/finite_temperature_pump_probe.py` |

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.21 |
| Matplotlib | ≥ 3.5 |
| SciPy | ≥ 1.7 |
| SciencePlots | ≥ 2.0 |
| Pillow | ≥ 9.0 |
| sympy | ≥ 1.10 |
| labellines | ≥ 0.5 |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/lys0605/chirality-probe.git
cd chirality-probe

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install numpy matplotlib scipy sympy Pillow scienceplots labellines
```

> **Note — SciencePlots** requires LaTeX to render labels.
> Install a TeX distribution (e.g. MacTeX on macOS, TeX Live on Linux)
> or disable the style by removing `import scienceplots` and the
> `plt.style.context(...)` calls if LaTeX is unavailable.

---

## Repository structure

```
chirality-probe/
│
├── common/                        # Shared library modules
│   ├── honeycomb_lattice.py       # Lattice geometry, BZ mesh, k-paths
│   ├── math_utils.py              # Lineshapes, complex helpers, normalisation
│   ├── mathfuntion.py             # Backward-compat shim → re-exports math_utils
│   ├── model_parameters.py        # Material & physical constants (CrI₃, canted AFM)
│   ├── bose_statistics.py         # Bose–Einstein and Boltzmann statistics
│   └── plot_utils.py              # Plotting helpers and panel builders
│
├── CrI3/                          # CrI₃ ferromagnet scripts
│   ├── CrI3_model.py              # Bond-vector constants + FM_eigs_exact eigensolver
│   ├── CrI3_band.py               # Magnon band structure
│   ├── CrI3_curvature.py          # Berry curvature and quantum metric
│   ├── CrI3_raman_scattering.py   # Raman cross-section and RCD k-maps
│   └── CrI3_pump_probe.py         # Finite-T RCD spectra χ(ω,T) and χ(ω,D)
│
├── canted/                        # Canted antiferromagnet scripts
│   ├── canted_energy_band.py      # Magnon band structure
│   ├── canted_chern_number.py     # Berry curvature and Chern number vs (B, D)
│   ├── canted_curvature.py        # RCD Berry curvature on BZ
│   ├── canted_RCD.py              # RCD amplitudes (exact & LMC methods)
│   ├── canted_raman_cross_section.py  # Raman cross-section on BZ mesh
│   ├── pump_probe.py              # T = 0 frequency-resolved RCD
│   └── finite_temperature_pump_probe.py   # Finite-T RCD χ(ω, T)
│
├── scripts/
│   └── panel_plot.py              # Assemble multi-panel publication figures
│
├── figures/                       # Output figures (auto-generated)
├── LICENSE
└── README.md
```

---

## Usage

All scripts are self-contained and should be run **from the repository root**:

```bash
# CrI3 magnon band structure
python CrI3/CrI3_band.py

# CrI3 Berry curvature (upper + lower bands)
python CrI3/CrI3_curvature.py

# CrI3 Raman cross-section and RCD map
python CrI3/CrI3_raman_scattering.py

# CrI3 finite-temperature RCD spectra (saves CrI3/chi_FM_computed.npz)
python CrI3/CrI3_pump_probe.py

# Canted AFM band structure
python canted/canted_energy_band.py

# Chern number and Berry curvature for canted AFM
python canted/canted_chern_number.py

# T = 0 pump-probe RCD (canted AFM)
python canted/pump_probe.py

# Finite-T RCD (canted AFM)
python canted/finite_temperature_pump_probe.py

# Assemble multi-panel publication figures (requires pre-generated PNGs)
python scripts/panel_plot.py
```

> **Note — `CrI3_pump_probe.py`** loads `CrI3/chi_FM_computed.npz` on startup.
> On a fresh clone this file does not exist; the script will compute and
> save it on the first run. Simply comment out the `np.load(...)` block at
> the top and run the computation cells first.

---

## Key physics computed

| Quantity | Symbol | Script |
|----------|--------|--------|
| Magnon band structure | ε±(**k**) | `CrI3/CrI3_band.py`, `canted/canted_energy_band.py` |
| Berry curvature | Ω(**k**) | `CrI3/CrI3_curvature.py`, `canted/canted_curvature.py` |
| Quantum geometric tensor | g(**k**) | `CrI3/CrI3_curvature.py` |
| Chern number | *C* | `canted/canted_chern_number.py` |
| Raman cross-section | σ(**k**) | `CrI3/CrI3_raman_scattering.py`, `canted/canted_raman_cross_section.py` |
| RCD spectrum | χ(ω) | `canted/canted_RCD.py`, `canted/pump_probe.py` |
| Finite-T RCD | χ(ω, T) | `CrI3/CrI3_pump_probe.py`, `canted/finite_temperature_pump_probe.py` |

---

## Unit conventions

| Quantity | Unit |
|----------|------|
| Energies — CrI₃ | meV |
| Energies — canted AFM | meV, normalised to *J* = 1 meV |
| Temperature | K |
| Boltzmann constant | 8.617 × 10⁻² meV K⁻¹  (`common/bose_statistics.py`) |
| Lattice constant | *a* = 1 (dimensionless) |

Material parameters are centralised in `common/model_parameters.py`:

```python
CrI3 = dict(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2)   # meV
CANTED_AFM = dict(J=1.0, D=0.1, S=5/2)
```

---

## Module architecture

```
common/
  model_parameters.py   ← material & physical constants
        ↑
  bose_statistics.py    ← Bose–Einstein / Boltzmann weights
        ↑
  honeycomb_lattice.py   math_utils.py   plot_utils.py
           ↑                  ↑                ↑
           └──────────────────┴────────────────┘
                              ↑
                        CrI3/CrI3_model.py    ← shared CrI3 Hamiltonian
                              ↑
                    ┌─────────┴──────────┐
              CrI3/ scripts        canted/ scripts
                                   pump_probe.py
                                   finite_temperature_pump_probe.py
                          scripts/panel_plot.py
```

**`common/` module summary**

| File | Contents |
|------|----------|
| `honeycomb_lattice.py` | High-symmetry points `GAMMA, K, K_PRIME, M_POINT`; `bzmesh()`, `bz_integration_honeycomb()`, k-path utilities |
| `math_utils.py` | `Im`, `Re`, `normalize()`, `gaussian_function()`, `lorentzian_function()`, `is_invertible()` |
| `model_parameters.py` | `CrI3`, `CANTED_AFM` parameter dicts; `k_B_meV`, `k_B_eV` |
| `bose_statistics.py` | `bose_einstein(E,T)`, `boltzmann_factor(E,T)`, `occupation_function(E,T)` |
| `plot_utils.py` | `plot()`, `panel()`, `panel_unequal()`, `letter_annotation()`, `plot_lines_with_colorbar()`, `plot_frequency_resolved_RCD()` |
| `mathfuntion.py` | Backward-compatibility shim — re-exports `math_utils` |

---

## Julia code

Higher-performance versions of the Berry curvature and Chern number
calculations are implemented in Julia (`.jl` files in the project root)
for parameter sweeps where the Python k-space loops are too slow.

---

## Citation

If you use this code, please cite it via Zenodo:

```
Ying Shing Liu. (2026). Chirality Probe: Raman Circular Dichroism and magnon topology
on the honeycomb lattice. Zenodo. https://doi.org/10.5281/zenodo.19884757
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19884757.svg)](https://doi.org/10.5281/zenodo.19884757)

---

## License

Copyright (c) 2026 Ying Shing Liu.
Released under the [MIT License](LICENSE).
