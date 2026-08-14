# Poroelastic Half-Space Deformation

A Python implementation of an analytical solution for deformation and strain induced by pore-pressure changes within a subsurface inclusion embedded in an elastic half-space.

<p align="center">
  <img src="assets/model_geometry.png" width="750" alt="Geometry of the poroelastic half-space model">
</p>

The model combines **Eshelby's eigenstrain formulation** with **Mindlin half-space solutions** and the corresponding **Chen equations** to calculate three-dimensional displacement and strain resulting from a pressurized subsurface inclusion.

The formulation provides a computationally efficient alternative to fully numerical poroelastic models and is designed for forward modeling, sensitivity analysis, parameter estimation, and comparison with field strain observations.

## Model

A pore-pressure perturbation within the subsurface inclusion produces a poroelastic eigenstrain. Eshelby's inclusion theory describes the resulting deformation in an infinite elastic medium, while the Mindlin–Chen formulation accounts for the presence of the traction-free surface.

The inclusion geometry is characterized by the semi-axes $a$, $b$, and $c$, its depth $h$, and its orientation relative to the observation coordinate system.

The implementation predicts components of the three-dimensional displacement and strain tensors at arbitrary observation locations.

## Repository Structure

```text
.
├── assets/          # Documentation images and diagrams
├── data/            # Input and observational datasets
├── src/             # Core analytical model
├── scripts/         # Reproducible analyses and model runs
├── tests/           # Model verification and tests
├── figures/         # Publication figures
├── docs/            # Theory and additional documentation
└── README.md
```

Generated numerical results are not stored in the source repository. Analysis scripts write their outputs to designated results directories.

## Installation

The model requires Python and standard scientific Python packages. A reproducible environment and complete installation instructions will be provided as the repository is prepared for public release.

## Usage

Examples for running the forward model, reproducing the analyses, and generating the figures presented in the associated publication will be provided in `scripts/` and `docs/`.

## Reproducibility

The repository is being organized to reproduce the analyses associated with the accompanying research paper.

Computationally inexpensive calculations can be executed locally, while larger sensitivity analyses, parameter sweeps, and inverse problems can be run on high-performance computing systems.

## Citation

If you use this model or code in published work, please cite the associated paper.

Citation information will be added upon publication.

## License

License information will be added prior to public release.
