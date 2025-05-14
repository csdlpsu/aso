# Contents:
### SU2_files:
- ffd_rae2822_4pts.cfg    : config file for 4 control points RAE2822
- ffd_rae2822_8pts.cfg    : config file for 8 control points RAE2822
- ffd_rae2822_16pts.cfg   : config file for 16 control points RAE2822
- ffd_rae2822_32pts.cfg   : config file for 32 control points RAE2822
- 4cpts_RAE2822_Mesh.su2  : mesh for RAE2822 with 4 control points
- 8cpts_RAE2822_Mesh.su2  : mesh for RAE2822 with 8 control points 
- 16cpts_RAE2822_Mesh.su2 : mesh for RAE2822 with 16 control points
- 32cpts_RAE2822_Mesh.su2 : mesh for RAE2822 with 32 control points 

- config_NACA0012_4pts.cfg  : config file for 4 control points NACA0012
- config_NACA0012_8pts.cfg  : config file for 8 control points NACA0012
- config_NACA0012_16pts.cfg : config file for 16 control points NACA0012
- mesh_NACA0012_4pts.su2    : mesh for NACA0012 with 4 control points
- mesh_NACA0012_8pts.su2    : mesh for NACA0012 with 8 control points
- mesh_NACA0012_16pts.su2   : mesh for NACA0012 with 16 control points

- ffd_oneram6_Euler_12pts.cfg : config file for 12 control points ONERA M6
- mesh_ONERAM6_inv_FFD.su2    : mesh for ONERA M6 with 12 control points

### Python Scripts
- SU2_Wrapper_All_Gradients_RANS.py                   : SU2 Wrapper to run CFD simulations
- Unconstrained_Bayesian_Shape_Optimization.py        : Unconstrained Bayesian Optimization Script
- Constrained_Bayesian_Shape_Optimization.py          : Constrained Bayesian optimization script for the RAE2822 constrained case
- ONERA_M6_Constrained_Bayesian_Shape_Optimization.py : Constrained Bayesian optimization script for the ONERA-M6 constrained case

## Dependencies/Required Packages:
- Stanford University Unstructured (SU2)
- Torch
- BoTorch
- GpyTorch
- NumPy
- Pandas

## To Run:
Unconstrained Cases:
    python Unconstrained_Bayesian_Shape_Optimization.py -mcfg "config_NACA0012_4pts.cfg" -mesh "mesh_NACA0012_4pts.su2" -bsize 0.002 -miters 300 -dim 12 -seed 1

Constrained RAE2822 Airfoil Cases:
    python Constrained_Bayesian_Shape_Optimization.py -mcfg "ffd_rae2822_4pts.cfg" -mesh "4cpts_RAE2822_Mesh.su2" -dim 4 -bsize 0.002 -batch 3 -tol 10e-3

Constrained ONERA-M6 Wing Case:
    python ONERA_M6_Constrained_Bayesian_Shape_Optimization.py -mcfg "ffd_oneram6_Euler_12pts.cfg" -mesh "mesh_ONERAM6_inv_FFD.su2" -dim 12 -bsize 0.00025 -batch 3 -tol 10e-3
