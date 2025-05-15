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
- SU2_Wrapper_All_Gradients_RANS.py                   : SU2 Wrapper to run 2D Viscous CFD simulations
- SU2_Wrapper_All_Gradients_EULER_3D.py               : SU2 Wrapper to run 3D Inviscid CFD simulations
- SU2_Wrapper_All_Gradients_EULER_2D.py               : SU2 Wrapper to run 2D Inviscid CFD simulations
- Gradient_Optimization_NACA_#pts.py             : Gradient Optimization Scripts for unconstrained NACA0012 (#: 4/8/16)
- Gradient_Optimization_uncon_RAE_#pts.py        : Gradient Optimization Scripts for unconstrained RAE2822 (#: 4/8/16/32)
- Gradient_Optimization_con_RAE_#pts.py        : Gradient Optimization Scripts for constrained RAE2822 (#: 4/8/16)
- Gradient_Optimization_uncon_ONERA_12pts.py     : Gradient Optimization Script for unconstrained ONERAM6
- Gradient_Optimization_con_ONERA_12pts.py     : Gradient Optimization Script for constrained ONERAM6

## Dependencies/Required Packages:
- Stanford University Unstructured (SU2)
- argparse
- csv
- re
- NumPy
- Pandas
- minimize (from scipy.optimize)
- Bounds   (from scipy.optimize)

## To Run:
    python Gradient_Optimization_##_#pts.py -np 144 (##: NACA/uncon_RAE/con_RAE/uncon_ONERA/con_ONERA)