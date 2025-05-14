
'''
This script optimizes the an airfoil for drag coefficient unconstrained using SingleTaskGP and the ExpectedImprovement acquisition function.

Using the NACA0012 airfoil as an example

Dependent files:
    SU2_Wrapper_Gradients.py : SU2 wrapper that acts as the objective function by running CFD using supplied DV_VALUE information
    config_NACA0012.cfg      : config file for NACA0012 airfoil 
    mesh_NACA0012.su2        : mesh for NACA0012 that has been set up with the proper FFD box

Outputs:
    DV_VALUE.npy            : NumPy array containing an array of DV_VALUE information that is passed to the SU2_Wrapper_Gradients.py script
    Best_Shape.npy          : NumPy array containing the DV_VALUE combination that results in the best drag
    Best_Drag.npy           : Value of the lowest drag found
    Best_Index.npy          : value of the index of the lowest drag so it can be located from the overall data
    train_y_tensor.npy      : NumPy array of the complete train_y tensor 
    train_x_tensor.npy      : NumPy array of the complete train_x tensor
    surface_deformed_#.vtu  : Paraview file for the airfoil deformed surface at iteration #
    ffd_boxes_def_#.vtk     : Paraview file for the deformed ffd box at iteration 

Running:
    python Unconstrained_Bayesian_Shape_Optimization.py -mcfg "config_NACA0012_4pts.cfg" -mesh "mesh_NACA0012_4pts.su2" -bsize 0.002 -miters 300 -dim 12 -seed 1

'''



## Bayesian Section ##
import numpy as np
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import subprocess
import argparse
import warnings
import time
from botorch.utils.transforms import standardize, normalize, unnormalize

warnings.filterwarnings('ignore')

## Running Objective Function ##
def objective_function(dv_value,main_cfg,mesh_name):
    print(f"dv_value is:\n{dv_value}\n")
    np.save("DV_VALUE.npy",dv_value)

    # Running SU2 to get the drag output by calling the SU2 Wrapper script
    
    objective_function_command=['python', 'SU2_Wrapper_All_Gradients_RANS.py','-mcfg', main_cfg,'-mesh', mesh_name,'-dv', 'DV_VALUE.npy','-np', '48','-si', '0','-cmin','-8','-gflag',"F",'-mno','0.729','-reno','6500000','-aoa','2.31']
    print("Running Objective Function:\n")
    result=subprocess.run(objective_function_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)
    print("Finished Objective Function:\n")
    drag=np.load("Drag_Coefficient.npy")
    print(f"Drag from objective function is: {drag}\n")
    drag=-drag
    return drag.item()
    

## Bayesian Section ##

def BayesianOptimizationLoop(main_cfg,mesh_name,dim,bsize,max_iters,seed):
    torch.manual_seed(seed)
    ntrain=dim*10 #Number of training points
    

    print(f"\n\n############################## INITIALIZATION ##############################\n\n")

    # Define the search space for the DV_VALUEs
    bounds = torch.tensor([
        [-bsize] * dim,  # lower bounds 
        [bsize] * dim   # upper bounds
        ], dtype=torch.float64)
    
    gp_bounds= torch.stack([torch.zeros(dim), torch.ones(dim)])
    
    # Initializing data
    train_x = torch.rand(ntrain, dim) * (bounds[1] - bounds[0]) + bounds[0]  # Initial random samples
    
    
    #Normalizing train_x to [0,1]
    train_x_normalized=normalize(train_x, bounds=bounds)
    
    #Generating Train_y
    train_y= torch.tensor([objective_function(x.tolist(),main_cfg,mesh_name) for x in train_x], dtype=torch.float64).unsqueeze(-1)

    #Standardizing train_y  
    train_y_standardized = standardize(train_y)
    
    print(f"Train_X initiated as: \n{train_x}\n")
    print(f"Train_x_normalized initiated as: \n{train_x_normalized}\n")
    print(f"Train_Y initiated as: \n{train_y}\n")
    print(f"Train_y Passed to GP (standardized) initiated as: \n{train_y_standardized}\n")

    NUM_RESTARTS=20
    RAW_SAMPLES=1024
    Q=1

    # Bayesian Optimization Loop
    for iteration in range(max_iters):  
        print(f"\n\n############################## ITERATION {iteration+1} ##############################\n\n")
        # Fit the GP model
        gp = SingleTaskGP(train_x_normalized, train_y_standardized)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Define the acquisition function
        EI= ExpectedImprovement(model=gp, best_f=train_y.max(), maximize=True)
        
        # Optimize the acquisition function to get new candidate
        print("Calculating new_x\n")
        new_x, _ = optimize_acqf(
            acq_function=EI,
            bounds=gp_bounds,
            q=Q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        #unnormalizing new_x
        new_x_un=unnormalize(new_x,bounds)
        #getting new_y
        new_y = torch.tensor([objective_function(new_x_un.tolist(),main_cfg,mesh_name)]).reshape(-1, 1)
        #updating train_x
        train_x=torch.cat([train_x,new_x_un])
        train_x_normalized=torch.cat([train_x_normalized,new_x])
        #updating train_y
        train_y=torch.cat([train_y,new_y])
        train_y_standardized=standardize(train_y)

        #Print Statements
        print("New_x is:\n",new_x)
        print("New_x_un is:\n",new_x_un)
        print(f"new_y is: \n{new_y}\n")
        
        #Copying files for each iteration 
        subprocess.run(["cp", "ffd_boxes_def_0.vtk", "ffd_boxes_def_" + str(iteration) + ".vtk"])
        subprocess.run(["cp", "DRAG_Gradient.npy", "DRAG_Gradient_" + str(iteration) + ".npy"])
        subprocess.run(["cp", "LIFT_Gradient.npy", "LIFT_Gradient_" + str(iteration) + ".npy"])
        subprocess.run(["cp", "MOMENT_Z_Gradient.npy", "MOMENT_Z_Gradient_" + str(iteration) + ".npy"])
        subprocess.run(["cp", "surface_deformed.vtu", "surface_deformed_" + str(iteration) + ".vtu"])
        
        print(f"\nIteration {iteration + 1}: Best drag coefficient so far = {train_y.max()}\n")
           
    # Best found configuration
    best_idx=train_y.argmax()
    best_x = train_x[best_idx]
    best_y =train_y[best_idx]
    best_y=-best_y #changing it to positive

    #Saving best parameters
    np.save('Best_Shape',best_x)
    np.save('Best_Drag',best_y)
    np.save('Best_Index',best_idx)
    np.save('train_y_tensor',train_y.numpy())
    np.save('train_x_tensor',train_x.numpy())

    #Printing final information
    print("Train_x Tensor:\n",train_x)
    print("Train_y Tensor:\n",train_y)
    print("Best FFD control points:", best_x.tolist())
    print("Minimum drag coefficient:", best_y)
    print("Best Index: ", best_idx-ntrain)

#Parser Block

#Setting parser arguments
parser = argparse.ArgumentParser()

parser.add_argument("-mcfg",  "--main_cfg",   type=str,   default='ffd_rae2822_4pts.cfg',   help='Path to the main cfg file')
parser.add_argument("-mesh",  "--mesh_su2",   type=str,   default="4cpts_RAE2822_Mesh.su2", help="Path to the SU2 mesh file")
parser.add_argument("-dim",   "--dimensions", type=int,   default=4,                        help='Number of control points for the FFD box')
parser.add_argument("-bsize", "--bound_size", type=float, default=0.002,                    help="Bound max/min of the control points")
parser.add_argument("-miters", "--max_iters", type=int,   default=100,                      help="Number of iterations")
parser.add_argument("-seed", "--seed",        type=int,   default=1,                        help="Torch seed for randomization")

args = parser.parse_args()

#Saving Inputs from parser to variables
main_cfg=args.main_cfg
mesh_name=args.mesh_su2
dim=args.dimensions
bsize=args.bound_size
max_iters=args.iterations
seed=args.seed

# Running:
print("START")
BayesianOptimizationLoop(main_cfg,mesh_name,dim,bsize,max_iters,seed)  
