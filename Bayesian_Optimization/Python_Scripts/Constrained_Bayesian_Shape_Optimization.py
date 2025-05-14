# Constrained Bayesian Optimization Based on SCBO 
"""
A script that uses Bayesian Optimization to minimize drag of a constrained problem with inequality constraints on moment and thickness, and an equality constraint on lift.

Input:
- mcfg: [Path to] SU2 config file
- mesh: [Path to] SU2 mesh file
- dim: Number of control points for the FFD box
- bsize: Bound max/min of the control points
- batch: Optimization Batch Size
- tol: Equality Constraint Tolerance

Output:
- train_y_tensor.npy : negative drag history
- train_x_tensor.npy : ffd box deformation history (unnormalized by the bounds)
- Constraint_1.npy : History of the Thickness Constraint
- Constraint_2.npy : History of the Moment Constraint
- Constraint_3.npy : History of the Lift Constraint

To run:
python Constrained_Bayesian_Shape_Optimization.py -mcfg "ffd_rae2822_4pts.cfg" -mesh "4cpts_RAE2822_Mesh.su2" -dim 4 -bsize 0.002 -batch 3 -tol 10e-3
"""
import math
import os
import warnings
from dataclasses import dataclass
import argparse

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll

from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Rosenbrock
from botorch.utils.transforms import unnormalize
import botorch 
import numpy as np
import subprocess

botorch.settings.debug(True)

warnings.filterwarnings("ignore")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting GPU or CPU if not avail
device=torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

#Parser Block

#Setting parser arguments
parser = argparse.ArgumentParser()

parser.add_argument("-mcfg",  "--main_cfg",   type=str,   default='ffd_rae2822_4pts.cfg',   help='Path to the main cfg file')
parser.add_argument("-mesh",  "--mesh_su2",   type=str,   default="4cpts_RAE2822_Mesh.su2", help="Path to the SU2 mesh file")
parser.add_argument("-dim",   "--dimensions", type=int,   default=4,                        help='Number of control points for the FFD box')
parser.add_argument("-bsize", "--bound_size", type=float, default=0.002,                    help="Bound max/min of the control points")
parser.add_argument("-batch", "--batch_size", type=int,   default=3,                        help="Optimization Batch Size")
parser.add_argument("-tol",   "--tolerance",  type=float, default=10e-3,                    help="Equality Constraint Tolerance")

args = parser.parse_args()

#Saving Inputs from parser to variables
main_cfg=args.main_cfg
mesh_name=args.mesh_su2
dim=args.dimensions
bsize=args.bound_size
batch_size=args.batch_size
tolerance=args.tolerance

bounds = torch.tensor([
        [-bsize] * dim,  # lower bounds 
        [bsize] * dim   # upper bounds
        ], dtype=torch.float64)

n_init = dim*10
max_cholesky_size = float("2000")  # Always use Cholesky

# When evaluating the function, we must first unnormalize the inputs since we will use normalized inputs x in the main optimizaiton loop
def eval_objective(dv_value,i): #This is an objective function that gets the value at the point
    dv_value=unnormalize(dv_value,bounds=bounds)
    print(f"dv_value is:\n{dv_value}\n")
    dv_value=dv_value.tolist()
    np.save("DV_VALUE.npy",dv_value)
    objective_function_command=['python', 'SU2_Wrapper_All_Gradients_RANS.py','-mcfg', main_cfg,'-mesh', mesh_name,'-dv', 'DV_VALUE.npy','-np', '48','-si', '0','-cmin','-8','-gflag',"F",'-mno','0.734','-reno','6500000','-aoa','2.79']
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
    subprocess.run(["cp", "Moment_Coefficient.npy", "Moment_Coefficient_"+str(i)+".npy"])
    subprocess.run(["cp", "Lift_Coefficient.npy", "Lift_Coefficient_"+str(i)+".npy"])
    subprocess.run(["cp", "surface_deformed.vtu", "surface_deformed_"+str(i)+".vtu"])
    subprocess.run(["cp", "ffd_boxes_def_0.vtk", "ffd_def_"+str(i)+".vtk"])
    subprocess.run(["cp", "Thickness.npy", "Thickness"+str(i)+".npy"])
    thickness=np.load("Thickness"+str(i)+".npy") #Getting Thickness
    moment=np.load("Moment_Coefficient_"+str(i)+".npy")
    lift=np.load("Lift_Coefficient_"+str(i)+".npy")
    cons1=c1(thickness)
    cons2=c2(moment)
    cons3=c3(lift) #Lift Equality Constraint
    return drag.item(),cons1,cons2,cons3

##BO Convention C<=0 is FEASIBLE

#Thickness Constraint
def c1(x):
    return .12 - x #thickness greater than .12

#Moment Constraint
def c2(x):
    return x-0.092 #Moment less than .092

#Lift Equality Constraint C_l = 0.75
def c3(x):
    error=abs(x-.75)
    return error - tolerance

# Define TuRBO Class
@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 5  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = 20 #Set Failure Tolerance Here

def update_tr_length(state: ScboState):
    # Update the length of the trust region 
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 1.5
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state

#Function to get the best index 
def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    print("getting the best index: \n")
    is_feas = (C <= 0).all(dim=-1) #Checking if the solution is feasible by checking if the constraints tensor is all less than 0 in the final row/dimension. Returns a True boolean if all are less than 0, returns a False boolean otherwise
    print("is_feas is:",is_feas)
    # Choosing best feasible candidate
    #If statement runs when there is a feasible solution somewhere
    if is_feas.any():  #Checks if any of the solutions are feasible (if True appears in the feasibility variable)
        score = Y.clone() #Copies the tensor Y into score, where Y contains the objective function value at the given solution points
        score[~is_feas] = -float("inf") #At the index of all False values in Feasible variable, it sets the objective function value to negative infinity such that it will not be found as the best point
        print("found a successful candidate, with index: ",score.argmax())
        return score.argmax() #returning the index of the maximum value of the score, AKA the best function value
    #Returns the closest to feasible solution index if no feasible solutions are found
    print("Did not find a successful candidate, returning index: ",C.clamp(min=0).sum(dim=-1).argmin())
    return C.clamp(min=0).sum(dim=-1).argmin() #sets all negatives to 0, then sums along the final dimension, and outputs the index of the minimum value, which is the closest-to-feasible point


def update_state(state, Y_next, C_next):
    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next) #Calling the best index function to get the best feasible or closest to feasible index
    y_next, c_next = Y_next[best_ind], C_next[best_ind] #Setting the constraints and y_next to the best feasible point from the batch of points and solutions

    if (c_next <= 0).all(): #checking if all of the constraints are met and running 
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value) #setting a threshold of improvement that the function needs to have improved by for it to accept the new solution as a success
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any(): #If the new_y is greater than the improvement threshold or if the best_constraint_values are greater than 0 anywhere, running
            state.success_counter += 1 #incrementing success counter
            state.failure_counter = 0 #resetting failure counter
            state.best_value = y_next.item() #Setting the best_value to the current new_y
            state.best_constraint_values = c_next #setting the best_constraint_values to the new constraints
        else:
            state.success_counter = 0 #resetting success
            state.failure_counter += 1 #incrementing failure
    else: #Running if no new candidate point is feasible
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1) #Setting all negatives in the constraints to 0, and summing over the last dimension
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1) #Doing the same to the best_constraint_values
        if total_violation_next < total_violation_center: #Running if the new point feasibility constraint violations are less than that of the old point's 
            state.success_counter += 1 #incrementing the success counter (getting closer to feasibility)
            state.failure_counter = 0 #resetting failure
            state.best_value = y_next.item() #setting the best_value as the current value
            state.best_constraint_values = c_next #setting the best_constraint values as the new constraints
        else: #if the new point is less feasible than the old point
            state.success_counter = 0 #Resetting the success counter
            state.failure_counter += 1 #Incrementing the failure counter

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state) #updating the trust region based on the last iteration
    return state #returning the new state

# Define example state
state = ScboState(dim=dim, batch_size=batch_size)
print(state)

#Method to get train_x initialized
def get_initial_points(dim, n_pts, seed=0): 
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed) #Using sobol engine to get a good spread of points that are still scrambled, generating an instance with then dimensions dim
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device) # Using the sobol instance to draw initial points based on the number of points given where the result is a tensor or array of size (n_pts, dim)
    return X_init 

# Generating a batch of candidates for SCBO 
def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the Trust Region bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C) #getting the best index
    x_center = X[best_ind, :].clone() #Getting the center of the trust region based on the best point found
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0) #setting the lower bounds, but still within 0 to 1
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0) #setting the upper bounds, but still within 0 to 1

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device) #setting perturbations
    pert = tr_lb + (tr_ub - tr_lb) * pert #setting the perturbations to the right bounds

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0) #creating probability of being perturbed
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb #setting the mask as a random tensor of the proper size
    ind = torch.where(mask.sum(dim=1) == 0)[0] #converting parts of the mask to 0
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1 #Converting parts of the mask to 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone() #expanding the central point to a tensor of the proper dimensions where each new instance is the same point as the center
    X_cand[mask] = pert[mask] #Masking the X_candidates by setting the parts that are masked to the perturbed case

    # Sample on the candidate points using Constrained Max Posterior Sampling
    #Finding instance for new points using the model and the constrained model
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad(): #setting gradient finding to NO
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size) #Finding new_x using thompson sampling

    return X_next #returning new_x


#Main Optimization Loop

# Generate initial data reworked for real case
print("#################### GENERATING INITIAL TRAINING DATA ####################")
train_X = get_initial_points(dim, n_init) #initializing train_x
print(f"\nInitial Train_x is:\n{train_X}\n")
i=0
train_y_temp=[]
c1_temp=[]
c2_temp=[]
c3_temp=[]
for x in train_X:
    drag,thick_con,moment_con,lift_con=eval_objective(x,i)
    train_y_temp.append(drag)
    c1_temp.append(thick_con)
    c2_temp.append(moment_con)
    c3_temp.append(lift_con)
    i+=1
train_Y = torch.tensor(train_y_temp, **tkwargs).unsqueeze(-1) #getting initial train_y
print(f"\nInitial Train_y is:\n{train_Y}\n")
C1 = torch.tensor(c1_temp, **tkwargs).unsqueeze(-1) #Getting the first constraint

C2 = torch.tensor(c2_temp, **tkwargs).unsqueeze(-1) #Getting the second constraint

C3 = torch.tensor(c3_temp, **tkwargs).unsqueeze(-1) #Getting the third constraint


# Initialize TuRBO state
state = ScboState(dim, batch_size=batch_size)

N_CANDIDATES = 5000 if not SMOKE_TEST else 4
sobol = SobolEngine(dim, scramble=True, seed=1) #setting the sobol instance for random point generation

#Defining a method to get and fit the GP
def get_fitted_model(X, Y):
    print("Get_Fitted_Model Running:\n")
    print(f"The train_x tensor for fitting is \n{X}\n")
    print(f"The train_y tensor for fitting is \n{Y}\n")
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-1)) #setting Gaussian Likelihood

    #Setting the Kernel to the scaled MaternKernel
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )

    Y = Y.squeeze(-1) if Y.dim() == 3 else Y

    #Setting the GP model to SingleTaskGP   
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )

    #setting the MLL
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll, options={"maxiter": 5000, "disp": True})
    return model #returning the fitted model

maximum_iterations=0
lift_i=n_init
while not state.restart_triggered and maximum_iterations<200:  # Run until TuRBO converges (AKA when restart_triggered is True, the solution is converged)
    # Fit GP models for objective and constraints
    model = get_fitted_model(train_X, train_Y) #getting the model using the model fitting method
    c1_model = get_fitted_model(train_X, C1) #Getting the Constraint 1 GP model
    c2_model = get_fitted_model(train_X, C2) #Getting the constraint 2 GP model
    c3_model = get_fitted_model(train_X, C3) #Getting the constraint 2 GP model

    # Dimension Fixing
    if C1.dim() > C2.dim() or C1.dim() > C3.dim():
        C1 = C1.squeeze(-1) 
    elif C2.dim() > C1.dim() or C2.dim() > C3.dim():
        C2 = C2.squeeze(-1) 
    elif C3.dim() > C1.dim() or C3.dim() > C2.dim():
        C3 = C3.squeeze(-1) 
    
    # Generate a batch of candidates
    X_next = generate_batch(
        state=state,
        model=model,
        X=train_X,
        Y=train_Y,
        C=torch.cat((C1, C2, C3), dim=-1),
        batch_size=batch_size,
        n_candidates=N_CANDIDATES,
        constraint_model=ModelListGP(c1_model, c2_model, c3_model),
        sobol=sobol,
    )
    print(f"x_next is: {X_next}")
    
    # Evaluate both the objective and constraints for the selected candidates
    y_next_temp=[]
    c1_next_temp=[]
    c2_next_temp=[]
    c3_next_temp=[]
    for x in X_next:
        drag,thick_con,moment_con,lift_con=eval_objective(x,lift_i)
        y_next_temp.append(drag)
        c1_next_temp.append(thick_con)
        c2_next_temp.append(moment_con)
        c3_next_temp.append(lift_con)
        lift_i+=1
    Y_next = torch.tensor(y_next_temp, dtype=dtype, device=device).unsqueeze(-1) #getting initial train_y
    print(f"y_next is: {Y_next}")
    C1_next = torch.tensor(c1_next_temp, dtype=dtype, device=device).unsqueeze(-1) #Getting the first constraint
    C2_next = torch.tensor(c2_next_temp, dtype=dtype, device=device).unsqueeze(-1) #Getting the second constraint
    C3_next = torch.tensor(c3_next_temp, dtype=dtype, device=device).unsqueeze(-1) #Getting the third constraint
    
    if C1_next.dim() > C2_next.dim() or C1_next.dim() > C3_next.dim():
        C1_next = C1_next.squeeze(-1) 
    elif C2_next.dim() > C1_next.dim() or C2_next.dim() > C3_next.dim():
        C2_next = C2_next.squeeze(-1) 
    elif C3_next.dim() > C1_next.dim() or C3_next.dim() > C2_next.dim():
        C3_next = C3_next.squeeze(-1) 
    
    C_next = torch.cat([C1_next, C2_next, C3_next], dim=-1) #Putting C1 C2 and C3 together
    print(f"C_next is: {C_next}")

    # Update TuRBO state
    print("Running update_state:\n")
    state = update_state(state=state, Y_next=Y_next, C_next=C_next)
    print("Update_state ran successfully. \n")

    # Append data
    train_X = torch.cat((train_X, X_next), dim=0) #concatenating the new training data for train_x
    train_Y = torch.cat((train_Y, Y_next), dim=0)#concatenating the new training data for train_y
    C1 = torch.cat((C1, C1_next), dim=0) #concatenating the new training data for constraint 1
    C2 = torch.cat((C2, C2_next), dim=0) #concatenating the new training data for constraint 2
    C3 = torch.cat((C3, C3_next), dim=0) #concatenating the new training data for constraint 3

    maximum_iterations=maximum_iterations+1

    #Constraint Iteration Saving
    np.save("train_y_tensor.npy",train_Y)
    np.save("train_x_tensor.npy",train_X)
    np.save("Constraint_1.npy",C1)
    np.save("Constraint_2.npy",C2)
    np.save("Constraint_3.npy",C3)

    #Printing the current progress
    if (state.best_constraint_values <= 0).all():
        print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
    else:
        violation = state.best_constraint_values.clamp(min=0).sum()
        print(
            f"{len(train_X)}) No feasible point yet! Smallest total violation: "
            f"{violation:.2e}, TR length: {state.length:.2e}"
        )
        