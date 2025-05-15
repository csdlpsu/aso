# import modules

import os
import pandas as pd
import csv
import numpy as np
import re
import subprocess
import argparse
import warnings
import time
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

def main(num_processes):

# Initializing Design Variables (DV_VALUES)

#x0 = np.array([1.0305e-03, -8.8276e-04, -3.8772e-04,  9.3874e-04, -1.8829e-03,
 #         1.1994e-03, -4.1145e-04,  1.0175e-03,  2.7803e-04, -2.4489e-04,
  #        5.5472e-04,  9.8664e-05])
   #x0 = np.array([0.0448, 0.0612, 0.0120, 0.0101])
   x0 = np.array([1.0305e-02, -8.8276e-02, -3.8772e-02, 9.3874e-02, -1.8829e-02, 1.1994e-02, -4.1145e-02, 1.0175e-02])
#x0 = np.array([-0.000095535, -0.000066228, -0.000041222, 0.0000037044]) #Peter2
#x0 = np.array([-8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05, -8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05]) #Peter3
#x0 = np.array([-3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05, -3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05]) #Peter4
#x0 = np.array([-7.4868e-07, 5.3644e-05, -8.2305e-05, -7.3594e-05]) #Peter5
#bounds = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)) # setting bounds for DV_VALUES
#bounds = Bounds(-0.001, 0.001, keep_feasible=True)
   bounds_norm = Bounds(0, 1, keep_feasible=True)
   bounds = np.array([[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
#x0 = np.array([0.0002, 0.0002, 0.0002, 0.0002])
   x0_normalized = (x0 - bounds[0])/(bounds[1] - bounds[0])

# Getting Initial Value of Objective Function

   DV_VALUES = np.array(x0)
   np.save('DV_VALUE', DV_VALUES)  # saving as .npy

   objective_function_command=['python', 'SU2_Wrapper_All_Gradients_EULER_2D.py', '-mcfg',  "config_NACA0012_8pts.cfg", '-mesh', "mesh_NACA0012_8pts.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-10', '-gflag', "T", '-mno', '0.85', '-aoa', '0.0']
   print("Running Objective Function:\n")
   result=subprocess.run(objective_function_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   print("Output:", flush=True)
   print(result.stdout)
   print("Errors:", flush=True)
   print(result.stderr)

   Cd_initial = np.load('Drag_Coefficient.npy')
   Grad_Initial = np.load('DRAG_Gradients.npy')

# Defining the objective function


   def objective(x):
       DV_VALUES_normalized = np.array(x)
       DV_VALUES = DV_VALUES_normalized*(bounds[1] - bounds[0]) + bounds[0]
       np.save('DV_VALUE', DV_VALUES)  # saving as .npy
       objective_function_command=['python', 'SU2_Wrapper_All_Gradients_EULER_2D.py', '-mcfg', "config_NACA0012_8pts.cfg", '-mesh', "mesh_NACA0012_8pts.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-10', '-gflag', "T", '-mno', '0.85', '-aoa', '0.0']
       result = subprocess.run(objective_function_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       #print("Output:", flush=True)
       #print(result.stdout)
       #print("Errors:", flush=True)
       #print(result.stderr)
       Cd = np.load('Drag_Coefficient.npy')
       return Cd*10000

# Defining gradient of objective function


   def gradient(x):
       #file_path = 'of_grad.dat'
       #with open(file_path, 'r') as file:
        #   content = str(file.readlines())
       #grad_str = re.findall(r"[-+]?\d*\.\d+|\d+", content)
       #grad_Cd = [float(gradient) for gradient in grad_str]
       #grad_Cd = np.array(grad_Cd)
       grad_Cd = np.load('DRAG_Gradients.npy')
       return grad_Cd

   def moment_con(x):
       Cmz = np.load('Moment_Coefficient.npy')
       Max_Cmz = 0.093
       Cmz_tol = -Cmz + 0.093
       return Cmz_tol

   def moment_grad(x):
       Cmz_grad = np.load('MOMENT_Z_Gradients.npy')
       return Cmz_grad

   def thickness_con(x):
       t = np.load('Thickness.npy')
       t_min = 0.12
       t_tol = t - t_min
       return t_tol

   def thickness_grad(x):
       t_grad = np.load('Thickness_Gradients.npy')
       return t_grad

   def lift_con(x):
       Cl = np.load('Lift_Coefficient.npy')
       Cl_target = 0.80
       Cl_tol = Cl - Cl_target
       return Cl_tol

   def lift_grad(x):
       Cl_grad = np.load('LIFT_Gradients.npy')
       return Cl_grad


   cons = ({'type': 'ineq', 'fun': moment_con, 'jac': moment_grad},
          {'type': 'ineq', 'fun': thickness_con, 'jac': thickness_grad},
          {'type': 'eq', 'fun': lift_con, 'jac': lift_grad})

# Defining callback function


#def callback(x, tc):
 #   xx.append(x*(bounds[1] - bounds[0]) + bounds[0]) # iterate xk
  #  fx.append(np.load('Drag_Coefficient.npy')) # function value
   # gd.append(np.load('DRAG_Gradients.npy')) # gradients of objective function
   # print(f"x {x}, f(x) {fx[-1]}, df(x) {gd[-1]}", flush=True)

   class GradientNormStopper:
       def __init__(self, tolerance=1e-3):
           self.tolerance = tolerance
           self.stopped = False
           self.iteration = 0

       def __call__(self, xk):
           gradient_obj = gradient(xk)
           grad_norm_inf = np.linalg.norm(gradient_obj, ord=np.inf)
           self.iteration += 1
           print(f"Iteration {self.iteration}: Infinity norm of gradient = {grad_norm_inf}")
           xx.append(xk*(bounds[1] - bounds[0]) + bounds[0]) # iterate xk
           fx.append(np.load('Drag_Coefficient.npy')) # function value 
           gd.append(np.load('DRAG_Gradients.npy')) # gradients of objective function
           print(f"x {xk}, f(x) {fx[-1]}, df(x) {gd[-1]}", flush=True)
           if grad_norm_inf < self.tolerance:
               self.stopped = True
               print(
                   f"Stopping criterion met: Infinity norm of gradient ({grad_norm_inf}) is below tolerance ({self.tolerance}).")
               return True  # This will signal the optimizer to stop
           return False


# Set up custom stopping criterion
   gradient_stopper = GradientNormStopper(tolerance=1e-3)


# Optimization loop


   xx = []
   fx = []
   gd = []

# Unconstrained Optimization Using SLSQP from SciPy Library

res = minimize(objective, x0_normalized, method='L-BFGS-B', jac=gradient,
           options={'gtol':1e-3, 'maxiter':50},
           bounds=bounds_norm, callback=gradient_stopper)
#res = minimize(objective, x0_normalized, method='Nelder-Mead',
 #       options={'fatol':1e-4, 'return_all':True, 'adaptive':True, 'disp':True },
  #      bounds=bounds_norm, callback=callback)
#res = minimize(objective, x0_normalized, method='SLSQP', jac=gradient,
 #          options={'ftol':1e-9, 'disp':True},
  #         bounds=bounds_norm, callback=gradient_stopper)
#res = minimize(objective, x0_normalized, method='trust-constr',  jac=gradient,
 #          options={'gtol': 1e-4, 'disp': True, 'verbose': 1},
  #         bounds=bounds_norm, callback=callback)
#res = minimize(objective, x0_normalized, method='TNC', jac=gradient,
 #          options={'ftol':1e-5, 'gtol':1e-3, 'disp':True},
  #         bounds=bounds_norm, callback=gradient_stopper)
   

   print(res.x)
   print(np.load('Drag_Coefficient.npy'))
   print(np.load('DRAG_Gradients.npy'))
   opt_dvs_normalized = np.asarray(res.x)
   opt_dvs = opt_dvs_normalized*(bounds[1] - bounds[0]) + bounds[0]
   opt_cd = np.load('Drag_Coefficient.npy')
   opt_gd = np.load('DRAG_Gradients.npy')
# Extracting and plotting output data


   xk_array = np.asarray(xx)
   Cd_array = np.asarray(fx)
   Gd_array = np.asarray(gd)
   niter = len(Cd_array)
   iters = range(niter + 2)

   xk_initial = np.array(x0)
   xk_values = np.concatenate((xk_initial.reshape(1,-1), xk_array))
   xk_values = np.concatenate((xk_values, opt_dvs.reshape(1, -1)))
   Cd_values = np.insert(Cd_array, 0, np.asarray(Cd_initial), axis=None)
   Cd_values = np.append(Cd_values, opt_cd)
   Gd_values = np.concatenate((Grad_Initial.reshape(1,-1), Gd_array))
   Gd_values = np.concatenate((Gd_values, opt_gd.reshape(1, -1)))

   inf_norm_grad = []
   for i in range(len(Cd_array) + 2):
       inf_norm_grad.append((np.linalg.norm(Gd_values[i], np.inf)))

   np.save('inf_norm_grad', inf_norm_grad)
   np.save('iters', iters)
   np.save('Cd_values', Cd_values)
   np.save('xk_values', xk_values)
   np.save('Gd_values', Gd_values)

#Parser Block
parser = argparse.ArgumentParser()

parser.add_argument("-np","--num_procs",type=int,default=1,help='Number of processes')
args = parser.parse_args()
num_processes=args.num_procs

main(num_processes=num_processes)
