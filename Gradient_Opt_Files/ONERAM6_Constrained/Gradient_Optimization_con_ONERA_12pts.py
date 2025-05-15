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
  # x0 = np.array([ 9.1307e-05, -9.7425e-05, -1.8227e-05, -2.2507e-05,  3.6236e-05,
   #     -9.9871e-07,  2.1854e-04,  7.7798e-05, -9.3102e-05, -1.5099e-04,
    #    -4.1904e-05, -1.0784e-04])
   x0 = np.ones(12)*0.0001
   x0 = np.array(x0)
#x0 = np.array([-0.000095535, -0.000066228, -0.000041222, 0.0000037044]) #Peter2
#x0 = np.array([-8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05, -8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05]) #Peter3
#x0 = np.array([-3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05, -3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05]) #Peter4
#x0 = np.array([-7.4868e-07, 5.3644e-05, -8.2305e-05, -7.3594e-05]) #Peter5
#bounds = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)) # setting bounds for DV_VALUES
#bounds = Bounds(-0.001, 0.001, keep_feasible=True)
   bounds_norm = Bounds(0, 1, keep_feasible=True)
   bounds = np.array([[-0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025, -0.00025], [0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025]])
#x0 = np.array([0.0002, 0.0002, 0.0002, 0.0002])
   x0_normalized = (x0 - bounds[0])/(bounds[1] - bounds[0])

# Getting Initial Value of Objective Function

   DV_VALUES = np.array(x0)
   np.save('DV_VALUE', DV_VALUES)  # saving as .npy

   objective_function_command=['python', 'SU2_Wrapper_All_Gradients_EULER_3D.py', '-mcfg',  "ffd_oneram6_Euler_12pts.cfg", '-mesh', "mesh_ONERAM6_inv_FFD.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-10', '-gflag', "F", '-mno', '0.8395', '-aoa', '3.06']
   print("Running Objective Function:\n")
   result=subprocess.run(objective_function_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   print("Output:", flush=True)
   print(result.stdout)
   print("Errors:", flush=True)
   print(result.stderr)

   Cd_initial = np.load('Drag_Coefficient.npy')
   #Grad_Initial = np.load('DRAG_Gradients.npy')
   Cl_initial = np.load('Lift_Coefficient.npy')
   th1 = np.load('Station1_thickness.npy')
   th2 = np.load('Station2_thickness.npy')
   th3 = np.load('Station3_thickness.npy')
   th4 = np.load('Station4_thickness.npy')
   th5 = np.load('Station5_thickness.npy')

# Defining the objective function


   def objective(x):
       DV_VALUES_normalized = np.array(x)
       DV_VALUES = DV_VALUES_normalized*(bounds[1] - bounds[0]) + bounds[0]
       np.save('DV_VALUE', DV_VALUES)  # saving as .npy
       objective_function_command=['python', 'SU2_Wrapper_All_Gradients_EULER_3D.py', '-mcfg', "ffd_oneram6_Euler_12pts.cfg", '-mesh', "mesh_ONERAM6_inv_FFD.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-10', '-gflag', "F", '-mno', '0.8395', '-aoa', '3.06']
       result = subprocess.run(objective_function_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       #print("Output:", flush=True)
       #print(result.stdout)
       #print("Errors:", flush=True)
       #print(result.stderr)
       Cd = np.load('Drag_Coefficient.npy')
       print('The current drag coefficient is: ', Cd)
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

   def th1_con(x):
       th1 = np.load('Station1_thickness.npy')
       print('The current station 1 thickness is: ', th1)
       th1_tol = th1 - 0.078
       return th1_tol

   def th1_grad(x):
       th1_grad = np.load('Station1_th_grads.npy')
       return th1_grad

   def th2_con(x):
       th2 = np.load('Station2_thickness.npy')
       print('The current station 2 thickness is: ', th2)
       th2_tol = th2 - 0.072
       return th2_tol

   def th2_grad(x):
       th2_grad = np.load('Station2_th_grads.npy')
       return th2_grad

   def th3_con(x):
       th3 = np.load('Station3_thickness.npy')
       print('The current station 3 thickness is: ', th3)
       th3_tol = th3 - 0.066
       return th3_tol

   def th3_grad(x):
       th3_grad = np.load('Station3_th_grads.npy')
       return th3_grad

   def th4_con(x):
       th4 = np.load('Station4_thickness.npy')
       print('The current station 4 thickness is: ', th4)
       th4_tol = th4 - 0.061
       return th4_tol

   def th4_grad(x):
       th4_grad = np.load('Station4_th_grads.npy')
       return th4_grad

   def th5_con(x):
       th5 = np.load('Station5_thickness.npy')
       print('The current station 5 thickness is: ', th5)
       th5_tol = th5 - 0.055
       return th5_tol

   def th5_grad(x):
       th5_grad = np.load('Station5_th_grads.npy')
       return th5_grad

   def lift_con(x):
       Cl = np.load('Lift_Coefficient.npy')
       Cl_target = 0.292
       print('The current lift coefficient is: ', Cl)
       Cl_tol = Cl - Cl_target + 0.000001
       return Cl_tol

   def lift_grad(x):
       Cl_grad = np.load('LIFT_Gradients.npy')
       return Cl_grad


   ineq_con1 = {'type': 'ineq', 'fun': th1_con}#, 'jac': th1_grad}
   ineq_con2 = {'type': 'ineq', 'fun': th2_con}#, 'jac': th2_grad}
   ineq_con3 = {'type': 'ineq', 'fun': th3_con}#, 'jac': th3_grad}
   ineq_con4 = {'type': 'ineq', 'fun': th4_con}#, 'jac': th4_grad}
   ineq_con5 = {'type': 'ineq', 'fun': th5_con}#, 'jac': th5_grad}
   eq_con6 = {'type': 'ineq', 'fun': lift_con}#, 'jac': lift_grad}

# Defining callback function


   def callback(x):
       xx.append(x*(bounds[1] - bounds[0]) + bounds[0]) # iterate xk
       fx.append(np.load('Drag_Coefficient.npy')) # function value
       gd.append(np.load('DRAG_Gradients.npy')) # gradients of objective function
       c1x.append(np.load('Station1_thickness.npy'))
       c2x.append(np.load('Station2_thickness.npy'))
       c3x.append(np.load('Station3_thickness.npy'))
       c4x.append(np.load('Station4_thickness.npy'))
       c5x.append(np.load('Station5_thickness.npy'))
       c6x.append(np.load('Lift_Coefficient.npy'))
       print(f"x {x}, f(x) {fx[-1]}, df(x) {gd[-1]}, th1 {c1x[-1]}, th2 {c2x[-1]}, th3 {c3x[-1]}, th4 {c4x[-1]}, th5 {c5x[-1]}, Cl {c6x[-1]}", flush=True)

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
  # gradient_stopper = GradientNormStopper(tolerance=1e-3)


# Optimization loop


   xx = []
   fx = []
   gd = []
   c1x = []
   c2x = []
   c3x = []
   c4x = []
   c5x = []
   c6x = []

# Unconstrained Optimization Using SLSQP from SciPy Library

   res = minimize(objective, x0_normalized, method='COBYLA',
           constraints = [ineq_con1, ineq_con2, ineq_con3, ineq_con4, ineq_con5, eq_con6],
           options={'tol':1e-9, 'disp':True},
           bounds=bounds_norm)
#res = minimize(objective, x0_normalized, method='SLSQP', jac=gradient,
 #          constraints = [ineq_con1, ineq_con2, ineq_con3, ineq_con4, ineq_con5, eq_con6],
  #         options={'ftol':1e-9, 'disp':True},
   #        bounds=bounds_norm, callback=callback)
#res = minimize(objective, x0_normalized, method='trust-constr', jac=gradient,
 #          constraints = [ineq_con1, ineq_con2, ineq_con3, ineq_con4, ineq_con5, eq_con6],
  #         options={'verbose': 3},
   #        bounds=bounds_norm, callback=callback)
   

   print(res.x)
   print(np.load('Drag_Coefficient.npy'))
   #print(np.load('DRAG_Gradients.npy'))
   print(np.load('Station1_thickness.npy'))
   print(np.load('Station2_thickness.npy'))
   print(np.load('Station3_thickness.npy'))
   print(np.load('Station4_thickness.npy'))
   print(np.load('Station5_thickness.npy'))
   print(np.load('Lift_Coefficient.npy'))
   opt_dvs_normalized = np.asarray(res.x)
   opt_dvs = opt_dvs_normalized*(bounds[1] - bounds[0]) + bounds[0]
   opt_cd = np.load('Drag_Coefficient.npy')
   #opt_gd = np.load('DRAG_Gradients.npy')
# Extracting and plotting output data


  # xk_array = np.asarray(xx)
#   Cd_array = np.asarray(fx)
  # Gd_array = np.asarray(gd)
 #  niter = len(Cd_array)
  # iters = range(niter + 2)

  # xk_initial = np.array(x0)
  # xk_values = np.concatenate((xk_initial.reshape(1,-1), xk_array))
  # xk_values = np.concatenate((xk_values, opt_dvs.reshape(1, -1)))
  # Cd_values = np.insert(Cd_array, 0, np.asarray(Cd_initial), axis=None)
  # Cd_values = np.append(Cd_values, opt_cd)
  # Gd_values = np.concatenate((Grad_Initial.reshape(1,-1), Gd_array))
  # Gd_values = np.concatenate((Gd_values, opt_gd.reshape(1, -1)))

  # inf_norm_grad = []
  # for i in range(len(Cd_array) + 2):
   #    inf_norm_grad.append((np.linalg.norm(Gd_values[i], np.inf)))

   #np.save('inf_norm_grad', inf_norm_grad)
  # np.save('iters', iters)
  # np.save('Cd_values', Cd_values)
   #np.save('xk_values', xk_values)
   #np.save('Gd_values', Gd_values)

#Parser Block
parser = argparse.ArgumentParser()

parser.add_argument("-np","--num_procs",type=int,default=1,help='Number of processes')
args = parser.parse_args()
num_processes=args.num_procs

main(num_processes=num_processes)
