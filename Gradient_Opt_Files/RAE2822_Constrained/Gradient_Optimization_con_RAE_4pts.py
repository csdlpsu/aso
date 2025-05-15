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

   x0 = np.array([1.0305e-03, -8.8276e-04, -3.8772e-04,  9.3874e-04]) 
   #x0 = np.array([-0.000095535, -0.000066228, -0.000041222, 0.0000037044]) #Peter2
   #x0 = np.array([-8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05, -8.8744e-05, 2.6461e-05, -3.0221e-05, -1.9657e-05]) #Peter3
   #x0 = np.array([-3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05, -3.8515e-05, 2.6816e-05, -1.9813e-06, 7.9289e-05]) #Peter4
   #x0 = np.array([-7.4868e-07, 5.3644e-05, -8.2305e-05, -7.3594e-05]) #Peter5
   #bounds = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)) # setting bounds for DV_VALUES
   #bounds = Bounds(-0.001, 0.001, keep_feasible=True)
   bounds_norm = Bounds(0, 1, keep_feasible=True)
   bounds = np.array([[-0.002, -0.002, -0.002, -0.002], [0.002, 0.002, 0.002, 0.002]])
   #x0 = np.array([0.0002, 0.0002, 0.0002, 0.0002])
   x0_normalized = (x0 - bounds[0])/(bounds[1] - bounds[0])

   # Getting Initial Value of Objective Function

   DV_VALUES = np.array(x0)
   np.save('DV_VALUE', DV_VALUES)  # saving as .npy

   objective_function_command=['python', 'SU2_Wrapper_All_Gradients_RANS.py', '-mcfg',  "ffd_rae2822_4pts.cfg", '-mesh', "4cpts_RAE2822_Mesh.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-8', '-gflag', "F", '-mno', '0.729', '-reno', '6500000', '-aoa', '2.31']
   print("Running Objective Function:\n")
   result=subprocess.run(objective_function_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   print("Output:", flush=True)
   print(result.stdout)
   print("Errors:", flush=True)
   print(result.stderr)

   Cd_initial = np.load('Drag_Coefficient.npy')
   Cm_initial = np.load('Moment_Coefficient.npy')
   Cl_initial = np.load('Lift_Coefficient.npy')
   th_initial = np.load('Thickness.npy')
  # Grad_Initial = np.load('DRAG_Gradients.npy')

   # Defining the objective function


   def objective(x):
       DV_VALUES_normalized = np.array(x)
       DV_VALUES = DV_VALUES_normalized*(bounds[1] - bounds[0]) + bounds[0]
       np.save('DV_VALUE', DV_VALUES)  # saving as .npy
       objective_function_command=['python', 'SU2_Wrapper_All_Gradients_RANS.py', '-mcfg', "ffd_rae2822_4pts.cfg", '-mesh', "4cpts_RAE2822_Mesh.su2", '-dv', "DV_VALUE.npy", '-np', str(num_processes), '-si', '0', '-cmin', '-8', '-gflag', "F", '-mno', '0.729', '-reno', '6500000', '-aoa', '2.31']
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

   def moment_con(x):
       Cmz = np.load('Moment_Coefficient.npy')
       print('The current moment coefficient is: ', Cmz)
       Max_Cmz = 0.092
       Cmz_tol = -Cmz + 0.092
       return Cmz_tol

   def moment_grad(x):
       Cmz_grad = np.load('MOMENT_Z_Gradients.npy')
       return Cmz_grad

   def thickness_con(x):
       t = np.load('Thickness.npy')
       print('Current max. airfoil thickness is: ', t)
       t_min = 0.12
       t_tol = t - t_min
       return t_tol

   def thickness_grad(x):
       t_grad = np.load('Thickness_Gradients.npy')
       return t_grad

   def lift_con(x):
       Cl = np.load('Lift_Coefficient.npy')
       print('Current lift coefficient is: ', Cl)
       Cl_target = 0.75
       Cl_tol = Cl - Cl_target + 0.000001
       return Cl_tol

   def lift_grad(x):
       Cl_grad = np.load('LIFT_Gradients.npy')
       return Cl_grad


   ineq_con1 = {'type': 'ineq', 'fun': moment_con}
   ineq_con2 = {'type': 'ineq', 'fun': thickness_con}
   eq_con3 = {'type': 'ineq', 'fun': lift_con}

   # Defining callback function


   def callback(x):
       xx.append(x*(bounds[1] - bounds[0]) + bounds[0]) # iterate xk
       fx.append(np.load('Drag_Coefficient.npy')) # function value
       #gd.append(np.load('DRAG_Gradients.npy')) # gradients of objective function
       c1x.append(np.load('Moment_Coefficient.npy'))
       c2x.append(np.load('Thickness.npy'))
       c3x.append(np.load('Lift_Coefficient.npy'))
       print(f"x {x}, f(x) {fx[-1]}, Cm {c1x[-1]}, th {c2x[-1]}, Cl {c3x[-1]}", flush=True)

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
   #gd = []
   c1x = []
   c2x = []
   c3x = []

   # Unconstrained Optimization Using SLSQP from SciPy Library

   res = minimize(objective, x0_normalized, method='COBYLA',
           constraints = [ineq_con1, ineq_con2, eq_con3],
           options={'tol':1e-9, 'disp':True},
           bounds=bounds_norm, callback=callback)
#res = minimize(objective, x0_normalized, method='SLSQP', jac=gradient,
 #          constraints = [ineq_con1, ineq_con2, eq_con3],
  #         options={'ftol':1e-9, 'disp':True},
   #        bounds=bounds_norm, callback=callback)
#res = minimize(objective, x0_normalized, method='trust-constr', jac=gradient,
 #          constraints = [ineq_con1, ineq_con2, eq_con3],
  #         options={'verbose': 3},
   #        bounds=bounds_norm, callback=callback)

   print(res.x)
   print(np.load('Drag_Coefficient.npy'))
  # print(np.load('DRAG_Gradients.npy'))
   print(np.load('Lift_Coefficient.npy'))
   print(np.load('Thickness.npy'))
   print(np.load('Moment_Coefficient.npy'))
   opt_dvs_normalized = np.asarray(res.x)
   opt_dvs = opt_dvs_normalized*(bounds[1] - bounds[0]) + bounds[0]
   opt_cd = np.load('Drag_Coefficient.npy')
   opt_cm = np.load('Moment_Coefficient.npy')
   opt_th = np.load('Thickness.npy')
   opt_cl = np.load('Lift_Coefficient.npy')
   #opt_gd = np.load('DRAG_Gradients.npy')
   # Extracting and plotting output data


   xk_array = np.asarray(xx)
   Cd_array = np.asarray(fx)
   Cm_array = np.asarray(c1x)
   th_array = np.asarray(c2x)
   Cl_array = np.asarray(c3x)
   #Gd_array = np.asarray(gd)
   niter = len(Cd_array)
   iters = range(niter + 2)

   xk_initial = np.array(x0)
   #xk_values = np.concatenate((xk_initial.reshape(1,-1), xk_array))
   #xk_values = np.concatenate((xk_values, opt_dvs.reshape(1, -1)))
   Cd_values = np.insert(Cd_array, 0, np.asarray(Cd_initial), axis=None)
   Cd_values = np.append(Cd_values, opt_cd)
   Cm_values = np.insert(Cm_array, 0, np.asarray(Cm_initial), axis=None)
   Cm_values = np.append(Cm_values, opt_cm)
   th_values = np.insert(th_array, 0, np.asarray(th_initial), axis=None)
   th_values = np.append(th_values, opt_th)
   Cl_values = np.insert(Cl_array, 0, np.asarray(Cl_initial), axis=None)
   Cl_values = np.append(Cl_values, opt_cl)

   #Gd_values = np.concatenate((Grad_Initial.reshape(1,-1), Gd_array))
   #Gd_values = np.concatenate((Gd_values, opt_gd.reshape(1, -1)))

   #inf_norm_grad = []
   #for i in range(len(Cd_array) + 2):
    #   inf_norm_grad.append((np.linalg.norm(Gd_values[i], np.inf)))

  # np.save('inf_norm_grad', inf_norm_grad)
   np.save('iters', iters)
   np.save('Cd_values', Cd_values)
   np.save('Cl_values', Cl_values)
   np.save('Cm_values', Cm_values)
   np.save('th_values', th_values)
   #np.save('xk_values', xk_values)
   #np.save('Gd_values', Gd_values)

# Parser Block
parser = argparse.ArgumentParser()

parser.add_argument("-np","--num_procs",type=int,default=1,help='Number of processes')
args = parser.parse_args()
num_processes=args.num_procs

main(num_processes=num_processes)
