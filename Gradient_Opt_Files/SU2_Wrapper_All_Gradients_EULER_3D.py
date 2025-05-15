r"""
A script that wraps around SU2 to run primal and adjoint solvers, taking in as inputs operating conditions
(Mach and AoA) for EULER solver as well as FFD control points.

Input:
- mcfg: [Path to] SU2 config file
- mesh: [Path to] SU2 mesh file
- si  : Experiment # to start with (redundant at the time of writing)
- np  : NUmber of processes for parallel SU2 run
- hf  : [Path to] history fule containing SU2 convergence history
- dv  : NumPy file containing the FFD control point displacements
- cmin: Minimum value of log10 of residual. -12 is full convergence; higher values partial convergence
- gflag: True/False for gradient computation
- mno : Mach number
- aoa : Angle of attack

Output:
- Drag_Coefficient.npy # Cd values
- Lift_Coefficient.npy # Cl values
- Moment_Coefficient.npy # CMz values (about 1/4 chord)

To run:
python SU2_Wrapper_All_Gradients_EULER_3D.py -mcfg "CRM_WBT_Halfbody_Tet_Euler_20pts.cfg" -mesh "Volume_Mesh_Tet_Meters_20million_Flight_Scale_Ysymmetry_Euler_FFD.su2" -dv "DV_VALUE.npy" -np 288 -si 0 -cmin -8 -gflag "F" -mno 0.85 -aoa 2.37
"""


import numpy as np
import pandas as pd
import subprocess
import argparse
import warnings
import time
import os
import re

os.environ['SU2_RUN'] = '/storage/icds/RISE/sw8/su2-7.5.1/bin/'

warnings.filterwarnings('ignore')

#Wrapper Class
class ExperimentDesignWrapper():
    #Initializer
    def __init__(self):
        self.input_file=None
        self.config_file=None
        return
    
    #Method to write the design values into the cfg files
    def write_config_file_DVs(self,config_file,dv_values, num_processes, minval, aoa, mach_num):

        #Opening and storing the config file data passed
        with open(config_file,'r') as infile:
            config_file_data=infile.readlines()

        #opening and rewriting the config file
        with open(config_file,'w') as outfile: 
            for line in config_file_data:
                if line.startswith("DV_VALUE"):
                    Label, old_value=line.split('=')
                    joined_dv=','.join(str(coefficient) for coefficient in dv_values)
                    newline=f"{Label}={joined_dv}\n"
                    outfile.write(newline)
                elif line.startswith("MESH_FILENAME"): #Writing the working mesh infile
                    Label, old_value=line.split('=')
                    newline=f"{Label}=working_mesh.su2\n"
                    outfile.write(newline)
                elif line.startswith("MESH_OUT_FILENAME"): #Writing the working mesh outfile
                    Label, old_value=line.split('=')
                    newline=f"{Label}=working_mesh.su2\n"
                    outfile.write(newline)
                elif line.startswith("CONV_RESIDUAL_MINVAL"): 
                    Label, old_value=line.split('=')
                    newline=f"{Label}= {minval}\n"
                    outfile.write(newline)
                elif line.startswith("MACH_NUMBER"):
                   Label, old_value=line.split('=')
                   newline=f"{Label}= {mach_num}\n"
                   outfile.write(newline)
                elif line.startswith("AOA"): 
                    Label, old_value=line.split('=')
                    newline=f"{Label}= {aoa}\n"
                    outfile.write(newline)   
                else:
                    outfile.write(line)
        
        return
    

    #Method to run SU2_CFD simulation
    def run_SU2_CFD(self,config_file,num_processes):
        result=subprocess.run(['srun', '/storage/icds/RISE/sw8/su2/intel-2021/v8.0.1/bin/SU2_CFD',config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", flush=True)
        print(result.stdout)
        print("Errors:", flush=True)
        print(result.stderr)
    
    #Method to get the gradients using SU2_DOT
    def run_SU2_DOT(self,config_file, num_processes):
        #Looping through all gradients
        gradients=["DRAG", "LIFT"]#, "MOMENT_Z"]
        tags=["cd"]#, "cl", "cmz"]#Tags for each gradient
        for tag in tags: #Copying the restart solution so that it will work with all 3 objective functions
            subprocess.run(["cp", "solution_adj.csv", "solution_adj_"+tag+".csv"])
                
        for objective in gradients:
            print("OBJECTIVE_FUNCTION=",objective)
            #Opening and storing the config file data passed
            with open(config_file,'r') as infile:
                config_file_data=infile.readlines()

            #opening and rewriting the config file
            with open(config_file,'w') as outfile: 
                for line in config_file_data:
                    if line.startswith("OBJECTIVE_FUNCTION"):
                        Label, old_value=line.split('=')
                        newline=f"{Label}={objective}\n"
                        outfile.write(newline)
                    elif line.startswith("GRAD_OBJFUNC_FILENAME"):
                        newline=f"GRAD_OBJFUNC_FILENAME= of_grad.dat\n"
                        outfile.write(newline)
                    else:
                        outfile.write(line)
            #Running continuous_adjoint solution with 
            print("RUNNING CONTINUOUS_ADJOINT")
            result=subprocess.run(['srun', '/storage/icds/RISE/sw8/su2/intel-2021/v8.0.1/bin/SU2_CFD',config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Output:")
            print(result.stdout)
            print("Errors:")
            print(result.stderr)
            
            
            #Running SU2_DOT
            result=subprocess.run(['srun', '/storage/icds/RISE/sw8/su2/intel-2021/v8.0.1/bin/SU2_DOT',config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)# Print the output and errors
            print("Output:", flush=True)
            print(result.stdout)
            print("Errors:", flush=True)
            print(result.stderr)
            dat_file="of_grad.dat"
            number_pattern = r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+(?:[eE][+-]?\d+)?|-?\d+'
            extracted_numbers= []
            with open(dat_file, 'r') as file:
                for line in file:
                    numbers = re.findall(number_pattern, line)
                    extracted_numbers.extend([float(num) for num in numbers])
                    grad_Cd = np.array(extracted_numbers)
                    np.save(objective+'_Gradients', grad_Cd)

    #Method to get the CD CL and CF from the history CSV file
    def get_output_data(self,history_output, num_processes):
        history_output_dataframe=pd.read_csv(history_output) 
        cd = history_output_dataframe['       "CD"       '].iloc[-1]
        cl = history_output_dataframe['       "CL"       '].iloc[-1]
        cmz = history_output_dataframe['       "CMz"      '].iloc[-1]
        
        return cd,cl,cmz
    
    #Method to set up continuous adjoint by running the direct and restart solutions
    def continuous_adjoint_setup(self,config_file,mathproblem,restartsol, num_processes):
        #Opening and storing the config file data passed
        with open(config_file,'r') as infile:
            config_file_data=infile.readlines()

        #opening and rewriting the config file
        with open(config_file,'w') as outfile: 
            for line in config_file_data:
                if line.startswith("MATH_PROBLEM"):
                    Label, old_value=line.split('=')
                    newline=f"{Label}={mathproblem}\n"
                    outfile.write(newline)
                    
                elif line.startswith("RESTART_SOL"):
                    Label, old_value=line.split('=')
                    newline=f"{Label}={restartsol}\n"
                    outfile.write(newline)
                
                elif line.startswith("SOLUTION_FILENAME") or line.startswith("SOLUTION_ADJ_FILENAME"): #Setting up file saving for continuous_adjoint
                    Label, old_value=line.split('=')
                    newline=f"{Label}=solution_adj.csv\n"
                    outfile.write(newline)

                elif line.startswith("RESTART_FILENAME") or line.startswith("RESTART_ADJ_FILENAME"): #setting up file saving for continuous_adjoint
                    Label, old_value=line.split('=')
                    newline=f"{Label}=solution_adj.csv\n"
                    outfile.write(newline)

                else:
                    outfile.write(line)

     #Method to run SU2_DEF for mesh deformation
    def run_SU2_DEF(self, config_file, num_processes):
        subprocess.run(['srun', '/storage/icds/RISE/sw8/su2/intel-2021/v8.0.1/bin/SU2_DEF', config_file])

    #Method to compute geometry details and gradient of thickness
    def run_SU2_GEO(self, config_file, geo_mode, num_processes):
        #opening and rewriting config file
        with open(config_file, 'r') as infile:
            config_file_data = infile.readlines()

        with open(config_file, 'w') as outfile:
            for line in config_file_data:
                if line.startswith("GEO_MODE"):
                    Label, old_value=line.split('=')
                    newline=f"{Label}={geo_mode}\n"
                    outfile.write(newline)
                else:
                    outfile.write(line)

        #Running SU2_GEO module
        print("Running SU2_GEO module", flush=True)
        result=subprocess.run(['srun', '/storage/icds/RISE/sw8/su2/intel-2021/v8.0.1/bin/SU2_GEO', config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output: ", flush=True)
        print(result.stdout)
        print("Errors: ", flush=True)
        print(result.stderr)

#Objective Function Block

def objective_function(dv_value, working_cfg, num_processes, history_file, minval,
                       grad_flag, aoa, mach_num):
    fulltimestart=time.time()
    EDW=ExperimentDesignWrapper()
    
    #Setting Direct solution for CFD simulation
    mathproblem="DIRECT"
    restartsol="NO"
    EDW.continuous_adjoint_setup(working_cfg,mathproblem=mathproblem,restartsol=restartsol,
                                 num_processes=num_processes)

    print(dv_value)
    print(minval)
    EDW.write_config_file_DVs(working_cfg,dv_value, num_processes, minval, aoa, mach_num) # Writing the DV_VALUE to main

    print("\n Running SU2_DEF in Obj_func")

    starttime= time.time()

    EDW.run_SU2_DEF(working_cfg, num_processes)

    endtime= time.time()

    timediff = endtime - starttime

    print(f"SU2_DEF took {timediff} seconds \n")

    print("\n Running Direct SU2_CFD in Obj_func", flush=True)
    starttime=time.time()

    EDW.run_SU2_CFD(working_cfg,num_processes)

    endtime=time.time()
    timediff=endtime-starttime
    print(f"SU2_CFD took {timediff} seconds \n", flush=True)
    
    #Getting the CD,CL,CMZ from the Direct solution and storing it
    Cd,Cl,Cmz=EDW.get_output_data(history_file, num_processes)
    
    # Saving Aero Coefficients
    np.save("Drag_Coefficient.npy", Cd)
    np.save("Lift_Coefficient.npy", Cl)
    np.save("Moment_Coefficient.npy", Cmz)


    #File Setup

    subprocess.run(["cp", "solution_adj.csv", "solution_adj_cd.csv"]) #Copying so that it can run the first time

    if grad_flag == "T":

       #Running the continuous_adjoint with restart for gradient sensitivities
       print("Running Continuous_adjoint")
       restartsol="NO"
       mathproblem="CONTINUOUS_ADJOINT"
       EDW.continuous_adjoint_setup(working_cfg,mathproblem=mathproblem,restartsol=restartsol, num_processes=num_processes)

    
       EDW.run_SU2_DOT(working_cfg, num_processes) #Saving the gradients
       print("Finished Saving Gradients")
    
    #Running SU2_GEO for obtaining thickness gradients
    print("\n Running SU2_GEO for function evaluation")
    geo_mode="FUNCTION"
    EDW.run_SU2_GEO(working_cfg, geo_mode=geo_mode, num_processes=num_processes)

    #Saving thickness of airfoil
    num_stations = 5
    thickness_arr = []
    for i in range(num_stations):
        cols = "STATION"+str(i+1)+"_THICKNESS"
        th_df=pd.read_csv('of_func.csv', header=0, usecols=[cols])
        th_data=th_df.head()
        th_num=th_data.to_numpy()
        th_reshape=th_num.reshape(1, -1)
        thickness=th_reshape[0]
        thickness_arr.append(thickness)
        np.save("Thickness", thickness_arr)
    
    thi = np.load('Thickness.npy')
    th1 = thi[0].squeeze()
    np.save('Station1_thickness.npy', th1)
    
    th2 = thi[1].squeeze()
    np.save('Station2_thickness.npy', th2)

    th3 = thi[2].squeeze()
    np.save('Station3_thickness.npy', th3)

    th4 = thi[3].squeeze()
    np.save('Station4_thickness.npy', th4)

    th5 = thi[4].squeeze()
    np.save('Station5_thickness.npy', th5)

    if grad_flag == "T":

       print("\n Running SU2_GEO for gradients")
       geo_mode="GRADIENT"
       EDW.run_SU2_GEO(working_cfg, geo_mode=geo_mode, num_processes=num_processes)
       print("\n Finished running SU2_GEO")

       #Saving thickness gradients
       thickness_grad_arr = []
       for i in range(num_stations):
           cols = "STATION"+str(i+1)+"_THICKNESS"
           th_grad_df=pd.read_csv('of_grad.csv', header=0, usecols=[cols])
           th_grad_data=th_grad_df.head()
           th_grad_num=th_grad_data.to_numpy()
           th_grad_reshape=th_grad_num.reshape(1, -1)
           thickness_grads=th_grad_reshape[0]
           thickness_grad_arr.append(thickness_grads)
           np.save("Thickness_Gradients", thickness_grad_arr)

       th_grads = np.load('Thickness_Gradients.npy')
       th1_grad = np.asarray(th_grads[0])
       np.save('Station1_th_grads.npy', th1_grad)

       th2_grad = np.asarray(th_grads[1])
       np.save('Station2_th_grads.npy', th2_grad)

       th3_grad = np.asarray(th_grads[2])
       np.save('Station3_th_grads.npy', th3_grad)

       th4_grad = np.asarray(th_grads[3])
       np.save('Station4_th_grads.npy', th4_grad)

       th5_grad = np.asarray(th_grads[4])
       np.save('Station5_th_grads.npy', th5_grad)

    #print("\n Running SU2_DEF in Obj_func")
    #starttime= time.time()

    #EDW.run_SU2_DEF(working_cfg, num_processes)
    #endtime= time.time()
    #timediff = endtime - starttime

    #print(f"SU2_DEF took {timediff} seconds \n")

    fulltimeend=time.time()
    print(f"The objective function took {fulltimeend-fulltimestart} seconds to run\n")
    
    return 
        


#Parser Block

#Setting parser arguments
parser = argparse.ArgumentParser()

parser.add_argument("-mcfg","--main_cfg",      type=str,   default='config_NACA0012.cfg', help='Path to the main cfg file')
parser.add_argument("-mesh","--mesh_su2",      type=str,   default="mesh_NACA0012.su2",   help="Path to the SU2 mesh file")
parser.add_argument("-si",  "--start_iters",   type=int,   default=0,                     help='Number of iterations to start at')
parser.add_argument("-np",  "--num_procs",     type=int,   default=1,             help='Number of processes')
parser.add_argument("-hf",  "--history_file",  type=str,   default='history.csv', help="History file containing aerodynamic coefficient and convergence history")
parser.add_argument("-dv",  "--DV_VALUE",      type=str,   default="DV_VALUE.npy",help="Numpy file containing the DV_VALUE information")
parser.add_argument("-cmin","--CONV_MIN",      type=float, default=-10,  help="Minimum value of log10 of residual")
parser.add_argument("-gflag", "--GRAD_FLAG",   type=str,   default="T",  help="True/False statement for gradient computation")
parser.add_argument("-mno",   "--MACH_NO",     type=float, default=0.83, help="Mach Number")
parser.add_argument("-aoa",   "--AOA",         type=float, default=3.06,    help="Angle of Attack")


args = parser.parse_args()

#Saving Inputs from parser to variables

main_cfg=args.main_cfg
mesh_name=args.mesh_su2
start_iters=args.start_iters
num_processes=args.num_procs
history_file=args.history_file
dvs=args.DV_VALUE
dv_value=np.load(dvs)
dv_value=dv_value.flatten().tolist()
minval=args.CONV_MIN
grad_flag=args.GRAD_FLAG
mach_num=args.MACH_NO
aoa=args.AOA
working_cfg="working_config.cfg"
#Copying the Files used to ensure they are correctly written at the start:

subprocess.run(["cp", main_cfg, working_cfg])
subprocess.run(["cp", mesh_name, "working_mesh.su2"])
        

#Control Block
print("START\n")

#Calling the function
objective_function(dv_value=dv_value, working_cfg=working_cfg, num_processes=num_processes,
                   history_file=history_file, minval=minval, grad_flag=grad_flag, aoa=aoa,
                   mach_num=mach_num)
