#!/bin/bash
#SBATCH -J ONERAM6

#SBATCH --output=SU2_log.%j.out

#SBATCH -N 3

#SBATCH --ntasks=144

#SBATCH -t 96:00:00

#SBATCH --mem=256GB

#SBATCH -A akr6198

#SBATCH -p sla-prio

#SBATCH --mail-type=ALL

#SBATCH --mail-user=pqn5121@psu.edu



echo "Job starting on `hostname` at `date`"



echo -e "Slurm job ID: $SLURM_JOBID"



cd $SLURM_SUBMIT_DIR



# A little useful information for the log file...

echo -e "Master process running on: $HOSTNAME"

echo -e "Directory is: $PWD"



# Load the su2 module. The su2-intel module also loads all of the Intel suite

module purge

#module load cuda/11.5.0

#module load anaconda/2021.11
#module load anaconda3/2021.05

#module load cmake/3.21.4
#module load cmake/3.25.2-intel-2021.4

#module load intel/2021.4.0

#module load impi/2021.4.0

#module load tbb/2021.4.0

#module load mkl/2021.4.0

module load su2/8.0.1-intel

#module load anaconda3/2021.05

#module load cmake/3.25.2-intel-2021.4

#source activate base

echo " "

echo "The following modules are in use"

module list

echo " "



# Call srun to launch the MPI-based job


#srun $SU2_RUN/SU2_CFD CRM_WBT_Halfbody_Voxel_20pts.cfg
#srun $SU2_RUN/SU2_DEF CRM_WBT_Halfbody_Tet_Euler_20pts.cfg
#shape_optimization.py -n 48 -g CONTINUOUS_ADJOINT -o SLSQP -f  inv_NACA0012_basic_modified.cfg
#set_ffd_design_var.py -i 10 -j 1 -k 0 -b MAIN_BOX -m 'airfoil' --dimension 2
#srun $SU2_RUN/SU2_DOT ffd_rae2822_4pts.cfg
#python SU2_Wrapper_CD_Gradient_Only.py -mcfg config_NACA0012.cfg -mesh mesh_NACA0012.su2 -dv DV_VALUE.npy -np 240 -si 0
#python Gradient_Optimization_SLSQP.py
#python SU2_Wrapper_CD_Gradient_Only_2.py -mcfg "config_NACA0012.cfg" -mesh "mesh_NACA0012.su2" -dv "DV_VALUE.npy" -np 48
#/usr/bin/time -v parallel_computation.py -n 4 -f inv_ONERAM6_adv.cfg
#python SU2_Wrapper_All_Gradients_EULER_2D.py -mcfg "config_NACA0012.cfg" -mesh "mesh_NACA0012.su2" -dv "DV_VALUE.npy" -np 144 -si 0 -cmin -10 -gflag "T" -mno 0.85 -aoa 0.0 
python Gradient_Optimization_BFGS_4pts.py -np 144
#python dvs.py



echo " "

echo "srun exited with return code $local_rc"

echo " "



# Job complete



echo "Job ending on `hostname` at `date`"
