#!/bin/bash
#PBS -q short
#PBS -l nodes=1:ppn=28
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -N sim_lj_3
#PBS -o /home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy/development/lammps/ethanol_water/automated/sim_lj_3/LOG_mix_ethanol_water_0
#PBS -l mem=3000mb

# Load standard enviroment

module purge
module load mpi/openmpi/3.1-gnu-9.2

# Specify job directory and input file

v_dir=/home/st/st_st/st_st163811/workspace/vle_using_solvation_free_energy/development/lammps/ethanol_water/automated/sim_lj_3
v_input=ethanol.input

cd $v_dir
echo Submitting LAMMPS file: $v_input
mpirun --bind-to core --map-by core -report-bindings lmp -i $v_input
