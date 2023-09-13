#!/bin/bash
#
#SBATCH --job-name test_diffusionscaling-1
#SBATCH -o test_diffusionscaling-1.slurm.out
#SBATCH -e test_diffusionscaling-1.slurm.err
#SBATCH -A standby
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 04:00:00

system="test_diffusionscaling"
prefix="test_diffusionscaling-1"
Temperature="188.0"
RelaxationTime="1000.0"
DiffusionTime="10000.0"
DiffusionCutoff="0.1"
ChangeThreshold="0.1"
Criteria_Slope="0.0"
Criteria_Cycles="1000.0"
Criteria_RxnSelectionCount="10000.0"
Window_Slope="100.0"
Window_Mean="nan"
Window_Pause="1000.0"
Window_RxnSelection="1.0"
Scaling_Adjuster="1.0"
Scaling_Minimum="1e0"

# Set environment variables
export PATH="/depot/bsavoie/apps/openmpi/3.0.1/bin:$PATH"
export LD_LIBRARY_PATH="/depot/bsavoie/apps/openmpi/3.0.1/lib:$LD_LIBRARY_PATH"

# Adjust modules
module --force purge
module load intel/17.0.1.132
export MKL_DEBUG_CPU_TYPE=5
export MKL_CBWR=AUTO

# Write out job information
echo "Running on host: $SLURM_NODELIST"
echo "Running on node(s): $SLURM_NNODES"
echo "Number of processors: $SLURM_NPROCS"
echo "Current working directory: $SLURM_SUBMIT_DIR"

# User supplied shell commands
cd $SLURM_SUBMIT_DIR

# Run script
echo "Start time: $(date)"

rm ${prefix}.concentration
rm ${prefix}.scale

# System prep
#python3 ~/bin/hybrid_mdmc/gen_initial_hybridmdmc.py ${system} 1 'A A2 B A2B' '100 100 100 100' -msf ${system}.msf -header ${system}.header
#mpirun -np 16 /depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in  ${prefix}.in.init > ${prefix}.lammps.out
#cp ${prefix}.in.init           ${prefix}_prep.in.init
#cp ${prefix}.in.data           ${prefix}_prep.in.data
#cp ${prefix}.end.data          ${prefix}_prep.end.data
#cp ${prefix}.lammps.out        ${prefix}_prep.lammps.out
#cp ${prefix}.lammps.log        ${prefix}_prep.lammps.log
#cp ${prefix}.thermo.avg        ${prefix}_prep.thermo.avg
#cp ${prefix}.relax.lammpstrj   ${prefix}_prep.relax.lammpstrj
#cp ${prefix}.density.lammpstrj ${prefix}_prep.density.lammpstrj
#cp ${prefix}.heat.lammpstrj    ${prefix}_prep.heat.lammpstrj
#cp ${prefix}.equil.lammpstrj   ${prefix}_prep.equil.lammpstrj

# Reactive loop
for i in `seq 0 5`;do

    # Run RMD script
    python3 ~/bin/hybrid_mdmc/Development/hybridmdmc.py ${prefix}.end.data -trj_file ${prefix}.diffusion.lammpstrj\
	-msf ${system}.msf -rxndf ${system}.rxndf -settings ${system}.in.settings -header ${system}.header\
	-diffusion_step ${i}\
	-prefix ${prefix} \
	-temp ${Temperature} \
	-relax ${RelaxationTime} \
	-diffusion ${DiffusionTime} \
	-atom_style 'molecular' \
	-num_voxels '2 2 2' \
	-change_threshold ${ChangeThreshold} \
	-diffusion_cutoff ${DiffusionCutoff} \
	-scalerates 'cumulative' \
	-scalingcriteria_concentration_slope ${Criteria_Slope} \
	-scalingcriteria_concentration_cycles ${Criteria_Cycles} \
	-scalingcriteria_rxnselection_count ${Criteria_RxnSelectionCount} \
	-windowsize_slope ${Window_Slope} \
	-windowsize_scalingpause ${Window_Pause} \
	-windowsize_rxnselection ${Window_RxnSelection} \
	-scalingfactor_adjuster ${Scaling_Adjuster} \
	-scalingfactor_minimum ${Scaling_Minimum} \
	--no-charged_atoms

    # Run MD
    mpirun -np 16 /depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in  ${prefix}.in.init > ${prefix}.lammps.out
    
    # Concatenate files
    python3 ~/bin/concatenate_files.py ${prefix}.in.data             ${prefix}.master.in.data             -bookmark "Step ${i}"
    python3 ~/bin/concatenate_files.py ${prefix}.end.data            ${prefix}.master.end.data            -bookmark "Step ${i}"
    python3 ~/bin/concatenate_files.py ${prefix}.thermo.avg          ${prefix}.master.thermo.avg          -bookmark "Step ${i}"
    python3 ~/bin/concatenate_files.py ${prefix}.relax.lammpstrj     ${prefix}.master.relax.lammpstrj     -bookmark "Step ${i}"
    python3 ~/bin/concatenate_files.py ${prefix}.diffusion.lammpstrj ${prefix}.master.diffusion.lammpstrj -bookmark "Step ${i}"

done

echo "End time: $(date)"
