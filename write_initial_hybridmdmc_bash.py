#!/usr/bin/env python3
# Author
#   Dylan M. Gilley
#   dgilley@purdue.edu


import argparse,sys
import pandas as pd


# Main function
def main(argv):
    """Write the bash file for initializing and running a new hybridmdmc Toy Problem.
    """

    # Create the parser
    parser = argparse.ArgumentParser()

    # Positional argument(s)
    parser.add_argument(dest='system', type=str,
                        help='System name (str).')
    parser.add_argument(dest='prefix', type=str,
                        help='Prefix (str).')
    parser.add_argument(dest='molecule_types', type=str,
                        help='String list of molecule types.')
    parser.add_argument(dest='molecule_counts', type=str,
                        help='String list of molecule counts.')

    # Optional arguments
    parser.add_argument('-parameters', dest='parameters', type=str, default='InitializeParameters.xlsx',
                         help='Name of the parameters excel file (str). Default: InitializeParameters.xlsx')
    parser.add_argument('-queue', dest='queue', type=str, default='bsavoie',
                        help='Queue on which to run (str). Defualt: bsavoie')
    parser.add_argument('-nodes', dest='nodes', type=int, default=1,
                       help='Number of nodes on which to run (int). Default: 1')
    parser.add_argument('-cores', dest='cores', type=int, default=16,
                        help='Number of cores on which to run (int). Defualt: 16')
    parser.add_argument('-timelim', dest='timelim', type=str, default='06:00:00',
                        help='Time limit of run (str). Default: 03:00:00')
    parser.add_argument('-num_voxels', dest='num_voxels', type=str, default='6 6 6',
                        help='String of voxel delineations. Default: 6 6 6')
    parser.add_argument('-diffusivesteps', dest='diffusivesteps', type=str, default='20000',
                        help='Number of diffusive steps to run.')
    parser.add_argument('-atom_style', dest='atom_style', type=str, default='full',
                        help='LAMMPS atom style. Default: full')
    parser.add_argument('-lammps_units', dest='lammps_units', type=str, default='real',
                        help='LAMMPS units. Default: real')

    # Parse the line arguments
    args = parser.parse_args()

    # Create dataframe holding the hybridmdmc parameters for this system by reading the args.parameters excel file
    variables = [
        'Temperature','Pressure',
        'RelaxationTime','DiffusionTime','DiffusionCutoff','ChangeThreshold',
        'Criteria_Slope','Criteria_Cycles','Criteria_RxnSelectionCount',
        'Window_Slope','Window_Mean','Window_Pause','Window_RxnSelection',
        'Scaling_Adjuster','Scaling_Minimum',
    ]
    parameters = pd.read_excel(args.parameters,header=0,index_col=0)
    if args.system not in parameters.columns:
        print('Error! \"{}\" not found in {}. Exiting {}...'.format(args.system,args.parameters,sys.argv[0]))
        return
    extraparameters = [_ for _ in parameters.index if _ not in variables]
    extravariables = [_ for _ in variables if _ not in parameters.index]
    if extraparameters:
        print('Error! Parameters listed in {} that are not expected by {}; {}'.format(args.parameters,sys.argv[0],' ',join(extraparameters)))
    if extravariables:
        print('Error! Parameters missing from {} that are required by {}; {}'.format(args.parameters,sys.argv[0],' '.join(extravariables)))
    if extraparameters or extravariables:
        print('Exiting...')
        return

    # Write the bash file
    with open(args.prefix+'.sh', 'w') as f:
        f.write(
            "#!/bin/bash\n"+\
            "#\n"+\
            "#SBATCH --job-name {}\n".format(args.prefix)+\
            "#SBATCH -o {}.slurm.out\n".format(args.prefix)+\
            "#SBATCH -e {}.slurm.err\n".format(args.prefix)+\
            "#SBATCH -A {}\n".format(args.queue)+\
            "#SBATCH -N {}\n".format(args.nodes)+\
            "#SBATCH -n {}\n".format(args.cores)+\
            "#SBATCH -t {}\n".format(args.timelim)+\
            "\n"+\
            "system=\"{}\"\n".format(args.system)+\
            "prefix=\"{}\"\n".format(args.prefix))
        for var in variables:
            f.write("{}=\"{}\"\n".format(var,parameters.loc[var,args.system]))
        f.write(


            "\n"+\
            "# Set environment variables\n"+\
            "export PATH=\"/depot/bsavoie/apps/openmpi/3.0.1/bin:$PATH\"\n"+\
            "export LD_LIBRARY_PATH=\"/depot/bsavoie/apps/openmpi/3.0.1/lib:$LD_LIBRARY_PATH\"\n"+\
            "\n"+\
            "# Adjust modules\n"+\
            "module --force purge\n"+\
            "module load intel/17.0.1.132\n"+\
            "export MKL_DEBUG_CPU_TYPE=5\n"+\
            "export MKL_CBWR=AUTO\n"+\
            "\n"+\
            "# Write out job information\n"+\
            "echo \"Running on host: $SLURM_NODELIST\"\n"+\
            "echo \"Running on node(s): $SLURM_NNODES\"\n"+\
            "echo \"Number of processors: $SLURM_NPROCS\"\n"+\
            "echo \"Current working directory: $SLURM_SUBMIT_DIR\"\n"+\
            "\n"+\
            "# User supplied shell commands\n"+\
            "cd $SLURM_SUBMIT_DIR\n"+\
            "\n"+\
            "# Run script\n"+\
            "echo \"Start time: $(date)\"\n"+\
            "\n"+\
            
            "rm ${prefix}.concentration\n"+\
            "rm ${prefix}.scale\n"+\
            "rm ${prefix}.diffusion\n"+\
            "rm ${prefix}.log\n"+\

            "\n"+\
            "# System prep\n"+\
            "python3 ~/bin/hybrid_mdmc/gen_initial_hybridmdmc.py ${system} ${prefix} "+\
            "\'{}\' \'{}\' ".format(args.molecule_types,args.molecule_counts)+\
            "-msf ${system}.msf -header ${system}.header -pressure ${Pressure} -temp ${Temperature} "+\
            "-lammps_units {}\n".format(args.lammps_units)+\
            "mpirun -np {} ".format(args.cores)+\
            "/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in  ${prefix}.in.init > ${prefix}.lammps.out\n"+\
            "cp ${prefix}.in.init               ${prefix}_prep.in.init\n"+\
            "cp ${prefix}.in.data               ${prefix}_prep.in.data\n"+\
            "cp ${prefix}.end.data              ${prefix}_prep.end.data\n"+\
            "cp ${prefix}.lammps.out            ${prefix}_prep.lammps.out\n"+\
            "cp ${prefix}.lammps.log            ${prefix}_prep.lammps.log\n"+\
            "cp ${prefix}.thermo.avg            ${prefix}_prep.thermo.avg\n"+\
            "cp ${prefix}.relax.lammpstrj       ${prefix}_prep.relax.lammpstrj\n"+\
            "cp ${prefix}.density.lammpstrj     ${prefix}_prep.density.lammpstrj\n"+\
            "cp ${prefix}.heat.lammpstrj        ${prefix}_prep.heat.lammpstrj\n"+\
            "cp ${prefix}.diffusion.lammpstrj   ${prefix}_prep.diffusion.lammpstrj\n"+\
            "\n"+\
            "# Reactive loop\n"+\
            "for i in `seq 0 {}`;do\n".format(int(args.diffusivesteps))+\
            "\n"+\
            "    # Run RMD script\n"+\
            "    python3 ~/bin/hybrid_mdmc/hybridmdmc.py ${prefix}.end.data -trj_file ${prefix}.diffusion.lammpstrj\\\n"+\
            "        -msf ${system}.msf -rxndf ${system}.rxndf -settings ${system}.in.settings -header ${system}.header\\\n"+\
            "        -diffusion_step ${i}\\\n"+\
            "        -prefix ${prefix}\\\n"+\
            "        -temp ${Temperature}\\\n"+\
            "        -relax ${RelaxationTime}\\\n"+\
            "        -diffusion ${DiffusionTime}\\\n"+\
            "        -atom_style {}\\\n".format(args.atom_style)+\
            "        -lammps_units {}\\\n".format(args.lammps_units)+\
            "        -num_voxels '{}'\\\n".format(args.num_voxels)+\
            "        -change_threshold ${ChangeThreshold}\\\n"+\
            "        -diffusion_cutoff ${DiffusionCutoff}\\\n"+\
            "        -scalerates 'cumulative'\\\n"+\
            "        -scalingcriteria_concentration_slope ${Criteria_Slope} \\\n"+\
            "        -scalingcriteria_concentration_cycles ${Criteria_Cycles} \\\n"+\
            "        -scalingcriteria_rxnselection_count ${Criteria_RxnSelectionCount} \\\n"+ 
            "        -windowsize_slope ${Window_Slope} \\\n"+\
            "        -windowsize_scalingpause ${Window_Pause} \\\n"+\
            "        -windowsize_rxnselection ${Window_RxnSelection} \\\n"+\
            "        -scalingfactor_adjuster ${Scaling_Adjuster} \\\n"+\
            "        -scalingfactor_minimum ${Scaling_Minimum} \\\n"+\
            "        --well_mixed \\\n"+\
            "        --no-charged_atoms\\\n"+\
            "    \n\n"+\
            "    # Run MD\n"+\
            "    mpirun -np {} ".format(args.cores)+\
                "/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in  ${prefix}.in.init > ${prefix}.lammps.out\n"+\
            "    \n"+\
            "    # Concatenate files\n"+\
            "    python3 ~/bin/concatenate_files.py ${prefix}.in.data             ${prefix}.master.in.data             -bookmark \"Step ${i}\"\n"+\
            "    python3 ~/bin/concatenate_files.py ${prefix}.end.data            ${prefix}.master.end.data            -bookmark \"Step ${i}\"\n"+\
            "    python3 ~/bin/concatenate_files.py ${prefix}.thermo.avg          ${prefix}.master.thermo.avg          -bookmark \"Step ${i}\"\n"+\
            "    python3 ~/bin/concatenate_files.py ${prefix}.relax.lammpstrj     ${prefix}.master.relax.lammpstrj     -bookmark \"Step ${i}\"\n"+\
            "    python3 ~/bin/concatenate_files.py ${prefix}.diffusion.lammpstrj ${prefix}.master.diffusion.lammpstrj -bookmark \"Step ${i}\"\n"+\
            "\n"+\
            "done\n"+\
            "\n"+\
            "echo \"End time: $(date)\"\n"
        )

    return

if __name__ == '__main__':
    main(sys.argv[1:])
