#!/usr/bin/env python3
#
# Author
#   Dylan M. Gilley
#   dgilley@purdue.edu


import os,argparse,sys,datetime
import pandas as pd
import numpy as np
from hybrid_mdmc.parsers import parse_concentration


# Main function
def main(argv):
    """Write the bash file for continuing a hybridmdmc Toy Problem.
    """

    # Create the parser
    parser = argparse.ArgumentParser()

    # Positional argument(s)
    parser.add_argument(dest='system', type=str,
                        help='System name (str).')

    parser.add_argument(dest='replicate', type=str,
                        help='Replicate (str).')

    parser.add_argument(dest='steps', type=int,
                        help='Number of steps.')

    # Optional arguments
    parser.add_argument('-parameters', dest='parameters', type=str, default='InitializeParameters.xlsx',
                         help='Name of the parameters excel file (str). Default: InitializeParameters.xlsx')

    parser.add_argument('-queue', dest='queue', type=str, default='bsavoie',
                        help='Queue on which to run (str). Defualt: bsavoie')

    parser.add_argument('-nodes', dest='nodes', type=int, default=1,
                       help='Number of nodes on which to run (int). Default: 1')

    parser.add_argument('-cores', dest='cores', type=int, default=16,
                        help='Number of cores on which to run (int). Defualt: 16')

    parser.add_argument('-timelim', dest='timelim', type=str, default='04:00:00',
                        help='Time limit of run (str). Default: 04:00:00')

    parser.add_argument('-voxels', dest='voxels', type=str, default='6 6 6',
                        help='String of voxel delineations. Default: 6 6 6')

    # Parse the line arguments
    args = parser.parse_args()

    # Create dataframe holding the hybridmdmc parameters for this system by reading the args.parameters excel file
    variables = [
        'Temperature',
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

    # Determine the next step number
    run = args.system + '-' + args.replicate
    counts,times,rxns = parse_concentration('{}.concentration'.format(run))
    begin = 1 + np.sum([1 for diffusion in counts.keys() for _ in counts[diffusion].keys()])

    # Write the bash file
    with open(run+'.sh', 'w') as f:
        f.write(
            "#!/bin/bash\n"+\
            "#\n"+\
            "#SBATCH --job-name {}\n".format(run)+\
            "#SBATCH -o {}.slurm.out\n".format(run)+\
            "#SBATCH -e {}.slurm.err\n".format(run)+\
            "#SBATCH -A {}\n".format(args.queue)+\
            "#SBATCH -N {}\n".format(args.nodes)+\
            "#SBATCH -n {}\n".format(args.cores)+\
            "#SBATCH -t {}\n".format(args.timelim)+\
            "\n"+\
            "system=\"{}\"\n".format(args.system)+\
            "prefix=\"{}\"\n".format(run))
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
            "# Reactive loop\n"+\
            "for i in `seq {} {}`;do\n".format(begin,begin+args.steps)+\
            "\n"+\
            "    # Run RMD script\n"+\
            "    python3 ~/bin/hybrid_mdmc/write_diffusion.py ${prefix}.end.data -prefix ${prefix} -num_voxels "+\
            "'{}' --well_mixed\n".format(args.voxels)+\
            "    python3 ~/bin/hybrid_mdmc/hybridmdmc.py ${prefix}.end.data ${prefix}.diffusion -prefix ${prefix} -diffusion_step ${i}\\\n"+\
            "        -msf ${system}.msf -rxndf ${system}.rxndf -settings ${system}.in.settings -header ${system}.header\\\n"+\
            "        -temp ${Temperature} -relax ${RelaxationTime} -diffusion ${DiffusionTime}\\\n"+\
            "        -change_threshold ${ChangeThreshold} -diffusion_cutoff ${DiffusionCutoff} -kmc_type 'rejection_free' -scalerates 'cumulative'\\\n"+\
            "        -scalingcriteria_concentration_slope ${Criteria_Slope} -scalingcriteria_concentration_cycles ${Criteria_Cycles}\\\n"+\
            "        -scalingcriteria_rxnselection_count ${Criteria_RxnSelectionCount}\\\n"+\
            "        -windowsize_slope ${Window_Slope} -windowsize_scalingpause ${Window_Pause} -windowsize_rxnselection ${Window_RxnSelection}\\\n"+\
            "        -scalingfactor_adjuster ${Scaling_Adjuster} -scalingfactor_minimum ${Scaling_Minimum}\n"+\
            "    \n"+\
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
