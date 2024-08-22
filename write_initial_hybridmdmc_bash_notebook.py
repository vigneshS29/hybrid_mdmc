#!/usr/bin/env python3
# Author
#   Dylan M. Gilley
#   dgilley@purdue.edu


import argparse,sys


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

    # Optional arguments
    parser.add_argument('-filename_notebook', dest='filename_notebook', type=str, default='default',
                         help='Name of the parameters excel file (str). Default: default')
    parser.add_argument('-queue', dest='queue', type=str, default='bsavoie',
                        help='Queue on which to run (str). Defualt: bsavoie')
    parser.add_argument('-nodes', dest='nodes', type=int, default=1,
                       help='Number of nodes on which to run (int). Default: 1')
    parser.add_argument('-cores', dest='cores', type=int, default=16,
                        help='Number of cores on which to run (int). Defualt: 16')
    parser.add_argument('-timelim', dest='timelim', type=str, default='06:00:00',
                        help='Time limit of run (str). Default: 03:00:00')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--synchronousparallel', dest='serial', action='store_false')
    parser.set_defaults(serial=True)
    
    # Parse the line arguments
    args = parser.parse_args()
    
    mainscript = '~/bin/hybrid_mdmc/hybridmdmc.py'
    if args.serial:
        mainscript = '~/bin/hybrid_mdmc/hybridmdmc_serial.py'

    # Write the bash file
    with open(args.prefix+'_hmdmc.sh', 'w') as f:
        f.write(
            """\
#!/bin/bash
#
#SBATCH --job-name {}
#SBATCH -o {}.slurm.out
#SBATCH -e {}.slurm.err
#SBATCH -A {}
#SBATCH -N {}
#SBATCH -n {}
#SBATCH -t {}

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

rm {}.concentration
rm {}.scale
rm {}.diffusion
rm {}.log

# System prep
python3 ~/bin/hybrid_mdmc/gen_initial_hybridmdmc_notebook.py {} {} -filename_notebook {} &&
mpirun -np {} /depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in {}.in.init > {}.lammps.out &&
cp {}.in.init               {}_prep.in.init
cp {}.in.data               {}_prep.in.data
cp {}.end.data              {}_prep.end.data
cp {}.lammps.out            {}_prep.lammps.out
cp {}.lammps.log            {}_prep.lammps.log
cp {}.thermo.avg            {}_prep.thermo.avg
cp {}.relax.lammpstrj       {}_prep.relax.lammpstrj
cp {}.density.lammpstrj     {}_prep.density.lammpstrj
cp {}.diffusion.lammpstrj   {}_prep.diffusion.lammpstrj

# Reactive loop
for i in `seq 0 2000`; do

    # Run RMD script
    python3 {} {} {} -filename_notebook {} &&
    if [ $? != 0 ]; then
        exit 1
    fi

    # Run MD
    mpirun -np {} /depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in {}.in.init > {}.lammps.out &&

done

echo "End time: $(date)"
""".format(
    args.prefix+'_hmdmc', args.prefix+'_hmdmc', args.prefix+'_hmdmc', args.queue, args.nodes, args.cores, args.timelim,
    args.prefix, args.prefix, args.prefix, args.prefix,
    args.system, args.prefix, args.filename_notebook,
    args.cores, args.prefix, args.prefix,
    args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix,
    args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix, args.prefix,
    mainscript, args.system, args.prefix, args.filename_notebook,
    args.cores, args.prefix, args.prefix
))

    return


if __name__ == '__main__':
    main(sys.argv[1:])