

------------------------------------------------------------------------------------------------------------------------
----------- ---------- ---------- ---------- ---------- Workflow ---------- ---------- ---------- ---------- -----------
------------------------------------------------------------------------------------------------------------------------

1. Create .rxndf, .msf, .header, and .in.settings files for the system of interest. See below for descriptions of each of these files. These files remain unchanged throughout the KMC/MD cycles, assuming the POSSIBLE (not necessarily present) species remains unchanged. That is, if a complete reaction network is created before hand, from which all possible species are determined, these files can be created at the beginning of the simulation. If the reaction network is appended after the Hybrid MD/KMC simulation has begun, these will need to be edited. It is advised that the former method is adopted. Currnetly, these files must be created manually.

2.Initialize a system, the result of which should be a LAMMPS data file containing the relaxed positions of the desired initial molecules. This can be done with "gen_initial_hybridmdmc.py" by specifying the system name, replicate, and molecule counts/types. This script generates an unrelaxed LAMMPS data file and an accompanying LAMMPS init file, which is setup to relax the system. See below for more details.

3. Execute the Hybrid MD/KMC procedure. Currently, this is performed using a bash script to iteratively loop over a series of steps performing diffusion and then KMC reaction selection. The looping procedure consists of the following steps:
   i. Perform MD simulation, relaxation followed by diffusion.
   ii. Analyzing the diffusion just performed, create the .diffusion file to determine voxel bounds and diffusion times between voxels. See below for details.
   iii. Perform KMC reaction selection. This is done by running "hybridmdmc.py" with the desired parameters.

The above steps can be performed with varying levels of manual input. For the Toy Problems included in the "Example" directory, the "write_initial_hybridmdmc_bash.py" script greatly simplifies this. Running this script creates a bash file that will perform all of the above steps. This script requires the use of an excel sheet containing the hybridmdmc parameters, included in the "Example" directory for some Toy Problems. To run the Toy Problems, simply run this script, then sbatch the resulting .sh file.

The package uses molecules/species as the bases for identifying reactive pairs and atoms to delete or add (documentation will use the words "molecule" and "species" interchangeably). UNIQUE names for species must be created by the user and USED CONSISTENTLY throughout all files. This is important to keep in mind for systems that include isomers.


------------------------------------------------------------------------------------------------------------------------
---------- ---------- ---------- ---------- ------------- Files ------------- ---------- ---------- ---------- ---------
------------------------------------------------------------------------------------------------------------------------

Every file used in the Hybrid MD/KMC package can be read using the custom parsers included in the package. All parsers should ignore blank lines and any line beginning with "#". If a parser crashes due to attempting to read a blank line or a line beginning with "#", please report this as it is a bug. Formatting rules and other information for LAMMPS files can be found on the LAMMPS website, https://docs.lammps.org/Manual.html. Custom file types created for the Hybrid MD/KMC package are described here, and examples of each can be found in the "Examples" directory.

Reaction Data File (.rxndf)
This file defines every reaction possible. Each reaction will be labeled with a reaction number, and will include the reactants and products (listed as molecules), and parameters allowing for the Arrhenius equation to be used to calculate the reaction rate. The reaciton numbers will be used throughout the scripts, and thus are unique the the specific reactions.

Master Species File (.msf)
This file defines the molecules that can appear in the system. For each molecule, it lists the atoms present, the partial charge of each atom (if coulombic forces are included), and the intramolecular interactions with their correpsonding parameters. The atom IDs are arbitrary, only used for bookkeeping within this file. The atom types, however, correspond to the atom types used throughout the simulation, including the LAMMPS settings file, and thus should be rigorously maintained.

Header File (.header)
This file contains a "header" of a LAMMPs data file. Included are total interaction types and particle masses. Because not all of the possible molecules and thus particles or interactions are present at the beginning of the simulation, or at any point in the simulation, this information won't be copied into new LAMMPS data files without explictly writing them in using the information provided in the header file. 


------------------------------------------------------------------------------------------------------------------------
--------- ---------- ---------- ---------- ---------- hybridmdmc.py ---------- ---------- ---------- ---------- --------
------------------------------------------------------------------------------------------------------------------------

This script analyzes information describing the previous difusive MD run, performs KMC reaction selection and time update, then creates files necessary for the next diffusive MD run. As stated in the workflow section, a full Hybrid MD/KMC simulation consists of iterative calls to this script. This script usually performs multiple cycles of KMC reaction selection, but this can be altered based on the arguments. The script begins by reading in information describing the current configuration, the voxel bounds, the diffusion scaling between voxels, the definition and Arrhenius parameters of all possible reactions, the definition and interaction parameters of all possible molecules, and the LAMMPS header information (i.e. mass of each particle type). Arguments passed to this script can be separated into bookkeeping, MD parameters for the ensuing diffusion, and KMC parameters.

~Necessary Files~
1. LAMMPS data file (.data)
   - ending LAMMPS data file, generated at the end of the MD diffusion
2. Diffusion file (.diffusion)
   - created with "write_diffusion.py"
   - defines the voxel bounds and the diffusion coefficients between each
3. Header file (.header)
4. Reaction Data File (.rxndf)
5. Master Species File (.msf)

~Output~

~Dependencies~

~KMC Reaction Selection~


