#!/usr/bin/env python3
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

from argparse import ArgumentParser

class HMDMC_ArgumentParser(ArgumentParser):

    def __init__(self):
        ArgumentParser.__init__(self)
        self.add_default_args()

    def add_default_args(self):

        # Positional arguments
        self.add_argument(dest='data_file', type=str,
                            help='Name of the LAMMPS data file to read in.')

        # Optional arguments - files
        self.add_argument('-prefix', dest='prefix', type=str, default='default',
                            help='Prefix of the rxndf file, msf file, and all output files. Prefix can be overridden for individual files. Default: data_file prefix')
        self.add_argument('-trj_file', dest='trj_file', type=str, default='default',
                            help='Name of the lammps trajectory file. If not provided, the prefix is prepended to ".lammpstrj"')
        self.add_argument('-conc_file', dest='conc_file', type=str, default='default',
                            help='Name of the concentration file. If not provided, the prefix is prepended to ".concentration"')
        self.add_argument('-log_file', dest='log_file', type=str, default='default',
                            help='Name of the log file. If not provided, the prefix is prepended to ".log"')
        self.add_argument('-scale_file', dest='scale_file', type=str, default='default',
                            help='Name of the scale file. If not provided, the prefix is prepended to ".scale"')
        self.add_argument('-settings', dest='settings', type=str, default='default',
                            help='Name of the settings file for LAMMPS MD run. If not provided, the prefix is preprended to ".in.settings"')
        self.add_argument('-write_data', dest='write_data', type=str, default='default',
                            help='Name of the data file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.data"')
        self.add_argument('-write_init', dest='write_init', type=str, default='default',
                            help='Name of the init file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.init"')
        self.add_argument('-header', dest='header', type=str, default='default',
                            help='Name of the data header file. If not provided, the prefix is prepended to ".header"')
        self.add_argument('-rxndf', dest='rxndf', type=str, default='default',
                            help='Name of the rxndf file. If not provided, the prefix is preprended to ".rxndf"')
        self.add_argument('-msf', dest='msf', type=str, default='default',
                            help='Name of the msf file. If not provided, the prefix is preprended to ".msf"')

        # Optional arguments - MD information
        self.add_argument('-temp', dest='temp', default=298.0,
                            help='Temperature of the simulation (K). Default: 298.0 K')
        self.add_argument('-relax', dest='relax', default=1.0e3,
                            help='Length of the relaxation nve/lim run. Default: 1.0e3 (fs)')
        self.add_argument('-diffusion', dest='diffusion', default=1.0e4,
                            help='Length of the diffusion npt run. Default: 1.0e4 (fs)')
        
        # Optional arguments - diffusion information
        self.add_argument('-num_voxels', dest='num_voxels', type=str, default='3 3 3',
                            help='Space delimited string of the number of voxels in the x, y, and z directions. Default: 3 3 3')
        self.add_argument('-x_bounds', dest='x_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the x-axis. Overrides first entry in -num_voxels')
        self.add_argument('-y_bounds', dest='y_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the y-axis. Overrides second entry in -num_voxels')
        self.add_argument('-z_bounds', dest='z_bounds', type=str, default='',
                            help='Space delimited string defining custom bounds for the z-axis. Overrides third entry in -num_voxels')

        # Optional arguments - KMC parameters
        self.add_argument('-change_threshold', dest='change_threshold', default=0.10,
                            help='Fraction of original molecules to react before this reactive KMC cycle completes.')
        self.add_argument('-diffusion_cutoff', dest='diffusion_cutoff', default=0.4,
                            help='Minimum diffusion coefficient to consider for possible reaction between voxels.')
        self.add_argument('-kmc_type', dest='kmc_type', default='rejection_free',
                            help='Type of KMC to perform in each voxel.')
        self.add_argument('-diffusion_step', dest='diffusion_step', default=0,
                            help='Diffusion step. Default: 0.')
        self.add_argument('-scalerates', dest='scalerates', default='cumulative',
                            help='Option for scaling reaction rates when species mole fractions go stagnant. Options: cumulative, static, off.')
        self.add_argument('-scalingcriteria_rollingmean_stddev', dest='scalingcriteria_rollingmean_stddev', default=0.1,
                            help='Maximum (less than or equal to) standard deviation of the rolling mean of the number fraction of a molecular species'+\
                            'for that number fraction to be considered stagnant.')
        self.add_argument('-scalingcriteria_rollingmean_cycles', dest='scalingcriteria_rollingmean_cycles', default=3,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.add_argument('-scalingcriteria_concentration_slope', dest='scalingcriteria_concentration_slope', default=0.1,
                            help='Maximum (less than or equal to) slope of the number fraction of a molecular species for that number fraction to be considered stagnant.')
        self.add_argument('-scalingcriteria_concentration_cycles', dest='scalingcriteria_concentration_cycles', default=1,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.add_argument('-scalingcriteria_rxnselection_count', dest='scalingcriteria_rxnselection_count', default=1,
                            help='Minimum number of times (greater than or equal to) that a reaciton must be selected in the previous _ steps in order ot be a candidate for scaling.')
        self.add_argument('-windowsize_rollingmean', dest='windowsize_rollingmean', default=3,
                            help='Window size for the calculation of the rolling mean, measured in MDMC cycles.')
        self.add_argument('-windowsize_slope', dest='windowsize_slope', default=5,
                            help='Window size for the calculation of the slope of concentration, measured in MDMC cycles.')
        self.add_argument('-windowsize_scalingpause', dest='windowsize_scalingpause', default=3,
                            help='Number of MDMC cycles after a reaciton is scaled or unscaled before it can be scaled again.')
        self.add_argument('-windowsize_rxnselection', dest='windowsize_rxnselection', default=10,
                            help='Window size for checking the number of times a reaction has been selected.')
        self.add_argument('-scalingfactor_adjuster', dest='scalingfactor_adjuster', default=0.1,
                            help='Quantity which a reaction rate is multiplied by to scale that rate.')
        self.add_argument('-scalingfactor_minimum', dest='scalingfactor_minimum', default=1e-6,
                            help='Minimum reaction rate scaling factor for all reactions.')
        self.add_argument('--charged_atoms', action='store_true')
        self.add_argument('--no-charged_atoms', dest='charged_atoms', action='store_false')
        self.set_defaults(charged_atoms=True)

        # Optional arguments - flags
        self.add_argument('--distance_criteria', dest='distance_criteria', default=False, action='store_const', const=True)
        self.add_argument('--well_mixed', dest='well_mixed', default=False, action='store_const', const=True)
        self.add_argument('--debug', dest='debug', default=False, action='store_const', const=True)
        self.add_argument('--log', dest='log', default=False, action='store_const', const=True)

    def HMDMC_parse_args(self):
        self.args = self.parse_args()

    def adjust_default_args(self):

        num_starting_args = len(self.args.__dict__)

        # Handle files
        if self.args.prefix == 'default':
            self.args.prefix = self.args.data_file
            if self.args.data_file.endswith('.in.data'):
                self.args.prefix = self.args.data_file[:-8]
            elif self.args.data_file.endswith('.end.data'):
                self.args.prefix = self.args.data_file[:-9]
            elif self.args.data_file.endswith('.data'):
                self.args.prefix = self.args.data_file[:-5]
        if self.args.rxndf == 'default':
            self.args.rxndf = self.args.prefix+'.rxndf'
        if self.args.msf == 'default':
            self.args.msf = self.args.prefix+'.msf'
        if self.args.settings == 'default':
            self.args.settings = self.args.prefix+'.in.settings'
        if self.args.header == 'default':
            self.args.header = self.args.prefix+'.header'
        if self.args.write_data == 'default':
            self.args.write_data = self.args.prefix+'.in.data'
        if self.args.write_init == 'default':
            self.args.write_init = self.args.prefix+'.in.init'
        if self.args.trj_file == 'default':
            self.args.trj_file = self.args.prefix+'.lammpstrj'
        if self.args.conc_file == 'default':
            self.args.conc_file = self.args.prefix+'.concentration'
        if self.args.log_file == 'default':
            self.args.log_file = self.args.prefix+'.log'
        if self.args.scale_file == 'default':
            self.args.scale_file = self.args.prefix+'.scale'

        # Handle MD information
        self.args.temp = float(self.args.temp)
        self.args.relax = float(self.args.relax)
        self.args.diffusion = float(self.args.diffusion)

        # Handle diffusion information
        self.args.num_voxels = [int(float(_)) for _ in self.args.num_voxels.split()]
        self.args.x_bounds = [int(float(_)) for _ in self.args.x_bounds.split()]
        self.args.y_bounds = [int(float(_)) for _ in self.args.y_bounds.split()]
        self.args.z_bounds = [int(float(_)) for _ in self.args.z_bounds.split()]

        # Handle KMC parameters
        self.args.change_threshold = float(self.args.change_threshold)
        self.args.diffusion_cutoff = float(self.args.diffusion_cutoff)
        self.args.kmc_type = str(self.args.kmc_type)
        self.args.diffusion_step = int(float(self.args.diffusion_step))
        self.args.scalerates = str(self.args.scalerates)
        self.args.scalingcriteria_rollingmean_stddev = float(self.args.scalingcriteria_rollingmean_stddev)
        self.args.scalingcriteria_rollingmean_cycles = int(float(self.args.scalingcriteria_rollingmean_cycles))
        self.args.scalingcriteria_concentration_slope = float(self.args.scalingcriteria_concentration_slope)
        self.args.scalingcriteria_concentration_cycles = int(float(self.args.scalingcriteria_concentration_cycles))
        self.args.scalingcriteria_rxnselection_count = int(float(self.args.scalingcriteria_rxnselection_count))
        self.args.windowsize_rollingmean = int(float(self.args.windowsize_rollingmean))
        self.args.windowsize_slope = int(float(self.args.windowsize_slope))
        self.args.windowsize_scalingpause = int(float(self.args.windowsize_scalingpause))
        self.args.windowsize_rxnselection = int(float(self.args.windowsize_rxnselection))
        self.args.scalingfactor_adjuster = float(self.args.scalingfactor_adjuster)
        self.args.scalingfactor_minimum = float(self.args.scalingfactor_minimum)

        # Check that all arguments were appropriately handled
        if len(self.args.__dict__) != num_starting_args:
            print('Error! The number of arguments changed during argument adjustment.'+\
                '\nPlease check hybrid_mdmc.customargparse.HMDMC_ArgumentParser.adjust_default_args()'+\
                '\nExiting...')
            self.args = None
            quit()