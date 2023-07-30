#!/uar/bin/env python3
# Author
#   Dylan M Gilley
#   dgilley@purdue.edu

import argparse

class HMDMC_argparser():

    def __init__(self,parser=None):
        if not parser:
            parser = argparse.Argumentself.Parser()
        self.parser = parser

    def add_defaults(self):

        # Positional arguments
        self.parser.add_argument(dest='data_file', type=str,
                            help='Name of the LAMMPS data file to read in.')

        # Optional arguments - files
        self.parser.add_argument('-prefix', dest='prefix', type=str, default='default',
                            help='Prefix of the rxndf file, msf file, and all output files. Prefix can be overridden for individual files. Default: data_file prefix')
        self.parser.add_argument('-conc_file', dest='conc_file', type=str, default='default',
                            help='Name of the concentration file. If not provided, the prefix is prepended to \.concentration"')
        self.parser.add_argument('-log_file', dest='log_file', type=str, default='default',
                            help='Name of the log file. If not provided, the prefix is prepended to ".log"')
        self.parser.add_argument('-scale_file', dest='scale_file', type=str, default='default',
                            help='Name of the scale file. If not provided, the prefix is prepended to ".log"')
        self.parser.add_argument('-settings', dest='settings', type=str, default='default',
                            help='Name of the settings file for LAMMPS MD run. If not provided, the prefix is preprended to ".in.settings"')
        self.parser.add_argument('-write_data', dest='write_data', type=str, default='default',
                            help='Name of the data file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.data"')
        self.parser.add_argument('-write_init', dest='write_init', type=str, default='default',
                            help='Name of the init file to write for LAMMPS MD run. If not provided, the prefix is preprended ".in.init"')
        self.parser.add_argument('-header', dest='header', type=str, default='default',
                            help='Name of the data header file. If not provided, the rpefix is prepended to ".header"')
        self.parser.add_argument('-rxndf', dest='rxndf', type=str, default='default',
                            help='Name of the rxndf file. If not provided, the prefix is preprended to ".rxndf"')
        self.parser.add_argument('-msf', dest='msf', type=str, default='default',
                            help='Name of the msf file. If not provided, the prefix is preprended to ".msf"')

        # Optional arguments - Diffusion information
        self.parser.add_argument('-temp', dest='temp', type=float, default=298.0,
                            help='Temperature of the simulation (K). Default: 298.0 K')
        self.parser.add_argument('-relax', dest='relax', type=float, default=1.0e3,
                            help='Length of the relaxation nve/lim run. Default: 1.0e3 (fs)')
        self.parser.add_argument('-diffusion', dest='diffusion', type=float, default=1.0e4,
                            help='Length of the diffusion npt run. Default: 1.0e4 (fs)')

        # Optional arguments - KMC parameters
        self.parser.add_argument('-change_threshold', dest='change_threshold', type=float, default=0.10,
                            help='Fraction of original molecules to react before this reactive KMC cycle completes.')
        self.parser.add_argument('-diffusion_cutoff', dest='diffusion_cutoff', type=float, default=0.4,
                            help='Minimum diffusion coefficient to consider for possible reaction between voxels.')
        self.parser.add_argument('-kmc_type', dest='kmc_type', type=str, default='rejection_free',
                            help='Type of KMC to perform in each voxel.')
        self.parser.add_argument('-diffusion_step', dest='diffusion_step', type=int, default=0,
                            help='Diffusion step. Default: 0.')
        self.parser.add_argument('-scalerates', dest='scalerates', type=str, default='cumulative',
                            help='Option for scaling reaction rates when species mole fractions go stagnant. Options: cumulative, static, off.')
        self.parser.add_argument('-scalingcriteria_rollingmean_stddev', dest='scalingcriteria_rollingmean_stddev', type=float, default=0.1,
                            help='Maximum (less than or equal to) standard deviation of the rolling mean of the number fraction of a molecular species'+\
                            'for that number fraction to be considered stagnant.')
        self.parser.add_argument('-scalingcriteria_rollingmean_cycles', dest='scalingcriteria_rollingmean_cycles', type=int, default=3,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.parser.add_argument('-scalingcriteria_concentration_slope', dest='scalingcriteria_concentration_slope', type=float, default=0.1,
                            help='Maximum (less than or equal to) slope of the number fraction of a molecular species for that number fraction to be considered stagnant.')
        self.parser.add_argument('-scalingcriteria_concentration_cycles', dest='scalingcriteria_concentration_cycles', type=int, default=1,
                            help='Minimum (greater than or equal to) number of MDMC cycles that a species number fraction must be stagnant for reactions involving that species to be scaled.')
        self.parser.add_argument('-scalingcriteria_rxnselection_count', dest='scalingcriteria_rxnselection_count', type=int, default=1,
                            help='Minimum number of times (greater than or equal to) that a reaciton must be selected in the previous _ steps in order ot be a candidate for scaling.')
        self.parser.add_argument('-windowsize_rollingmean', dest='windowsize_rollingmean', type=int, default=3,
                            help='Window size for the calculation of the rolling mean, measured in MDMC cycles.')
        self.parser.add_argument('-windowsize_slope', dest='windowsize_slope', type=int, default=5,
                            help='Window size for the calculation of the slope of concentration, measured in MDMC cycles.')
        self.parser.add_argument('-windowsize_scalingpause', dest='windowsize_scalingpause', type=int, default=3,
                            help='Number of MDMC cycles after a reaciton is scaled or unscaled before it can be scaled again.')
        self.parser.add_argument('-windowsize_rxnselection', dest='windowsize_rxnselection', type=int, default=10,
                            help='Window size for checking the number of times a reaction has been selected.')
        self.parser.add_argument('-scalingfactor_adjuster', dest='scalingfactor_adjuster', type=float, default=0.1,
                            help='Quantity which a reaction rate is multiplied by to scale that rate.')
        self.parser.add_argument('-scalingfactor_minimum', dest='scalingfactor_minimum', type=float, default=1e-6,
                            help='Minimum reaction rate scaling factor for all reactions.')
        self.parser.add_argument('-charged_atoms', dest='charged_atoms', type=bool, default=True,
                        help='If True, atoms have charges. If False, atoms are treated as LJ particles and written without charges.')

        # Optional arguments - flags
        self.parser.add_argument('--debug', dest='debug', default=False, action='store_const', const=True)
        self.parser.add_argument('--log', dest='log', default=False, action='store_const', const=True)

    def parse_args(self):
        self.args = self.parser.parse_args()

    def adjust_args(self):
        if self.args.prefix == 'default':
            self.prefix = self.data_file
            if self.data_file.endswith('.in.data'):
                self.prefix = self.data_file[:-8]
            elif self.data_file.endswith('.end.data'):
                self.prefix = self.data_file[:-9]
            elif self.data_file.endswith('.data'):
                self.prefix = self.data_file[:-5]

        if self.rxndf == 'default':
            self.rxndf = self.prefix+'.rxndf'
        if self.msf == 'default':
            self.msf = self.prefix+'.msf'
        if self.settings == 'default':
            self.settings = self.prefix+'.in.settings'
        if self.header == 'default':
            self.header = self.prefix+'.header'
        if self.write_data == 'default':
            self.write_data = self.prefix+'.in.data'
        if self.write_init == 'default':
            self.write_init = self.prefix+'.in.init'
        if self.conc_file == 'default':
            self.conc_file = self.prefix+'.concentration'
        if self.log_file == 'default':
            self.log_file = self.prefix+'.log'
        if self.scale_file == 'default':
            self.scale_file = self.prefix+'.scale'

        self.scalingcriteria_rollingmean_cycles = int(float(self.args.scalingcriteria_rollingmean_cycles))
        self.scalingcriteria_concentration_slope = float(self.scalingcriteria_concentration_slope)
        self.scalingcriteria_concentration_cycles = int(float(self.scalingcriteria_concentration_cycles))
        self.scalingcriteria_rxnselection_count = int(float(self.scalingcriteria_rxnselection_count))

        self.windowsize_rollingmean = int(float(self.windowsize_rollingmean))
        self.windowsize_slope = int(float(self.windowsize_slope))
        self.windowsize_scalingpause = int(float(self.windowsize_scalingpause))
        self.windowsize_rxnselection = int(float(self.windowsize_rxnselection))

        self.scalingfactor_adjuster = float(self.scalingfactor_adjuster)
        self.scalingfactor_minimum = float(self.scalingfactor_minimum)