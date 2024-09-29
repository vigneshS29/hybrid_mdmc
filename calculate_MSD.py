import numpy as np
from hybrid_mdmc.frame_generator import *
from hybrid_mdmc.data_file_parser import parse_data_file
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='filename_data')
    parser.add_argument(dest='filename_trj')
    parser.add_argument('lammps_type')
    parser.add_argument('-atom_style', dest='atom_style', default='full')
    parser.add_argument('-frames', dest='frames', default='0 1 1000')
    parser.add_argument('-filename_output', dest='filename_output', default='default')
    args = parser.parse_args()

    args.lammps_type = [int(_) for _ in args.lammps_type.split()]
    args.frames = [int(_) for _ in args.frames.split()]
    if args.filename_output == 'default':
        args.filename_output = args.filename_data.split('.')[0] + '.msdoutput.txt'

    msd = MSDHandler(args.filename_data, args.filename_trj, args.filename_output, args.lammps_type, atom_style=args.atom_style, frames=args.frames)
    msd.calculate_msd()
    
    with open(msd.filename_output,'w') as f:
        f.write('timesteps\n{}\n\n'.format([_ for _ in msd.timesteps]))
        f.write('boxes\n{}\n\n'.format([_ for _ in msd.boxes]))
        f.write('msd_mean\n{}\n\n'.format([_ for _ in msd.msd_mean]))
        f.write('msd_std\n{}\n\n'.format([_ for _ in msd.msd_std]))

    return

class MSDHandler():

    def __init__(self, filename_data, filename_trj, filename_output, lammps_type, atom_style='full', frames=[0,1,100]):
        self.filename_data = filename_data
        self.filename_trj = filename_trj
        self.filename_output = filename_output
        self.lammps_type = lammps_type
        self.atom_style = atom_style
        self.frames = frames
        self.parse_data_file()
        return
    
    def parse_data_file(self):
        outputs = parse_data_file(
            self.filename_data,
            atom_style = self.atom_style,
            preserve_atom_order=False,
            preserve_bond_order=False,
            preserve_angle_order=False,
            preserve_dihedral_order=False,
            preserve_improper_order=False,
            tdpd_conc=[],
            unwrap=False)
        labels = ['atoms','bonds','angles','dihedrals','impropers','box','adj_mat','extra_prop']
        for idx,val in enumerate(labels):
            setattr(self, val, outputs[idx])
        return
    
    def get_centers_of_mass(self):
        if not hasattr(self,'atoms'):
            self.parse_data_file()
        centers_of_mass_IDs = self.atoms.ids[self.atoms.get_idx(lammps_type=self.lammps_type)]
        centers_of_mass = np.zeros((len(centers_of_mass_IDs),int((self.frames[2]-self.frames[0])/self.frames[1])))
        timesteps = []
        boxes = []
        frame_idx = 0
        for atom,timestep,box in frame_generator(self.filename_trj,start=self.frames[0],end=self.frames[2]-1,every=self.frames[1],unwrap=False):
            idxs = atom.get_idx(ids=centers_of_mass_IDs)
            centers_of_mass[:,frame_idx] = np.array([np.sqrt(atom.x[idx]**2 + atom.y[idx]**2 + atom.z[idx]**2) for idx in idxs])
            timesteps.append(timestep)
            boxes.append(box)
            frame_idx += 1
        setattr(self, 'centers_of_mass', centers_of_mass)
        setattr(self, 'timesteps', np.array([int(_) for _ in timesteps]))
        setattr(self, 'boxes', boxes)
        return
    
    def calculate_msd(self):
        if not hasattr(self, 'centers_of_mass'):
            self.get_centers_of_mass()

        # Get the number of atoms and frames
        num_atoms, num_frames = self.centers_of_mass.shape

        # Initialize an array to store MSD values
        msd_values = np.zeros(num_frames - 1)
        msd_std = np.zeros(num_frames - 1)

        # Loop over all possible frame differences (lag times)
        for lag in range(1, num_frames):
            # Calculate the squared displacements for the given lag time
            displacements = self.centers_of_mass[:, lag:] - self.centers_of_mass[:, :-lag]
            squared_displacements = displacements ** 2

            # Average over all atoms and all pairs of frames with the given lag
            msd_values[lag - 1] = np.mean(squared_displacements)
            msd_std[lag-1] = np.std(squared_displacements)

        setattr(self, 'msd_mean', msd_values)
        setattr(self, 'msd_std', msd_std)
        return
    
if __name__ == '__main__':                                                                                                                                             
    main(sys.argv[1:])
