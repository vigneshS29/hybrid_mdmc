#!/usr/bin/env python

import numpy as np

# Simple function for testing whether a given string can be made into a float
def isfloat_str(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# Function for writing a lammps data file, given a "data" dictionary
def write_lammps_data(file_name,atoms,bonds,angles,dihedrals,impropers,box,header={},charge=True):

    header_defaults = {
        'atoms': atoms.ids.flatten().shape[0],
        'atom_types': len(set(atoms.lammps_type)),
        'bonds': bonds.ids.flatten().shape[0],
        'bond_types': len(set(bonds.lammps_type)),
        'angles': angles.ids.flatten().shape[0],
        'angle_types': len(set(angles.lammps_type)),
        'dihedrals': dihedrals.ids.flatten().shape[0],
        'dihedral_types': len(set(dihedrals.lammps_type)),
        'impropers': impropers.ids.flatten().shape[0],
        'improper_types': len(set(impropers.lammps_type))}
    if 'masses' not in header.keys():
        header_defaults['masses'] = sorted(list(set([ (atoms.lammps_type[idx],atoms.mass[idx]) for idx in range(len(atoms.ids)) ])))

    for keyword in header_defaults.keys():
        if keyword not in header:
            header[keyword] = header_defaults[keyword]

    with open(file_name,'w') as f:

        f.write('LAMMPS data file.\n\n')

        f.write('{:<5d} {}\n'.format(header['atoms'],'atoms') )
        f.write('{:<5d} {}\n'.format(header['atom_types'],'atom types') )
        f.write('{:<5d} {}\n'.format(header['bonds'],'bonds') )
        f.write('{:<5d} {}\n'.format(header['bond_types'],'bond types') )
        f.write('{:<5d} {}\n'.format(header['angles'],'angles') )
        f.write('{:<5d} {}\n'.format(header['angle_types'],'angle types') )
        f.write('{:<5d} {}\n'.format(header['dihedrals'],'dihedrals') )
        f.write('{:<5d} {}\n'.format(header['dihedral_types'],'dihedral types') )
        f.write('{:<5d} {}\n'.format(header['impropers'],'impropers') )
        f.write('{:<5d} {}\n\n'.format(header['improper_types'],'improper types') )

        f.write('{:<12.8f} {:<12.8f} {} {}\n'.format(box[0][0],box[0][1],'xlo','xhi'))
        f.write('{:<12.8f} {:<12.8f} {} {}\n'.format(box[1][0],box[1][1],'ylo','yhi'))
        f.write('{:<12.8f} {:<12.8f} {} {}\n\n'.format(box[2][0],box[2][1],'zlo','zhi'))

        f.write('Masses\n\n')
        for mass in header['masses']:
            f.write('{:<5d} {:<8.4f}\n'.format(mass[0],mass[1]))

        f.write('\nAtoms\n\n')
        if charge:
            for idx,atom_id in enumerate(atoms.ids):
                f.write('{:<5d} {:<5d} {:<5d} {:>14.8f} {:>14.8f} {:>14.8f} {:>14.8f}\n'.format(
                    atom_id,atoms.mol_id[idx],atoms.lammps_type[idx],atoms.charge[idx],atoms.x[idx],atoms.y[idx],atoms.z[idx]))
        if not charge:
            for idx,atom_id in enumerate(atoms.ids):
                f.write('{:<5d} {:<5d} {:<5d} {:>14.8f} {:>14.8f} {:>14.8f}\n'.format(
                    atom_id,atoms.mol_id[idx],atoms.lammps_type[idx],atoms.x[idx],atoms.y[idx],atoms.z[idx]))

        if bonds.ids.flatten().shape[0]>0:
            f.write('\nBonds\n\n')
            for idx,bond_id in enumerate(bonds.ids):
                f.write('{:<4d} {:<4d} {:<4d} {:<4d}\n'.format(bond_id,bonds.lammps_type[idx],bonds.atom_ids[idx][0],bonds.atom_ids[idx][1]))

        if angles.ids.flatten().shape[0]>0:
            f.write('\nAngles\n\n')
            for idx,angle_id in enumerate(angles.ids):
                f.write('{:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n'.format(angle_id,angles.lammps_type[idx],angles.atom_ids[idx][0],angles.atom_ids[idx][1],angles.atom_ids[idx][2]))

        if dihedrals.ids.flatten().shape[0]>0:
            f.write('\nDihedrals\n\n')
            for idx,dihedral_id in enumerate(dihedrals.ids):
                f.write('{:<4d} {:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n'.format(
                    dihedral_id,dihedrals.lammps_type[idx],dihedrals.atom_ids[idx][0],dihedrals.atom_ids[idx][1],dihedrals.atom_ids[idx][2],dihedrals.atom_ids[idx][3]))

        if impropers.ids.flatten().shape[0]>0:
            f.write('\nImpropers\n\n')
            for idx,improper_id in enumerate(impropers.ids):
                f.write('{:<4d} {:<4d} {:<4d} {:<4d} {:<4d} {:<4d}\n'.format(
                    improper_id,impropers.lammps_type[idx],impropers.atom_ids[idx][0],impropers.atom_ids[idx][1],impropers.atom_ids[idx][2],impropers.atom_ids[idx][3]))

    return

# Function for writing a lammps input file
# init_info = {
#    'settings': str,
#    'prefix': str,
#    'avg_freq': int,
#    'coords_freq': int,
#    'thermo_freq': int,
#    'dump4avg': int,
#    'run_type': [ str, ],
#    'run_name': [ str, ],
#    'run_steps': [ int, ],
#    'run_temp': [ [float,float,float], ],
#    'run_press': [ [float,float,float], ],
#    'restart': Boolean,
#    'reset_steps': Boolean,
#    'thermo_keywords': [ str, ]
#    }
def write_lammps_init(init_info,out_file,trj=True,step_restarts=True,final_data=True,final_restart=True):

    defaults = {
        'settings': '',
        'prefix': '',
        'data': '',
        'thermo_freq': 1000,
        'avg_freq': 1000,
        'dump4avg': 100,
        'coords_freq': 1000,
        'units': 'real',
        'atom_style': 'full',
        'pair_style': 'lj/cut/coul/long 14.0 14.0',
        'kspace_style': 'pppm 0.0001',
        'bond_style': 'harmonic',
        'angle_style': 'harmonic',
        'dihedral_style': 'opls',
        'improper_style': 'cvff',
        'run_name': [ 'equil1' ],
        'run_type': [ 'npt' ],
        'run_steps': [ 1000000 ],
        'run_temp': [ [298.0,298.0,100.0] ],
        'run_press': [ [1.0,1.0,100.0] ],
        'run_timestep': [1.0],
        'restart': False,
        'reset_steps': False,
        'thermo_keywords': ['temp', 'press', 'ke', 'pe'],
        'neigh_modify': 'every 1 delay 10 check yes one 10000'
        }

    for key in defaults.keys():
        if key not in init_info:
            init_info[key] = defaults[key]

    if init_info['data'] == '':
        init_info['data'] = init_info['prefix']+'.in.data'

    with open(out_file,'w') as f:
        f.write(
            "# LAMMPS init file\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Variables\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "# File-wide variables\n"+\
            "variable    settings_name    index    {}\n".format(init_info['settings'])+\
            "variable    prefix           index    {}\n".format(init_info['prefix'])+\
            "variable    thermo_freq      index    {}\n".format(init_info['thermo_freq'])+\
            "variable    avg_freq         index    {}\n".format(init_info['avg_freq'])+\
            "variable    dump4avg         index    {}\n".format(init_info['dump4avg'])+\
            "variable    coords_freq      index    {}\n".format(init_info['coords_freq'])+\
            "variable    vseed            index    123\n"+\
            "\n")

        for idx in range(len(init_info['run_name'])):
            f.write(
                "# {}\n".format(init_info['run_name'][idx])+\
                "variable    nSteps_{}    index    {:d} # fs\n".format(init_info['run_name'][idx],int(init_info['run_steps'][idx]))+\
                "variable    press0_{}    index    {:.1f} # atm\n".format(init_info['run_name'][idx],init_info['run_press'][idx][0])+\
                "variable    pressf_{}    index    {:.1f} # atm\n".format(init_info['run_name'][idx],init_info['run_press'][idx][1])+\
                "variable    temp0_{}     index    {:.1f} # K\n".format(init_info['run_name'][idx],init_info['run_temp'][idx][0])+\
                "variable    tempf_{}     index    {:.1f} # K\n".format(init_info['run_name'][idx],init_info['run_temp'][idx][1])+\
                "\n")

        if init_info['restart']:
            f.write(
                "#===========================================================\n"+\
                "# Initialize System\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "read_restart {}\n".format(init_info['restart'])+\
                "log ${prefix}.lammps.log\n")
            if init_info['reset_steps']:
                f.write("reset_timestep {}\n".format(int(init_info['reset_steps'])))
            f.write("\n")

        else:
            f.write(
                "#===========================================================\n"+\
                "# Initialize System\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "units {}\n".format(init_info['units'])+\
                "dimension 3\n"+\
                "newton on\n"+\
                "boundary p p p\n"+\
                "atom_style {}\n".format(init_info['atom_style'])+\
                "log ${prefix}.lammps.log\n"+\
                "\n")

        f.write(
            "#===========================================================\n"+\
            "# Force Field Definitions\n"+\
            "#===========================================================\n"+\

            "\n"+\
            "special_bonds   lj 0.0 0.0 0.0 coul 0.0 0.0 0.0    # NO 1-4 LJ/COUL interactions\n"+\
            "pair_style      {}\n".format(init_info['pair_style'])+\
            "pair_modify     shift yes mix sixthpower           # using Waldman-Hagler mixing rules\n")

        if init_info['kspace_style']!=None:
            f.write("kspace_style    {}\n".format(init_info['kspace_style']))
        if init_info['bond_style']!=None:
            f.write("bond_style      {}\n".format(init_info['bond_style']))
        if init_info['angle_style']!=None:
            f.write("angle_style     {}\n".format(init_info['angle_style']))
        if init_info['dihedral_style']!=None:
            f.write("dihedral_style  {}\n".format(init_info['dihedral_style']))
        if init_info['improper_style']!=None:
            f.write("improper_style  {}\n".format(init_info['improper_style']))

        f.write(
            "\n"+\
            
            "#===========================================================\n"+\
            "# Setup System\n"+\
            "#===========================================================\n"+\
            "\n")

        if init_info['restart']:
            f.write(
                "include ${settings_name}\n"+\
                "run_style verlet # Velocity-Verlet integrator\n"+\
                "neigh_modify {}\n".format(init_info['neigh_modify'])+\
                "\n"+\

                "thermo_style custom")

        else:
            f.write(
                "read_data {}\n".format(init_info['data'])+\
                "include ${settings_name}\n"+\
                "run_style verlet # Velocity-Verlet integrator\n"+\
                "neigh_modify {}\n".format(init_info['neigh_modify'])+\
                "\n"+\

                "thermo_style custom")


        for keyword in init_info['thermo_keywords']:
            f.write(" {}".format(keyword))
        f.write("\n")
        f.write(
            "thermo_modify format float %20.10f\n"+\
            "thermo ${thermo_freq}\n"+\
            "\n")

        for keyword in init_info['thermo_keywords']:
            f.write("variable    my_{}  equal  {}\n".format(keyword,keyword))
        f.write(
            "\n"+\

            "# Set the averages fix\n"+\
            "fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq}")

        for keyword in init_info['thermo_keywords']:
            f.write(" v_my_{}".format(keyword))
        f.write(" file ${prefix}.thermo.avg\n\n"+\

            "# Set momentum fix to zero out momentum (linear and angular) every ps\n"+\
            "fix mom all momentum 1000 linear 1 1 1 angular\n"+\
            "velocity all create ${{temp0_{}}} ${{vseed}} mom yes rot yes\n".format(init_info['run_name'][0])+\
            "\n")

        for idx in range(len(init_info['run_name'])):
            name = init_info['run_name'][idx]

            # Simple npt or nvt run
            if init_info['run_type'][idx] == 'npt':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({}, Nose-Hoover)\n".format(init_info['run_name'][idx],init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx]))
                if trj:
                    f.write(
                        "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                        "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx]))
                f.write(
                    "fix {} all {} temp ${{temp0_{}}} ${{tempf_{}}} {} iso ${{press0_{}}} ${{pressf_{}}} {}\n".format(
                        name,init_info['run_type'][idx],name,name,init_info['run_temp'][idx][2],name,name,init_info['run_press'][idx][2])+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix {}\n".format(name))
                if trj:
                    f.write(
                        "undump {}\n".format(name))
                if step_restarts:
                    f.write(
                        "write_restart ${{prefix}}.{}.end.restart\n".format(name))
                f.write("\n")

            # Simple nvt run
            if init_info['run_type'][idx] == 'nvt':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({}, Nose-Hoover)\n".format(init_info['run_name'][idx],init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx]))
                if trj:
                    f.write(
                        "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                        "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx]))
                f.write(
                    "fix {} all {} temp ${{temp0_{}}} ${{tempf_{}}} {}\n".format(
                        name,init_info['run_type'][idx],name,name,init_info['run_temp'][idx][2])+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix {}\n".format(name))
                if trj:
                    f.write(
                        "undump {}\n".format(name))
                if step_restarts:
                    f.write(
                        "write_restart ${{prefix}}.{}.end.restart\n".format(name))
                f.write("\n")

            # An nve with a fix langevin
            if init_info['run_type'][idx] == 'nve/langevin':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({})\n".format(init_info['run_name'][idx],init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx])+\
                    "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                    "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx])+\
                    "fix nve_{} all nve \n".format(name)+\
                    "fix langevin_{} all langevin ${{temp0_{}}} ${{tempf_{}}} {} 123\n".format(name,name,name,init_info['run_temp'][idx][2])+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix nve_{}\n".format(name)+\
                    "unfix langevin_{}\n".format(name)+\
                    "undump {}\n".format(name)+\
                    "write_restart ${{prefix}}.{}.end.restart\n".format(name)+\
                    "\n")

            # An npt with a "fix deform" to final x/y/z values
            if init_info['run_type'][idx] == 'npt/deform/finalxyz':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({})\n".format(init_info['run_name'][idx],init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx])+\
                    "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                    "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx])+\
                    "fix {}_deform all deform 1 x final {} {} y final {} {} z final {} {} remap x units box\n".format(
                        name,init_info['deform'][idx][0][0],init_info['deform'][idx][0][1],init_info['deform'][idx][1][0],init_info['deform'][idx][1][1],init_info['deform'][idx][2][0],init_info['deform'][idx][2][1])+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix {}_deform\n".format(name)+\
                    "undump {}\n".format(name)+\
                    "write_restart ${{prefix}}.{}.end.restart\n".format(name)+\
                    "\n")

            # An npt with a "fix deform" at specified erate
            if init_info['run_type'][idx] == 'npt/deform/erate':
                stable = ['y','z']
                if init_info['deform'][idx][0] == 'y':
                    stable = ['x','z']
                if init_info['deform'][idx][0] == 'z':
                    stable = ['x','y']
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({})\n".format(init_info['run_name'][idx],init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx])+\
                    "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                    "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx])+\
                    "fix {}_npt all npt temp ${{temp0_{}}} ${{tempf_{}}} {} {} ${{press0_{}}} ${{pressf_{}}} {} {} ${{press0_{}}} ${{pressf_{}}} {}\n".format(
                        name,name,name,init_info['run_temp'][idx][2],stable[0],name,name,init_info['run_press'][idx][2],stable[1],name,name,init_info['run_press'][idx][2])+\
                    "fix {}_deform all deform 1 {} erate {} remap x units box\n".format(name,init_info['deform'][idx][0],init_info['deform'][idx][1])+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix {}_npt\n".format(name)+\
                    "unfix {}_deform\n".format(name)+\
                    "undump {}\n".format(name)+\
                    "write_restart ${{prefix}}.{}.end.restart\n".format(name)+\
                    "\n")


            # nve/lim run
            if init_info['run_type'][idx] == 'nve/limit':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({})\n".format(name,init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx])+\
                    "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                    "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx])+\
                    "fix nve_{} all nve/limit 0.1 \n".format(name)+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix nve_{}\n".format(name)+\
                    "undump {}\n".format(name))
                if step_restarts:
                    f.write(
                        "write_restart ${{prefix}}.{}.end.restart\n".format(name)+\
                        "\n")
                f.write("\n")


            # nve run
            if init_info['run_type'][idx] == 'nve':
                f.write(
                    "#===========================================================\n"+\
                    "# {} ({})\n".format(name,init_info['run_type'][idx])+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "timestep {}\n".format(init_info['run_timestep'][idx])+\
                    "dump {} all custom ${{coords_freq}} ${{prefix}}.{}.lammpstrj id mol type xu yu zu vx vy vz\n".format(init_info['run_name'][idx],init_info['run_name'][idx])+\
                    "dump_modify {} sort id format float %20.10g\n".format(init_info['run_name'][idx])+\
                    "fix {} all nve\n".format(name)+\
                    "run ${{nSteps_{}}}\n".format(name)+\
                    "unfix {}\n".format(name)+\
                    "undump {}\n".format(name)+\
                    "write_restart ${{prefix}}.{}.end.restart\n".format(name)+\
                    "\n")

        f.write(
            "#===========================================================\n"+\
            "# Clean and exit\n"+\
            "#===========================================================\n"+\
            "\n")
        if final_data:
            f.write("write_data ${prefix}.end.data\n")
        if final_restart:
            f.write("write_restart ${prefix}.end.restart\n")
        f.write(
            "unfix averages\n"+\
            "unfix mom\n")

    return

def parse_lammps_out(fname):

    data = {}
    flag = False
    with open(fname,'r') as f:
        for line in f:
            fields = line.split()
            if fields == []: continue
            if fields[0] == 'Step':
                flag = True
                keywords = fields[1:]
                continue
            if len(fields) > 7:
                if fields[0] == 'Loop' and fields[1] == 'time':
                    flag = False
                    continue
            if flag:
                data[int(fields[0])] = [float(i) for i in fields[1:]]

    return keywords,data

def parse_data_header_and_masses(fname):

    header = {'masses':{}}
    flag = None

    int_types = ['atom','bond','angle','dihedral','improper']

    with open(fname,'r') as f:
        for line in f:

            fields = line.split()
            if fields == []: continue
            if fields[0] == '#': conitnue
            if fields[0] == 'Masses':
                flag = 'Masses'
                continue

            if not flag and len(fields) == 3:
                if fields[1] in int_types and fields[2] == 'types':
                    header[fields[1]+'_types'] = int(fields[0])
                    continue

            if flag == 'Masses':
                try:
                    header['masses'][int(fields[0])] = float(fields[1])
                except:
                    header['masses'] = [ (k,header['masses'][k]) for k in sorted(list(header['masses'].keys())) ]
                    return header

    header['masses'] = [ (k,header['masses'][k]) for k in sorted(list(header['masses'].keys())) ]
    return header
