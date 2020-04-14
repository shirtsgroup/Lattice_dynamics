#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import subprocess
import scipy.optimize
import numpy as np
import ThermodynamicProperties as Pr
import Expand as Ex
import program_specific_functions as psf


def constrained_minimization(inputs, Coordinate_file, Program, Parameter_file=''):
    # Determining the file ending of the coordinate file
    file_ending = psf.assign_coordinate_file_ending(Program)

    # Determining the lattice parameters and volume of the input coordinate file
    lp_0 = psf.Lattice_parameters(Program, Coordinate_file)
    V0 = Pr.Volume(lattice_parameters=lp_0)
    cm_0 = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(lp_0))

    bnds = np.matrix([[cm_0[0] - cm_0[0] * 0.2, cm_0[0] + cm_0[0] * 0.2],
                      [cm_0[1] - cm_0[1] * 0.2, cm_0[1] + cm_0[1] * 0.2],
                      [cm_0[2] - cm_0[2] * 0.2, cm_0[2] + cm_0[2] * 0.2],
                      [cm_0[3] - cm_0[0] * 0.2, cm_0[3] + cm_0[0] * 0.2],
                      [cm_0[4] - cm_0[0] * 0.2, cm_0[4] + cm_0[0] * 0.2],
                      [cm_0[5] - cm_0[0] * 0.2, cm_0[5] + cm_0[0] * 0.2]])

    output = scipy.optimize.minimize(Return_U_from_Aniso_Expand, cm_0, (inputs, Coordinate_file,
                                                                        Parameter_file, Program,
                                                                        'constV_minimize', 4), method='SLSQP',
                                     constraints=({'type': 'eq', 'fun': lambda cm:
                                     np.linalg.det(Ex.array_to_triangle_crystal_matrix(cm)) - V0}), bounds=bnds,
                                     tol=1e-08,options={'ftol':float(1e-6), 'disp':True, 'eps':float(1e-4)})

    dlattice_parameters = Ex.crystal_matrix_to_lattice_parameters(Ex.array_to_triangle_crystal_matrix(output.x)) - lp_0
    Ex.Expand_Structure(inputs, Coordinate_file, 'lattice_parameters', 'constV_minimize',
                        dlattice_parameters=dlattice_parameters)
    subprocess.call(['mv', 'constV_minimize' + file_ending, Coordinate_file])


def Return_U_from_Aniso_Expand(inputs, new_crystal_matrix, coordinate_file, Parameter_file, Program, output_file_name,
                               molecules_in_coord):

    # Converting the crystal matrix parameter to lattice parameters
    new_lattice_parameters = Ex.crystal_matrix_to_lattice_parameters(
        Ex.array_to_triangle_crystal_matrix(new_crystal_matrix))

    # Determining the file ending of the coordinate file
    file_ending = psf.assign_coordinate_file_ending(Program)

    # Determine the input coordinate files lattice parameters
    old_lattice_parameters = psf.Lattice_parameters(Program, coordinate_file)

    # Expand input coordinate file using the new lattice parameters
    Ex.Expand_Structure(inputs, coordinate_file, 'lattice_parameters', output_file_name,
                        dlattice_parameters=new_lattice_parameters[:6] - old_lattice_parameters)
    # Computing the potential energy of the new expanded structure
    U = psf.Potential_energy(output_file_name + file_ending, Program, Parameter_file=Parameter_file) / molecules_in_coord
    return U



