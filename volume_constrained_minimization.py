#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import subprocess
import scipy.optimize
import numpy as np
import ThermodynamicProperties as Pr
import Expand as Ex

def constrained_minimization(Coordinate_file, Program, molecules_in_coord=1, min_RMS_gradient=1e-04,
                                      Parameter_file=''):
    # Determining the file ending of the coordinate file
    file_ending = Ex.assign_file_ending(Program)

    # Determining the lattice parameters and volume of the input coordinate file
    lp_0 = Pr.Lattice_parameters(Program, Coordinate_file)
    V0 = Pr.Volume(lattice_parameters=lp_0)
    cm_0 = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(lp_0))

    U0 = Pr.Potential_energy(Coordinate_file, Program, Parameter_file=Parameter_file)

    # Copy input coordinate file into a temporary file
    subprocess.call(['cp', Coordinate_file, 'constV_minimize' + file_ending])

    isotropic_gradients = dfunc('constV_minimize' + file_ending, Parameter_file, Program, molecules_in_coord,
                                      min_RMS_gradient, V0)

    minimize = True
    count = 0
    while minimize == True:
        # Setting the minimization bounds
        bnds = np.matrix([[cm_0[0] - cm_0[0] * 0.2, cm_0[0] + cm_0[0] * 0.2],
                          [cm_0[1] - cm_0[1] * 0.2, cm_0[1] + cm_0[1] * 0.2],
                          [cm_0[2] - cm_0[2] * 0.2, cm_0[2] + cm_0[2] * 0.2],
                          [cm_0[3] - cm_0[0] * 0.2, cm_0[3] + cm_0[0] * 0.2],
                          [cm_0[4] - cm_0[0] * 0.2, cm_0[4] + cm_0[0] * 0.2],
                          [cm_0[5] - cm_0[0] * 0.2, cm_0[5] + cm_0[0] * 0.2]])
        # Minimizing the systems potential energy by changing the lattice parameters while constraining the volume
        if count == 0:
            output = scipy.optimize.minimize(Return_U_from_Aniso_Expand, cm_0, ('constV_minimize' + file_ending,
                                                                                Parameter_file, Program,
                                                                                'temp_constV_minimize', molecules_in_coord,
                                                                                min_RMS_gradient), method='SLSQP',
                                             constraints=({'type': 'eq', 'fun': lambda cm:
                                             np.linalg.det(Ex.array_to_triangle_crystal_matrix(cm)) - V0}), bounds=bnds,
                                             tol=1e-07)
            output = output.x
        else:
            output = scipy.optimize.minimize(off_diag_minimization, cm_0[:3], ('constV_minimize' + file_ending,
                                                                                Parameter_file, Program,
                                                                                'temp_constV_minimize', molecules_in_coord,
                                                                                min_RMS_gradient, np.array([0,1,2])), method='SLSQP',
                                             constraints=({'type': 'eq', 'fun': lambda cm:
                                             cm[0] * cm[1] * cm[2] - V0}), bounds=bnds[:3],
                                             tol=1e-07)
            #gradients = dfunc('temp_constV_minimize' + file_ending, Parameter_file, Program, molecules_in_coord,
            #                  min_RMS_gradient, V0)
            output = np.append(output.x, cm_0[3:])
        

        dlattice_parameters = Ex.crystal_matrix_to_lattice_parameters(Ex.array_to_triangle_crystal_matrix(output)) - Pr.Lattice_parameters(Program, 'constV_minimize' + file_ending)
        Ex.Expand_Structure('constV_minimize' + file_ending, Program, 'lattice_parameters', molecules_in_coord, 'temp_constV_minimize',
                            min_RMS_gradient, dlattice_parameters=dlattice_parameters,
                            Parameter_file=Parameter_file)

        U = Pr.Potential_energy('temp_constV_minimize' + file_ending, Program, Parameter_file=Parameter_file)
        # Will only move on if the energy is less than the preivous structure
        gradients = dfunc('temp_constV_minimize' + file_ending, Parameter_file, Program, molecules_in_coord,
                          min_RMS_gradient, V0)
        if U <= U0:
            #print(gradients)

            subprocess.call(['mv', 'temp_constV_minimize' + file_ending, 'constV_minimize' + file_ending])
            U0 = 1. * U

            if np.any(np.absolute(gradients[3:6]) > 1e-05):
                # Determining where the values are to large
                placement = np.where(np.absolute(gradients[3:6]) > 1e-03)[0] + 3
    
                output_2 = scipy.optimize.minimize(off_diag_minimization, output[placement], ('constV_minimize' + file_ending,
                                                                                     Parameter_file, Program,
                                                                                     'temp_constV_minimize',
                                                                                     molecules_in_coord,
                                                                                     min_RMS_gradient, placement),
                                        method='Nelder-Mead',
                                        tol=1e-07)

                new_crystal_array = 1. * output
                new_crystal_array[placement] = output_2.x
                dlattice_parameters =  Ex.crystal_matrix_to_lattice_parameters(Ex.array_to_triangle_crystal_matrix(new_crystal_array)) - Ex.crystal_matrix_to_lattice_parameters(Ex.array_to_triangle_crystal_matrix(output))
                Ex.Expand_Structure('constV_minimize' + file_ending, Program, 'lattice_parameters', molecules_in_coord, 'temp_constV_minimize',
                            min_RMS_gradient, dlattice_parameters=dlattice_parameters,
                            Parameter_file=Parameter_file)
                U = Pr.Potential_energy('temp_constV_minimize' + file_ending, Program, Parameter_file=Parameter_file)
                if U <= U0:
                    U0 = 1. * U
                    gradients = dfunc('temp_constV_minimize' + file_ending, Parameter_file, Program, molecules_in_coord,
                                      min_RMS_gradient, V0)
                    #print(gradients)
                    subprocess.call(['mv', 'temp_constV_minimize' + file_ending, 'constV_minimize' + file_ending])
            elif np.any(np.absolute(gradients[:3]) > 1e-05) and np.any(np.absolute(gradients[3:6]) < 1e-05):
                pass
            else:
                minimize = False
        else:
            print("Lattice energy has not been reduced, exiting run")
            subprocess.call(['rm', 'temp_constV_minimize' + file_ending])
            minimize = False

        # Exiting if it's been running too long
        if count == 10:
            print("Have run sub-routine 10 times, exiting.")
            minimize = False

        cm_0 = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
            Pr.Lattice_parameters(Program, 'constV_minimize' + file_ending)))
        count = count + 1

        subprocess.call(['cp', 'constV_minimize' + file_ending, 'hold_for_failure' + file_ending])

    print(np.around(np.array(gradients)[:6] / np.array(isotropic_gradients)[:6],4))
    # Replacing the coordinate file with the new minimized structure
    subprocess.call(['rm', 'hold_for_failure' + file_ending])
    subprocess.call(['mv', 'constV_minimize' + file_ending, Coordinate_file])

def Return_U_from_Aniso_Expand(new_crystal_matrix, coordinate_file, Parameter_file, Program, output_file_name,
                               molecules_in_coord, min_RMS_gradient):

    # Converting the crystal matrix parameter to lattice parameters
    new_lattice_parameters = Ex.crystal_matrix_to_lattice_parameters(
        Ex.array_to_triangle_crystal_matrix(new_crystal_matrix))

    # Determining the file ending of the coordinate file
    file_ending = Ex.assign_file_ending(Program)

    # Determine the input coordinate files lattice parameters
    old_lattice_parameters = Pr.Lattice_parameters(Program, coordinate_file)

    # Expand input coordinate file using the new lattice parameters
    Ex.Expand_Structure(coordinate_file, Program, 'lattice_parameters', molecules_in_coord, output_file_name,
                        min_RMS_gradient, dlattice_parameters=new_lattice_parameters[:6] - old_lattice_parameters,
                        Parameter_file=Parameter_file)
    # Computing the potential energy of the new expanded structure
    U = Pr.Potential_energy(output_file_name + file_ending, Program, Parameter_file=Parameter_file) / molecules_in_coord
    return U

def dfunc(coordinate_file, Parameter_file, Program, molecules_in_coord, min_RMS_gradient, V0):
    # crystal matrix parameters
    x = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
        Pr.Lattice_parameters(Program, coordinate_file)))

    # Assigning the system file ending
    file_ending = Ex.assign_file_ending(Program)

    # Setting an array for energy gradients
    dU = compute_dU(x, coordinate_file, Parameter_file, Program, molecules_in_coord, min_RMS_gradient, file_ending)

    # Solving for the value of lambda that minimizes the lagrangian equation
    L = compute_Lambda(x, dU)
    return constrained_minimization_gradients(L, x, dU, V0)

def compute_Lambda(x, dU):
    return (x[1] * x[2] * dU[0] + x[0] * x[2] * dU[1] + x[0] * x[1] * dU[2]) / ((x[1] * x[2]) ** 2 + (x[0] * x[2]) ** 2 + (x[0] * x[1]) ** 2)

def compute_dU(x, coordinate_file, Parameter_file, Program, molecules_in_coord, min_RMS_gradient, file_ending):
    # Setting an array for energy gradients
    dU = np.zeros(len(x))

    # Numerical step sizes for dU
    h = [1e-03, 1e-03, 1e-03, 1e-3, 1e-3, 1e-3]

    for i in range(len(x)):
        # Setting the change in the crystal matrix parameters
        dX = np.zeros(len(x))
        dX[i] = h[i]

        # Computing the energy gradient due to change a particular crystal matrix parameter
        dU[i] = (Return_U_from_Aniso_Expand(x + dX, coordinate_file, Parameter_file, Program, 'temp',
                                            molecules_in_coord, min_RMS_gradient) -
                 Return_U_from_Aniso_Expand(x - dX, coordinate_file, Parameter_file, Program, 'temp',
                                            molecules_in_coord, min_RMS_gradient)) / (2 * h[i])

        # Removing excess files
        subprocess.call(['rm', 'temp' + file_ending])
    return dU

def constrained_minimization_gradients(L, x, dU, V0):
    eq0 = dU[0] - L * x[1] * x[2]
    eq1 = dU[1] - L * x[0] * x[2]
    eq2 = dU[2] - L * x[0] * x[1]

    eq3 = dU[3]
    eq4 = dU[4]
    eq5 = dU[5]

    eq6 = V0 - np.linalg.det(Ex.array_to_triangle_crystal_matrix(x))
    return eq0, eq1, eq2, eq3, eq4, eq5, eq6


def off_diag_minimization(new_values, coordinate_file, Parameter_file, Program, output_file_name, molecules_in_coord,
                          min_RMS_gradient, placement):
    # Putting the parameters in their correct place
    new_crystal_matrix = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
        Pr.Lattice_parameters(Program, coordinate_file)))

    for i in range(len(new_values)):
        new_crystal_matrix[placement[i]] = new_values[i]

    U = Return_U_from_Aniso_Expand(new_crystal_matrix, coordinate_file, Parameter_file, Program, output_file_name,
                                   molecules_in_coord, min_RMS_gradient)
    return U








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Geometry optimizes a coordinate file while constraining the volume')
    parser.add_argument('-C', '--coord_file', dest='coord_file', default='None',
                        help='Coordinate file to minimize')
    parser.add_argument('-p', '--program', dest='program', default='None',
                        help='Program for the coordinate file')
    parser.add_argument('-k', '--parameter_file', dest='parameter_file', default='None',
                        help='Parameter file if needed')

    args = parser.parse_args()
    if args.coord_file == 'None':
        print("Coordinate file must be specified by -C")
        print("Exiting")
        sys.exit()
    elif args.program == 'None':
        print("Program for coordinate file must be specified by -p")
        print("Exiting")
        sys.exit()

    constrained_minimization(args.coord_file, args.program, Parameter_file=args.parameter_file)
