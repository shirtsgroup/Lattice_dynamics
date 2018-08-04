#!/usr/bin/env python
from __future__ import print_function
import subprocess
import Expand as Ex
import ThermodynamicProperties as Pr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

def program_cutoff(Program):
    if Program == 'Tinker':
        cutoff = 5e-04
    elif Program == 'Test':
        cutoff = 5e-05
    elif Program == 'CP2K':
        cutoff = 5e-04
    elif Program == 'QE':
	cutoff = 5e-04
    return cutoff

def isotropic_gradient_settings(Coordinate_file, Program, Parameter_file, molecules_in_coord, min_RMS_gradient, Output):
    # Determining the file ending based on the program
    file_ending = Ex.assign_file_ending(Program)

    # Setting the energy cutoff
    cutoff = program_cutoff(Program)

    # Fractional step sizes to take
    steps = np.array([5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02, 5e-02, 1e-01, 5e-01])

    # Number of total step sizes
    n_steps = len(steps)

    # Potential energy of input file and a place to store the expanded structures potential energy
    U_0 = Pr.Potential_energy(Program, Coordinate_file=Coordinate_file, Parameter_file=Parameter_file) / \
          molecules_in_coord
    U = np.zeros((n_steps))

    for i in range(n_steps):
        # Setting how much the lattice parameters must be changed
        dlattice_parameters = Ex.Isotropic_Change_Lattice_Parameters(1. + steps[i], Program, Coordinate_file)

        # Expanding the strucutre
        Ex.Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, Output,
                            min_RMS_gradient, Parameter_file=Parameter_file, dlattice_parameters=dlattice_parameters)

        # Computing the potential energy
        U[i] = Pr.Potential_energy(Program, Coordinate_file=Output + file_ending, Parameter_file=Parameter_file) / \
               molecules_in_coord
        subprocess.call(['rm', Output + file_ending])

        if (U[i] - U_0) > cutoff:
            # Ending the run if we've exceeded the energy cut-off
            LocGrd_Vol_FracStep = steps[i]
            steps = steps[:i + 1]
            U = U[:i + 1]
            break

    # Plotting the results
    plt.plot(np.log10(steps), U - U_0, linestyle='--', marker='o')
    plt.xlabel('$\log({V/V_{0}})$', fontsize=22)
    plt.ylabel('$\Delta U$ [kcal/mol]', fontsize=22)
    plt.ylim((0., 2*cutoff))
    plt.axhline(y=cutoff, c='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig(Output + '_LocGrd_Vol_FracStep.png')
    plt.close()

    # Printing step size
    print("After analysis, LocGrd_Vol_FracStep = ", LocGrd_Vol_FracStep)

    # initial volume
    V_0 = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)

    # returning the value of dV
    return LocGrd_Vol_FracStep * V_0



def anisotropic_gradient_settings(Coordinate_file, Program, Parameter_file, molecules_in_coord, min_RMS_gradient, Output):
    # Determining the file ending based on the program
    file_ending = Ex.assign_file_ending(Program)

    # Setting the energy cutoff
    cutoff = program_cutoff(Program)

    # Fractional step sizes to take
    steps = np.array([5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02, 5e-02, 1e-01, 5e-01])

    # Number of total step sizes
    n_steps = len(steps)

    # Potential energy of input file and a place to store the expanded structures potential energy
    U_0 = Pr.Potential_energy(Program, Coordinate_file=Coordinate_file, Parameter_file=Parameter_file) / \
          molecules_in_coord
    U = np.zeros((6, n_steps))

    # Determining the tensor parameters of the input file
    crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file)))

    LocGrd_CMatrix_FracStep = np.zeros(6)
    for j in range(6):
        plot_marker = False
        for i in range(n_steps):
            dlattice_matrix_array = np.zeros(6)
            dlattice_matrix_array[j] = np.absolute(crystal_matrix_array[j] * steps[i])
            if dlattice_matrix_array[j] < 1e-7:
                continue
            dlattice_matrix = Ex.array_to_triangle_crystal_matrix(dlattice_matrix_array)

            Ex.Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord,
                                Output, min_RMS_gradient, Parameter_file=Parameter_file,
                                dcrystal_matrix=dlattice_matrix)
            U[j, i] = Pr.Potential_energy(Program, Coordinate_file=Output + file_ending, Parameter_file=Parameter_file) / molecules_in_coord
            subprocess.call(['rm', Output + file_ending])
            if (U[j, i] - U_0) > cutoff:
                LocGrd_CMatrix_FracStep[j] = steps[i]
                plot_marker = True
                break
        if plot_marker == True:
            plt.plot(np.log10(steps[:i + 1]), U[j, :i + 1] - U_0, linestyle='--', marker='o', label='C' + str(j + 1))

    # Plotting the results
    plt.xlabel('$\log({C/C_{0}})$', fontsize=22)
    plt.ylabel('$\Delta U$ [kcal/mol]', fontsize=22)
    plt.ylim((0., 2*cutoff))
    plt.axhline(y=cutoff, c='grey', linestyle='--')
    plt.legend(loc='upper right',ncol=2, fontsize=18)
    plt.tight_layout()
    plt.savefig(Output + '_LocGrd_CMatrix_FracStep.png')
    plt.close()

    # Printing step size
    print("After analysis, LocGrd_CMatrix_FracStep = ", LocGrd_CMatrix_FracStep)

    # returning the value of dV
    return np.absolute(LocGrd_CMatrix_FracStep * crystal_matrix_array)


def anisotropic_gradient_settings_1D(Coordinate_file, Program, Parameter_file, molecules_in_coord, min_RMS_gradient,
                                     Output, dC_dLambda):
    # Determining the file ending based on the program
    file_ending = Ex.assign_file_ending(Program)

    # Setting the energy cutoff
    cutoff = program_cutoff(Program)

    # Lambda step sizes to take
    steps = np.array([5e-02, 1e-01, 5e-01, 1e01, 5e01, 1e02, 5e02, 1e03, 5e03])

    # Number of total step sizes
    n_steps = len(steps)

    # Potential energy of input file and a place to store the expanded structures potential energy
    U_0 = Pr.Potential_energy(Program, Coordinate_file=Coordinate_file, Parameter_file=Parameter_file) / \
          molecules_in_coord
    U = np.zeros(n_steps)

    for i in range(n_steps):
        dlattice_matrix = Ex.array_to_triangle_crystal_matrix(steps[i] * dC_dLambda)

        Ex.Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord,
                            Output, min_RMS_gradient, Parameter_file=Parameter_file,
                            dcrystal_matrix=dlattice_matrix)
        U[i] = Pr.Potential_energy(Program, Coordinate_file=Output + file_ending, Parameter_file=Parameter_file) / \
               molecules_in_coord
        subprocess.call(['rm', Output + file_ending])
        if (U[i] - U_0) > cutoff:
            LocGrd_dLambda = steps[i]
            end_plot = i
            break

    # Plotting the results
#    plt.plot(np.log10(steps[:end_plot + 1]), U[:end_plot + 1] - U_0, linestyle='--', marker='o')
#    plt.xlabel('$\log({\lambda})$', fontsize=22)
#    plt.ylabel('$\Delta U$ [kcal/mol]', fontsize=22)
#    plt.ylim((0., 2*cutoff))
#    plt.axhline(y=cutoff, c='grey', linestyle='--')
#    plt.tight_layout()
#    plt.show()

    # returning the value of dV
    return LocGrd_dLambda
