#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import subprocess
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr
import Numerical_Outputs as NO
from munkres import Munkres, print_matrix
import program_specific_functions as psf


##########################################
#                 Input                  #
##########################################
def Call_Wavenumbers(inputs, **keyword_parameters):
    """
    This function helps to direct how the wavenumbers will be calculated and calls other functions to calculate and 
    return the wavenumbers

    **Required Inputs
    Method = Harmonic approximation ('HA');
             Stepwise Isotropic QHA ('SiQ');
             Stepwise Isotropic QHA w/ Gruneisen Parameter ('SiQg');
             Gradient Isotropic QHA ('GiQ');
             Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
             Gradient Anisotropic QHA ('GaQ');
             Gradient Anisotropic QHA w/ Gruneisen Parameter ('GaQg');

    **Optional Inputs
    Output = Name of file to put wavenumbers into, if it already exists it will be loaded
    Gruneisen = Gruneisen parameters found with Setup_Isotropic_Gruneisen
    Wavenumber_Reference = Reference wavenumbers for Gruneisen parameter, will be output from Setup_Isotropic_Gruneisen
    Volume_Reference = Reference volume of structure for Wavenumber_Reference, will be output from 
    Setup_Isotropic_Gruneisen
    New_Volume = Volume of new structure to calculate wavenumbers for
    Gruneisen_Vol_FracStep = Volumetric stepsize to expand lattice minimum structure to numerically determine the
    Gruneisen parameter
    molecules_in_coord = number of molecules in Coordinate_file
    Coordinate_file = File containing lattice parameters and atom coordinates
    Parameter_file = Optional input for program
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    Crystal_matrix_Reference
    New_Crystal_matrix
    Gruneisen_Lat_FracStep
    """
    if (inputs.method == 'SiQ') or (inputs.method == 'GiQ') or (inputs.method == 'GaQ') or (inputs.method == 'HA'):
        return psf.program_wavenumbers(keyword_parameters['Coordinate_file'], inputs.tinker_parameter_file,
                                       inputs.output, inputs.coordinate_file, inputs.program, inputs.method)

    elif (inputs.method == 'SiQg') or (inputs.method == 'GiQg'):
        # Methods that use the Gruneisen parameter
        if ('Gruneisen' in keyword_parameters) and ('Wavenumber_Reference' in keyword_parameters) and \
                ('Volume_Reference' in keyword_parameters) and ('New_Volume' in keyword_parameters):
            # Calculating the wavenumbers of the new Isotropically expanded structure
            wavenumbers = Get_Iso_Gruneisen_Wavenumbers(keyword_parameters['Gruneisen'],
                                                        keyword_parameters['Wavenumber_Reference'],
                                                        keyword_parameters['Volume_Reference'],
                                                        keyword_parameters['New_Volume'])
            return wavenumbers
        else:
            # If there is a saved Gruneisen parameter and set of wavenumbers
            if os.path.isfile(inputs.output + '_GRUwvn_' + inputs.method + '.npy') and \
                    os.path.isfile(inputs.output + '_GRU_' + inputs.method + '.npy'):
                print("   ...Using Gruneisen parameters from: " + inputs.output + '_GRU_' + inputs.method + '.npy')
                gruneisen = np.load(inputs.output + '_GRU_' + inputs.method + '.npy')
                wavenumber_reference = np.load(inputs.output + '_GRUwvn_' + inputs.method + '.npy')
                volume_reference = Pr.Volume(Coordinate_file=inputs.coordinate_file, Program=inputs.program,
                                             Parameter_file=inputs.tinker_parameter_file)
            # If the Gruneisen parameter has yet to be determined, here it will be calculated
            # It is assumed that the input Coordinate_file is the lattice minimum strucutre
            else:
                gruneisen, wavenumber_reference, volume_reference = Setup_Isotropic_Gruneisen(inputs)
                print("   ... Saving reference wavenumbers and Gruneisen parameters to: " + inputs.output
                      + '_GRU_' + inputs.method + '.npy')
                np.save(inputs.output + '_GRU_' + inputs.method, gruneisen)
                np.save(inputs.output + '_GRUwvn_' + inputs.method, wavenumber_reference)
            return gruneisen, wavenumber_reference, volume_reference

    elif inputs.method == 'GaQg':
        if ('Gruneisen' in keyword_parameters) and ('Wavenumber_Reference' in keyword_parameters) and \
                ('ref_crystal_matrix' in keyword_parameters):
            # Calculating the wavenumbers of the new anisotropically expanded structure
            # The Gruniesen parameter and reference wavenumbers have already been calculated
            wavenumbers = Get_Aniso_Gruneisen_Wavenumbers(keyword_parameters['Gruneisen'],
                                                          keyword_parameters['Wavenumber_Reference'],
                                                          keyword_parameters['ref_crystal_matrix'],
                                                          keyword_parameters['Coordinate_file'],
                                                          inputs.program)
            return wavenumbers

        else:
            if os.path.isfile(inputs.output + '_GRUwvn_' + inputs.method + '.npy') and \
                    os.path.isfile(inputs.output + '_GRU_' + inputs.method + '.npy'):
                # If the current directory has saved Gruneisen outputs, it will open those and use them
                print("   ...Using Gruneisen parameters from: " + inputs.output + '_GRU_' + inputs.method + '.npy')
                gruneisen = np.load(inputs.output + '_GRU_' + inputs.method + '.npy')
                wavenumber_reference = np.load(inputs.output + '_GRUwvn_' + inputs.method + '.npy')
            else:
                # Calculating the Gruneisen parameter and wavenumbers
                gruneisen, wavenumber_reference = Setup_Anisotropic_Gruneisen(inputs)

                # Saving the wavenumbers for future use
                print("   ... Saving reference wavenumbers and Gruneisen parameters to: " + inputs.output
                      + '_GRU_/_GRUwvn' + inputs.method + '.npy')
                np.save(inputs.output + '_GRU_' + inputs.method, gruneisen)
                np.save(inputs.output + '_GRUwvn_' + inputs.method, wavenumber_reference)
            return gruneisen, wavenumber_reference
    elif inputs.method == 'SaQply':
        return psf.Wavenumber_and_Vectors(inputs.program, keyword_parameters['Coordinate_file'],
                                          inputs.tinker_parameter_file)


##########################################
#     Isotropic Gruneisen Parameter      #
##########################################
def Setup_Isotropic_Gruneisen(inputs):
    """
    This function calculates the Isotropic Gruneisen parameters for a given coordinate file.
    Calculated numerically given a specified volume fraction stepsize
    ******Eventually! Impliment a second order Gruneisen parameter in here

    **Required Inputs
    Coordinate_file = File containing lattice parameters and atom coordinates
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    Gruneisen_Vol_FracStep = Volumetric stepsize to expand lattice minimum structure to numerically determine the 
    Gruneisen parameter
    molecules_in_coord = number of molecules in Coordinate_file

    **Optional inputs
    Parameter_file = Optional input for program
    """
    # Change in lattice parameters for expanded structure
    dLattice_Parameters = Ex.Isotropic_Change_Lattice_Parameters((1 + inputs.gruneisen_volume_fraction_stepsize),
                                                                 inputs.program, inputs.coordinate_file)

    # Determining wavenumbers of lattice strucutre and expanded strucutre
    # Also, assigning a file ending name for the nex coordinate file (program dependent)
    lattice_parameters = psf.Lattice_parameters(inputs.program, inputs.coordinate_file)

    if inputs.program == 'Tinker':
        Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', 'temp',
                            dlattice_parameters=dLattice_Parameters)
        Organized_wavenumbers = Tinker_Gru_organized_wavenumbers('Isotropic', inputs.coordinate_file, 'temp.xyz',
                                                                 inputs.tinker_parameter_file)
        Wavenumber_Reference = Organized_wavenumbers[0] 
        Wavenumber_expand = Organized_wavenumbers[1]
        file_ending = '.xyz'

    if inputs.program == 'CP2K':
        Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', 'temp',
                            dlattice_parameters=dLattice_Parameters)
        Organized_wavenumbers = CP2K_Gru_organized_wavenumbers('Isotropic', inputs.coordinate_file, 'temp.pdb',
                                                               inputs.tinker_parameter_file, inputs.output)
        Wavenumber_Reference = Organized_wavenumbers[0] 
        Wavenumber_expand = Organized_wavenumbers[1]
        file_ending = '.pdb'

    elif inputs.program == 'QE':
        Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', inputs.coordinate_file[0:-3],
                            dlattice_parameters=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', 'temp',
                            dlattice_parameters=dLattice_Parameters)
        Organized_wavenumbers = QE_Gru_organized_wavenumbers('Isotropic', inputs.coordinate_file, 'temp.pw',
                                                             inputs.tinker_parameter_file, inputs.output)
        Wavenumber_Reference = Organized_wavenumbers[0]
        Wavenumber_expand = Organized_wavenumbers[1]
        file_ending = '.pw'

    elif inputs.program == 'Test':
        Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', 'temp',
                            dlattice_parameters=dLattice_Parameters)
        Wavenumber_Reference = psf.program_wavenumbers(inputs.coordinate_file, 0, 0, 0, inputs.program, 'GiQg')
        Wavenumber_expand = psf.program_wavenumbers('temp.npy', 0, 0, 0, inputs.program, 'GiQg')
        file_ending = '.npy'

    # Calculating the volume of the lattice minimum and expanded structure
    Volume_Reference = Pr.Volume(lattice_parameters=lattice_parameters)
    Volume_expand = Volume_Reference + inputs.gruneisen_volume_fraction_stepsize * Volume_Reference

    Gruneisen = np.zeros(len(Wavenumber_Reference))
    Gruneisen[3:] = -(np.log(Wavenumber_Reference[3:]) - np.log(Wavenumber_expand[3:]))/(np.log(Volume_Reference) -
                                                                                         np.log(Volume_expand))
    for x in range(0,len(Gruneisen)):
        if Wavenumber_Reference[x] == 0:
            Gruneisen[x] = 0.0
    # Removing extra files created in process
    subprocess.call(['rm', 'temp' + file_ending])
    return Gruneisen, Wavenumber_Reference, Volume_Reference


def Get_Iso_Gruneisen_Wavenumbers(Gruneisen, Wavenumber_Reference, Volume_Reference, New_Volume): 
    """
    This function calculates new wavenumber for an isotropically expanded strucutre using the gruneisen parameter
    ******Eventually! Impliment a second order Gruneisen parameter in here

    **Required Inputs
    Gruneisen = Gruneisen parameters found with Setup_Isotropic_Gruneisen
    Wavenumber_Reference = Reference wavenumbers for Gruneisen parameter, will be output from Setup_Isotropic_Gruneisen
    Volume_Reference = Reference volume of strucutre for Wavenumber_Reference, will be output from Setup_Isotropic_Gruneisen
    New_Volume = Volume of new structure to calculate wavenumbers for
    """
    wavenumbers = np.diag(np.power(New_Volume/Volume_Reference, -1*Gruneisen))
    wavenumbers = np.dot(Wavenumber_Reference, wavenumbers)
    return wavenumbers


##########################################
# Strain Ansotropic Gruneisen Parameter  #
##########################################

def Setup_Anisotropic_Gruneisen(inputs):
    # Determining the file ending for the program used
    file_ending = psf.assign_coordinate_file_ending(inputs.program)

    # Starting by straining the crystal in the six principal directions
    re_run = False
    if os.path.isfile('GRU_wvn.npy') and os.path.isfile('GRU_eigen.npy'):
        wavenumbers = np.load('GRU_wvn.npy')
        eigenvectors = np.load('GRU_eigen.npy')
        re_run = True
    else:
        # Computing the reference wavenumbers and eigenvectors
        wavenumbers_ref, eigenvectors_ref = psf.Wavenumber_and_Vectors(inputs.program, inputs.coordinate_file,
                                                                       inputs.tinker_parameter_file)

        # Setting the number of vibrational modes
        number_of_modes = int(len(wavenumbers_ref))

        # Setting a place to store all the wavenumbers and eigenvalues for the Gruenisen paremeters
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
        wavenumbers[0] = wavenumbers_ref
        eigenvectors[0] = eigenvectors_ref

        # Outputing information for user output
        NO.start_anisoGru()

        # Saving that data computed thusfar in case the system crashes or times out
        np.save('GRU_eigen', eigenvectors)
        np.save('GRU_wvn', wavenumbers)

    for i in range(6):
        if re_run and (wavenumbers[i + 1, 3] != 0.):
            pass
        else:
            # Making expanded structures in th direction of the six principal strains
            applied_strain = np.zeros(6)
            applied_strain[i] = inputs.gruneisen_matrix_strain_stepsize
            Ex.Expand_Structure(inputs, inputs.coordinate_file, 'strain', 'temp',
                                strain=Ex.strain_matrix(applied_strain),
                                crystal_matrix=Ex.Lattice_parameters_to_Crystal_matrix(
                                    psf.Lattice_parameters(inputs.program, inputs.coordinate_file)))

            # Computing the strained wavenumbers and eigenvectors
            wavenumbers_unorganized, eigenvectors_unorganized = \
                psf.Wavenumber_and_Vectors(inputs.program, 'temp' + file_ending, inputs.tinker_parameter_file)

            # Determining how the strained eigenvectors match up with the reference structure
            z, weight = matching_eigenvectors_of_modes(number_of_modes, eigenvectors[0], eigenvectors_unorganized)
            NO.GRU_weight(weight)

            # Re-organizing the expanded wavenumbers
            wavenumbers[i + 1], eigenvectors[i + 1] = reorder_modes(z, wavenumbers_unorganized,
                                                                    eigenvectors_unorganized)

            # Saving the eigenvectors and wavenumbers
            np.save('GRU_eigen', eigenvectors)
            np.save('GRU_wvn', wavenumbers)

            # Removing the strained coordinate file
            subprocess.call(['rm', 'temp' + file_ending])

    # Setting a blank matrix to save the Gruneisen parameters in
    gruneisen = np.zeros((number_of_modes, 6))

    for i in range(6):
        # Calculating the Gruneisen parameters
        gruneisen[3:, i] = -(np.log(wavenumbers[i + 1, 3:]) - np.log(wavenumbers[0, 3:])) \
                           / inputs.gruneisen_matrix_strain_stepsize
    return gruneisen, wavenumbers[0]


def Get_Aniso_Gruneisen_Wavenumbers(Gruneisen, Wavenumber_Reference, ref_crystal_matrix, Coordinate_file, Program):
    # Setting a blank array for new wavenumbers
    new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(psf.Lattice_parameters(Program, Coordinate_file))
    applied_strain = Pr.RotationFree_StrainArray_from_CrystalMatrix(ref_crystal_matrix, new_crystal_matrix)

    wavenumbers = np.zeros(len(Wavenumber_Reference))

    for i in np.arange(3, len(wavenumbers), 1):
        # Computing the change to each wavenumber due to the current strain
        wavenumbers[i] = Wavenumber_Reference[i]*np.exp(-1.*np.sum(np.dot(applied_strain, Gruneisen[i])))
    return wavenumbers

##########################################
#     Organizing Wavenumbers for Gru     #
##########################################

def matching_eigenvectors_of_modes(number_of_modes, eigenvectors_1, eigenvectors_2):
    m = Munkres()
    weight = np.zeros((number_of_modes - 3, number_of_modes - 3))
    for i in range(3, number_of_modes):
        diff = np.dot(eigenvectors_1[i], eigenvectors_2[i]) \
               / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[i]))
        if np.absolute(diff) > 0.95:
            weight[i - 3] = 10000000.
            weight[i - 3, i - 3] = 1. - diff
        else:
            for j in range(3, number_of_modes):
                hold_weight = np.zeros(4)
                hold_weight[0] = 1 - np.dot(-1 * eigenvectors_1[i], eigenvectors_2[j]) \
                                     / (np.linalg.norm(-1 * eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[j]))
                hold_weight[1] = 1 - np.dot(eigenvectors_1[i], -1 * eigenvectors_2[j]) \
                                     / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(-1 * eigenvectors_2[j]))
                hold_weight[2] = 1 - np.dot(eigenvectors_1[i], eigenvectors_2[j]) \
                                     / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[j]))
                hold_weight[3] = 1 - np.dot(-1 * eigenvectors_1[i], -1 * eigenvectors_2[j]) \
                                     / (np.linalg.norm(-1 * eigenvectors_1[i])*np.linalg.norm(-1 * eigenvectors_2[j]))
                weight[i - 3, j - 3] = min(hold_weight)
    # Using the Hungarian algorithm to match wavenumbers
    Wgt = m.compute(weight)
    x, y = zip(*Wgt)
    z = np.column_stack((x, y))
    z = z + 3
    return z, weight[z[:, 0] - 3, z[:, 1] - 3]

def reorder_modes(z, wavenumbers, eigenvectors):
    wavenumbers_out = np.zeros(len(wavenumbers))
    eigenvectors_out = np.zeros((len(wavenumbers), len(wavenumbers)))
    for i in z:
        wavenumbers_out[i[0]] = wavenumbers[i[1]]
        eigenvectors_out[i[0]] = eigenvectors[i[1]]
    return wavenumbers_out, eigenvectors_out



# To do: Condense the Gru_organized_wavenumbers section (tinker and CP2K can be together
def Tinker_Gru_organized_wavenumbers(Expansion_type, Coordinate_file, Expanded_Coordinate_file, Parameter_file):
    number_of_modes = int(3*psf.atoms_count('Tinker', Coordinate_file))

    if os.path.isfile('GRU_eigen.npy') and os.path.isfile('GRU_wvn.npy'):
        wavenumbers = np.load('GRU_wvn.npy')
        eigenvectors = np.load('GRU_eigen.npy')
    else:
        if Expansion_type == 'Isotropic':
            wavenumbers = np.zeros((2, number_of_modes))
            eigenvectors = np.zeros((2, number_of_modes, number_of_modes))
            Expanded_Coordinate_file = [Expanded_Coordinate_file]
            NO.start_isoGru()

        elif Expansion_type == 'Anisotropic':
            wavenumbers = np.zeros((7, number_of_modes))
            eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
            NO.start_anisoGru()

    wavenumbers[0], eigenvectors[0] = psf.Wavenumber_and_Vectors('Tinker', Coordinate_file, Parameter_file)

    for i in range(1, len(Expanded_Coordinate_file) + 1):
        if not np.all(wavenumbers[i] == 0.):
            pass
        else:
            wavenumbers_unorganized, eigenvectors_unorganized = \
                psf.Wavenumber_and_Vectors('Tinker', Expanded_Coordinate_file[i - 1], Parameter_file)

            z, weight = matching_eigenvectors_of_modes(number_of_modes, eigenvectors[0], eigenvectors_unorganized)
            NO.GRU_weight(weight)

            # Re-organizing the expanded wavenumbers
            wavenumbers[i], eigenvectors[i] = reorder_modes(z, wavenumbers_unorganized, eigenvectors_unorganized)

        np.save('GRU_eigen', eigenvectors)
        np.save('GRU_wvn', wavenumbers)
    return wavenumbers

def CP2K_Gru_organized_wavenumbers(Expansion_type, Coordinate_file, Expanded_Coordinate_file, Parameter_file, Output):
    from munkres import Munkres, print_matrix
    m = Munkres()
    number_of_modes = 3*psf.atoms_count('CP2K', Coordinate_file)

    if Expansion_type == 'Isotropic':
        wavenumbers = np.zeros((2, number_of_modes))
        eigenvectors = np.zeros((2, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = psf.Wavenumber_and_Vectors('CP2K', Coordinate_file, Parameter_file)
        wavenumbers[1], eigenvectors[1] = psf.Wavenumber_and_Vectors('CP2K', Expanded_Coordinate_file, Parameter_file)
    elif Expansion_type == 'Anisotropic':
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = psf.Wavenumber_and_Vectors('CP2K', Coordinate_file, Parameter_file)
        for i in range(1,7):
            wavenumbers[i], eigenvectors[i] = psf.Wavenumber_and_Vectors('CP2K', Expanded_Coordinate_file[i-1],
                                                                         Parameter_file)


    # Weighting the modes matched together
    wavenumbers_out = np.zeros((len(wavenumbers[:, 0]), number_of_modes))
    wavenumbers_out[0] = wavenumbers[0]
    for k in range(1, len(wavenumbers[:, 0])):
        weight = np.zeros((number_of_modes - 3, number_of_modes - 3))
        for i in range(3, number_of_modes):
            diff = np.linalg.norm(np.dot(eigenvectors[0, i], eigenvectors[k, i]))/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, i]))
            if diff > 0.95:
                weight[i - 3] = 10000000.
                weight[i - 3, i - 3] = 1. - diff
            else:
                for j in range(3, number_of_modes):
                    weight[i - 3, j - 3] = 1 - np.linalg.norm(np.dot(eigenvectors[0, i], eigenvectors[k, j]))/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, j]))

        # Using the Hungarian algorithm to match wavenumbers
        Wgt = m.compute(weight)
        x,y = zip(*Wgt)
        z = np.column_stack((x,y))
        z = z +3

    # Re-organizing the expanded wavenumbers
        for i in z:
            wavenumbers_out[k, i[0]] = wavenumbers[k, i[1]]
    return wavenumbers_out

def QE_Gru_organized_wavenumbers(Expansion_type, Coordinate_file, Expanded_Coordinate_file, Parameter_file, Output):
    from munkres import Munkres, print_matrix
    m = Munkres()

    number_of_modes = 3*psf.atoms_count('QE', Coordinate_file)
    if Expansion_type == 'Isotropic':
        wavenumbers = np.zeros((2, number_of_modes))
        eigenvectors = np.zeros((2, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = psf.Wavenumber_and_Vectors('QE', Coordinate_file, Parameter_file)
        wavenumbers[1], eigenvectors[1] = psf.Wavenumber_and_Vectors('QE', Expanded_Coordinate_file, Parameter_file)

    elif Expansion_type == 'Anisotropic':
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = psf.Wavenumber_and_Vectors('QE', Coordinate_file, Parameter_file)
        for i in range(1,7):
            wavenumbers[i], eigenvectors[i] = psf.Wavenumber_and_Vectors('QE', Expanded_Coordinate_file[i-1],
                                                                         Parameter_file)


    # Weighting the modes matched together
    wavenumbers_out = np.zeros((len(wavenumbers[:, 0]), number_of_modes))
    wavenumbers_out[0] = wavenumbers[0]
    for k in range(1, len(wavenumbers[:, 0])):
        z, _ = matching_eigenvectors_of_modes(number_of_modes, eigenvectors[0], eigenvectors[k])
        # Re-organizing the expanded wavenumbers
        for i in z:
            wavenumbers_out[k, i[0]] = wavenumbers[k, i[1]]
    return wavenumbers_out

