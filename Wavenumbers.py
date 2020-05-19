#!/usr/bin/env python

from __future__ import print_function
import os
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
    r"""
    Given the inputs class and other keyword_parameters this will direct the computation of any wavenumber needs,
    including setup of the Gruneisen parameters.

    Parameters
    ----------
    inputs: Class
        Contains all user defined values and filled in with default program values

    Returns
    -------
    wavenumbers: Dict
        If Gruneisen parameters need to be calculated:

            * `'gruneisen'`: Gruneisen parameters of the input coordinate file.
            * `'wavenumber_reference'`: The reference wavenumbers [cm$^{-1}$] of the input coordinate file.
            * `'volume_reference'`: The reference volume [$\AA^{3}$] of the input coordinate file.

        If wavnumebers need to be calculated:

            * `'wavenumbers'`: wavenumbers [cm$^{-1}$] of the input coordinate file
    """

    # If the method does not require the Gruneisen parameter the wavenumbers are directly computed
    if inputs.method in ['HA', 'SiQ', 'GiQ', 'GaQ']:
        return psf.program_wavenumbers(keyword_parameters['Coordinate_file'], inputs.tinker_parameter_file,
                                       inputs.output, inputs.coordinate_file, inputs.program, inputs.method)

    # Wavenumber / Gruneisen calculation for isotropic expansion methods that use the Gruensein parameters
    elif inputs.method in ['SiQg', 'GiQg']:
        # Checking if the Gruneisen parameters and reference wavenumbers are being fed in
        if ('Gruneisen' in keyword_parameters) and ('Wavenumber_Reference' in keyword_parameters) and \
                ('Volume_Reference' in keyword_parameters) and ('New_Volume' in keyword_parameters):
            # Directly determining the new wavenumbers
            wavenumbers = Get_Iso_Gruneisen_Wavenumbers(keyword_parameters['Gruneisen'],
                                                        keyword_parameters['Wavenumber_Reference'],
                                                        keyword_parameters['Volume_Reference'],
                                                        keyword_parameters['New_Volume'])
            return wavenumbers

        else:
            # Checking if this is a re-run and if the Gruneisen parameters have already been saved
            if os.path.isfile(inputs.output + '_GRUwvn_' + inputs.method + '.npy') and \
                    os.path.isfile(inputs.output + '_GRU_' + inputs.method + '.npy'):
                # Loading the previously saved Gruneisen parameters and wavenumbers
                print("   ...Using Gruneisen parameters from: " + inputs.output + '_GRU_' + inputs.method + '.npy')
                gruneisen = np.load(inputs.output + '_GRU_' + inputs.method + '.npy')
                wavenumber_reference = np.load(inputs.output + '_GRUwvn_' + inputs.method + '.npy')
                volume_reference = Pr.Volume(Coordinate_file=inputs.coordinate_file, Program=inputs.program,
                                             Parameter_file=inputs.tinker_parameter_file)

            else:
                # Calculating the Gruneisen parameters and reference wavenumbers
                gruneisen, wavenumber_reference, volume_reference = Setup_Isotropic_Gruneisen(inputs)

                # Saving the Gruneisen parameters and wavenumbers
                print("   ... Saving reference wavenumbers and Gruneisen parameters to: " + inputs.output
                      + '_GRU_' + inputs.method + '.npy')
                np.save(inputs.output + '_GRU_' + inputs.method, gruneisen)
                np.save(inputs.output + '_GRUwvn_' + inputs.method, wavenumber_reference)
            return gruneisen, wavenumber_reference, volume_reference

    # Wavenumber / Gruneisen calculation for anisotropic expansion methods that use the Gruensein parameters
    elif inputs.method in ['GaQg']:
        # Checking if the Gruneisen parameters and reference wavenumbers are being fed in
        if ('Gruneisen' in keyword_parameters) and ('Wavenumber_Reference' in keyword_parameters) and \
                ('ref_crystal_matrix' in keyword_parameters):
            # Directly determining the new wavenumbers
            wavenumbers = Get_Aniso_Gruneisen_Wavenumbers(keyword_parameters['Gruneisen'],
                                                          keyword_parameters['Wavenumber_Reference'],
                                                          keyword_parameters['ref_crystal_matrix'],
                                                          keyword_parameters['Coordinate_file'], inputs.program)
            return wavenumbers

        else:
            # Checking if this is a re-run and if the Gruneisen parameters have already been saved
            if os.path.isfile(inputs.output + '_GRUwvn_' + inputs.method + '.npy') and \
                    os.path.isfile(inputs.output + '_GRU_' + inputs.method + '.npy'):
                # Loading the previously saved Gruneisen parameters and wavenumbers
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

    elif inputs.method in ['SaQply']:
        return psf.Wavenumber_and_Vectors(inputs.program, keyword_parameters['Coordinate_file'],
                                          inputs.tinker_parameter_file)


##########################################
#     Isotropic Gruneisen Parameter      #
##########################################
def Setup_Isotropic_Gruneisen(inputs):
    r"""
    Computes the isotropic Gruneisen parameters of the lattice minimum structure

    Parameters
    ----------
    inputs: Class
        Contains all user defined values and filled in with default program values

    Returns
    -------
    gruneisen: List[float]
        Isotropic Gruneisen parameters.
    wavenumbers_ref: List[float]
        Wavenumbers [cm$^{-1}$] of the lattice minimum structure.
    volume_ref: float
        Volume [$\AA^{3}$] of the lattice minimum structure.
    """

    # Change in lattice parameters for expanded structure
    dLattice_Parameters = Ex.Isotropic_Change_Lattice_Parameters((1 + inputs.gruneisen_volume_fraction_stepsize),
                                                                 inputs.program, inputs.coordinate_file)

    # Determining wavenumbers of lattice strucutre and expanded structure
    lattice_parameters = psf.Lattice_parameters(inputs.program, inputs.coordinate_file)

    # Expanding structure
    Ex.Expand_Structure(inputs, inputs.coordinate_file, 'lattice_parameters', 'temp',
                        dlattice_parameters=dLattice_Parameters)

    # File ending of the coordinate file
    file_ending = psf.assign_coordinate_file_ending(inputs.program) 

    # Computing the reference wavenumbers and eigenvectors
    wavenumbers_ref, eigenvectors_ref = psf.Wavenumber_and_Vectors(inputs.program, inputs.coordinate_file,
                                                                   inputs.tinker_parameter_file)
    number_of_modes = len(wavenumbers_ref)  # Number of vibrational modes

    # Computing the strained wavenumbers and eigenvectors
    wavenumbers_unorganized, eigenvectors_unorganized = psf.Wavenumber_and_Vectors(inputs.program, 'temp' + file_ending,
                                                                                   inputs.tinker_parameter_file)

    # Determining how the strained eigenvectors match up with the reference structure
    z, weight = matching_eigenvectors_of_modes(number_of_modes, eigenvectors_ref, eigenvectors_unorganized)

    # Outputing information for user output
    NO.start_isoGru()
    NO.GRU_weight(weight)

    # Re-organizing the expanded wavenumbers
    wavenumbers_organized, eigenvectors_organized = reorder_modes(z, wavenumbers_unorganized, eigenvectors_unorganized)

    # Calculating the volume of the lattice minimum and expanded structure
    volume_ref = Pr.Volume(lattice_parameters=lattice_parameters)
    volume_expand = volume_ref + inputs.gruneisen_volume_fraction_stepsize * volume_ref

    # Computing the Gruneisen parameters
    gruneisen = np.zeros(len(wavenumbers_ref))
    gruneisen[3:] = -(np.log(wavenumbers_ref[3:]) - np.log(wavenumbers_organized[3:])) / \
                    (np.log(volume_ref) - np.log(volume_expand))
    for x in range(0, len(gruneisen)):
        if wavenumbers_ref[x] == 0:
            gruneisen[x] = 0.0

    # Removing extra files created in process
    subprocess.call(['rm', 'temp' + file_ending])
    return gruneisen, wavenumbers_ref, volume_ref


def Get_Iso_Gruneisen_Wavenumbers(gruneisen, wavenumber_ref, volume_ref, new_volume):
    r"""
    Computes the isotropic wavenumbers using the Gruneisen parmaeters due to an isotropic volume change to the crystal
    lattice.

    Parameters
    ----------
    gruneisen: List[float]
        Isotropic Gruneisen parameters.
    wavenumbers_ref: List[float]
        Wavenumbers [cm$^{-1}$] of the lattice minimum structure.
    volume_ref: List[float]
        Volume [$\AA^{3}$] of the lattice minimum structure.
    new_volume: float
        Volume [$\AA^{3}$] of the isotropically expanded crystal that the wavenumbers need to be computed for.

    Returns
    -------
    wavenumbers: List[float]
        Wavenumbers [cm$^{-1}$] of the crystal structure at the new volume.
    """
    wavenumbers = np.diag(np.power(new_volume / volume_ref, -1 * gruneisen))
    wavenumbers = np.dot(wavenumber_ref, wavenumbers)
    return wavenumbers


##########################################
# Strain Ansotropic Gruneisen Parameter  #
##########################################

def Setup_Anisotropic_Gruneisen(inputs):
    r"""
    Computes the anisotropic Gruneisen parameters of the lattice minimum structure

    Parameters
    ----------
    inputs: Class
        Contains all user defined values and filled in with default program values

    Returns
    -------
    gruneisen: List[float]
        Anisotropic Gruneisen parameters.
    wavenumbers_ref: List[float]
        Wavenumbers [cm$^{-1}$] of the lattice minimum structure.
    """
    # Determining the file ending for the program used
    file_ending = psf.assign_coordinate_file_ending(inputs.program)

    # Determining if the anisotropic Gruneisen parameters have already been calculated
    # Typically, this take a long time. It may be advantageous to parallelize this in the future.
    re_run = False
    if os.path.isfile('GRU_wvn.npy') and os.path.isfile('GRU_eigen.npy'):
        # Loading in previously computed Gruneisen parameters
        wavenumbers = np.load('GRU_wvn.npy')
        eigenvectors = np.load('GRU_eigen.npy')
        re_run = True
    else:
        # Computing the reference wavenumbers and eigenvectors (the lattice minimum structure fed in)
        wavenumbers_ref, eigenvectors_ref = psf.Wavenumber_and_Vectors(inputs.program, inputs.coordinate_file,
                                                                       inputs.tinker_parameter_file)

        # Setting the number of vibrational modes
        number_of_modes = int(len(wavenumbers_ref))
        number_of_atoms = psf.atoms_count(inputs.program, inputs.coordinate_file)

        # Setting a place to store all the wavenumbers and eigenvalues for the Gruenisen parameters
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, 3 * number_of_atoms))
        wavenumbers[0] = wavenumbers_ref
        eigenvectors[0] = eigenvectors_ref

        # Out putting information for user output
        NO.start_anisoGru()

        # Saving that data computed thusfar in case the system crashes or times out
        np.save('GRU_eigen', eigenvectors)
        np.save('GRU_wvn', wavenumbers)

    # Cycling through the six principle six principal strains
    for i in range(6):
        if re_run and (wavenumbers[i + 1, 3] != 0.):
            # Skipping a given strain if the wavenumbers have previously been computed and loaded in
            pass
        else:
            # Expanding the lattice minimum structure in the direction of the i-th principal strain
            applied_strain = np.zeros(6)
            applied_strain[i] = inputs.gruneisen_matrix_strain_stepsize
            Ex.Expand_Structure(inputs, inputs.coordinate_file, 'strain', 'temp',
                                strain=Ex.strain_matrix(applied_strain),
                                crystal_matrix=Ex.Lattice_parameters_to_Crystal_matrix(psf.Lattice_parameters(
                                    inputs.program, inputs.coordinate_file)))

            # Computing the strained wavenumbers and eigenvectors
            wavenumbers_unorganized, eigenvectors_unorganized = psf.Wavenumber_and_Vectors(inputs.program,
                                                                                           'temp' + file_ending,
                                                                                           inputs.tinker_parameter_file)

            # Determining how the strained eigenvectors match up with the reference structure
            z, weight = matching_eigenvectors_of_modes(number_of_modes, eigenvectors[0], eigenvectors_unorganized)
            NO.GRU_weight(weight)  # Writing out the precision of matching the modes with one another

            # Re-organizing the expanded wavenumbers
            wavenumbers[i + 1], eigenvectors[i + 1] = reorder_modes(z, wavenumbers_unorganized,
                                                                    eigenvectors_unorganized)

            # Saving the eigenvectors and wavenumbers
            np.save('GRU_eigen', eigenvectors)
            np.save('GRU_wvn', wavenumbers)

            # Removing the strained coordinate file
            subprocess.call(['rm', 'temp' + file_ending])

    # Calculating the Gruneisen parameters due to the six principal strains
    gruneisen = np.zeros((number_of_modes, 6))
    for i in range(6):
        # Calculating the Gruneisen parameters
        gruneisen[3:, i] = -(np.log(wavenumbers[i + 1, 3:]) - np.log(wavenumbers[0, 3:])) \
                           / inputs.gruneisen_matrix_strain_stepsize
    return gruneisen, wavenumbers[0]


def Get_Aniso_Gruneisen_Wavenumbers(gruneisen, wavenumber_ref, crystal_matrix_ref, strained_coordinate_file, program):
    r"""
    Computes the isotropic wavenumbers using the Gruneisen parmaeters due to an isotropic volume change to the crystal
    lattice.

    Parameters
    ----------
    gruneisen: List[float]
        Anisotropic Gruneisen parameters.
    wavenumbers_ref: List[float]
        Wavenumbers [cm$^{-1}$] of the lattice minimum structure.
    crystal_matrix_ref: List[float]
        Lattice tensor [$\AA$] of the lattice minimum structure.
    strained_coordinate_file: float
        Coordinate file of the anisotropically expanded crystal that the wavenumbers need to be computed for.
    program: str
        Program being used.

    Returns
    -------
    wavenumbers: List[float]
        Wavenumbers [cm$^{-1}$] of the crystal structure in the strained crystal.
    """
    # Determining the strain placed on the expanded crystal relative to the reference matrix
    new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(psf.Lattice_parameters(program,
                                                                                        strained_coordinate_file))
    applied_strain = Pr.RotationFree_StrainArray_from_CrystalMatrix(crystal_matrix_ref, new_crystal_matrix)

    # Setting a blank array for new wavenumbers
    wavenumbers = np.zeros(len(wavenumber_ref))

    # Computing the change to each wavenumber due to the current strain
    for i in np.arange(3, len(wavenumbers), 1):
        wavenumbers[i] = wavenumber_ref[i] * np.exp(-1. * np.sum(np.dot(applied_strain, gruneisen[i])))
    return wavenumbers


##########################################
#     Organizing Wavenumbers for Gru     #
##########################################

def matching_eigenvectors_of_modes(number_of_modes, eigenvectors_1, eigenvectors_2):
    r"""
    Matches the eigenvectors of two crystal structures and returns new order of the modes and the weight to match each
    mode.

    Parameters
    ----------
    number_of_modes: int
        Number of vibrational modes in the crystal lattice
    eigenvectors_1: List[float]
        Eigenvectors of the reference crystal structure in the form of a 3N x 3N matrix, where N is the number of atoms
        in the crystal lattice.
    eigenvectors_2: List[float]
        Eigenvectors of the crystal structure that needs the modes reorganized in the form of a 3N x 3N matrix, where N
        is the number of atoms in the crystal lattice.

    Returns
    -------
    z: List[int]
        List to match the order of the eigenvectors_2 with eigenvectors_1.
    weight: List[float]
        Weight describing how well the reordered eigenvectors matched. A weight of 0 indicates a perfect match.

    """
    # Using the Munkres matching package
    m = Munkres()

    # Setting a matrix to save the weights to match all of the vibrational modes
    weight = np.zeros((number_of_modes - 3, number_of_modes - 3))

    # Cycling through all modes to apply a weight of how well they match
    for i in range(3, number_of_modes):
        # Using 1 - Cos(Theta) between the two eigenvectors as the weight
        diff = np.dot(eigenvectors_1[i], eigenvectors_2[i]) \
               / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[i]))

        if np.absolute(diff) > 0.95:
            # If Cos(Theta) is close to 1 for director order comparison of the eigenvectors than they are well matched and
            #    all other comparisons are set with a much higher weight
            weight[i - 3] = 10000000.
            weight[i - 3, i - 3] = 1. - diff
        else:
            # Otherwise the weight compared to all other eigenvectors are computed
            for j in range(3, number_of_modes):
                # Here we will check to see if the match is better if the direction of the eigenvalue is flipped
                #    (Eigenvectors are representative of vibrations and therefore the opposite director is also valid)
                hold_weight = np.zeros(4)
                hold_weight[0] = 1 - np.dot(-1 * eigenvectors_1[i], eigenvectors_2[j]) \
                                 / (np.linalg.norm(-1 * eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[j]))
                hold_weight[1] = 1 - np.dot(eigenvectors_1[i], -1 * eigenvectors_2[j]) \
                                 / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(-1 * eigenvectors_2[j]))
                hold_weight[2] = 1 - np.dot(eigenvectors_1[i], eigenvectors_2[j]) \
                                 / (np.linalg.norm(eigenvectors_1[i]) * np.linalg.norm(eigenvectors_2[j]))
                hold_weight[3] = 1 - np.dot(-1 * eigenvectors_1[i], -1 * eigenvectors_2[j]) \
                                 / (np.linalg.norm(-1 * eigenvectors_1[i])*np.linalg.norm(-1 * eigenvectors_2[j]))

                # The weight matching the two eigenvectors is the minimum value computed
                weight[i - 3, j - 3] = min(hold_weight)

    # Using the Hungarian algorithm to match wavenumbers
    Wgt = m.compute(weight)
    x, y = zip(*Wgt)
    z = np.column_stack((x, y))
    z = z + 3
    return z, weight[z[:, 0] - 3, z[:, 1] - 3]

def reorder_modes(z, wavenumbers, eigenvectors):
    r"""
    Uses the output (z) of matching_eigenvectors_of_modes to reorganize the wavenumbers and eigenvectors of
    eigenvectors_2

    Parameters
    ----------
    z: List[int]
        List to match the order of the eigenvectors_2 with eigenvectors_1 computed from matching_eigenvectors_of_modes.
    wavenumbers: List[float]
        Unorganized wavenumbers that need to be reordered based on z.
    eigenvectors: List[float]
        Eigenvectors of the crystal structure that needs the modes reorganized in the form of a 3N x 3N matrix, where N
        is the number of atoms in the crystal lattice. This should be eigenvectors_2 from
        matching_eigenvectors_of_modes.

    Returns
    -------
    wavenumbers_out: List[float]
        Organized wavenumbers.
    eigenvectors_out: List[float]
        Organized eigenvectors.
    """
    wavenumbers_out = np.zeros(len(wavenumbers))
    eigenvectors_out = np.zeros(eigenvectors.shape)
    for i in z:
        wavenumbers_out[i[0]] = wavenumbers[i[1]]
        eigenvectors_out[i[0]] = eigenvectors[i[1]]
    return wavenumbers_out, eigenvectors_out


