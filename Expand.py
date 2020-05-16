#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Numerical_Outputs as NO
import subprocess
import program_specific_functions as psf

##########################################
#                 Input                  #
##########################################
def Call_Expansion(inputs, purpose, coordinate_file, zeta=-1., **keyword_parameters):
    """
    :param Method: Harmonic approximation ('HA');
                   Stepwise Isotropic QHA ('SiQ');
                   Stepwise Isotropic QHA w/ Gruneisen Parameter ('SiQg');
                   Gradient Isotropic QHA ('GiQ');
                   Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
                   Gradient Anisotropic QHA ('GaQ');
                   Gradient Anisotropic QHA w/ Gruneisen Parameter ('GaQg');
    :param Purpose: 'expand' expanding a strucutre
                    'local_gradient' finding the local gradient of expansion for a strucutre
    :param Program: 'Tinker' Tinker molecular modeling
                    'Test' Test case
    :param Coordinate_file: file containing lattice parameters (and coordinates)
    :param molecules_in_coord: number of molecules in Coordinate_file
    :param keyword_parameters: Parameter_file, volume_fraction_chage, matrix_parameters_fraction_change, Temperature,
    Pressure, matrix_parameters_fraction_change, Statistical_mechanics, Gruneisen, 
    Wavenumber_Reference, Volume_Reference, Output
    
    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    volume_fraction_change: fractional volumetric step size (either for numerical gradient or amount to expand by).
    **matrix_parameters_fraction_change: fraction to change crystal matrix parameter by
    Temperature: in Kelvin
    Pressure: in atm
    Statistical_mechanics: 'Classical' Classical mechanics
                           'Quantum' Quantum mechanics
    Gruneisen: isotropic Gruneisen parameter
    Wavenumber_reference: reference wavenumbers for the Gruneisen parameter
    Volume_reference: reference volume for the Gruneisen parameter
    Output: string to name expanded coordinate file
    dcrystal_matrix: changes in the crystal matrix
    crystal_matrix_reference
    """
    # Expanded a strucutre being input
    if purpose == 'expand':
        if (inputs.method == 'GiQ') or (inputs.method == 'GiQg') or (inputs.method == 'SiQ') or \
                (inputs.method == 'SiQg') or (inputs.method == 'SaQply'):
            dlattice_parameters = Isotropic_Change_Lattice_Parameters(keyword_parameters['volume_fraction_change'],
                                                                      inputs.program, coordinate_file)
            
            Expand_Structure(inputs, coordinate_file, 'lattice_parameters', keyword_parameters['output_file'],
                             dlattice_parameters=dlattice_parameters)
        elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
            Expand_Structure(inputs, coordinate_file, 'crystal_matrix', keyword_parameters['output_file'],
                             dcrystal_matrix=keyword_parameters['dcrystal_matrix'])

    # Fining the local gradient of expansion for inputted strucutre
    elif purpose == 'local_gradient':
        if (inputs.method == 'GiQg') or (inputs.method == 'GiQ'):
            isotropic_local_gradient, wavenumbers, volume, left_minimum = \
                Isotropic_Local_Gradient(inputs, coordinate_file, keyword_parameters['Temperature'],
                                         keyword_parameters['LocGrd_dV'], Gruneisen=keyword_parameters['Gruneisen'],
                                         Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                         Volume_Reference=keyword_parameters['Volume_Reference'])
            return isotropic_local_gradient, wavenumbers, volume, left_minimum

        elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
            if inputs.anisotropic_type != '1D' or zeta != -1.:
                strain_local_gradient, wavenumbers, left_minimum = \
                    Anisotropic_Local_Gradient(inputs, coordinate_file, keyword_parameters['Temperature'],
                                               keyword_parameters['LocGrd_dC'], zeta=zeta,
                                               ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                               Gruneisen=keyword_parameters['Gruneisen'],
                                               Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])
            else:
                strain_local_gradient, wavenumbers, left_minimum = \
                    Anisotropic_Local_Gradient_1D(inputs, coordinate_file, keyword_parameters['Temperature'],
                                                  keyword_parameters['LocGrd_dLambda'],
                                                  keyword_parameters['dC_dLambda'],
                                                  ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                                  Gruneisen=keyword_parameters['Gruneisen'],
                                                  Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])
            return strain_local_gradient, wavenumbers, left_minimum

##########################################
#          Assistant Functions           #
##########################################


#NSA: Why is this here? This is just a duplicate of something already written
def crystal_coord_to_cartesian(coord, lattice_parameters):
    a = lattice_parameters[0]
    b = lattice_parameters[1]
    c = lattice_parameters[2]
    al = lattice_parameters[3]
    be = lattice_parameters[4]
    ga = lattice_parameters[5]
    omega = a*c*c*(1+2*np.cos(al)*np.cos(be)*np.cos(ga)-(np.cos(al))**2-(np.cos(ga))**2-(np.cos(be))**2)**0.5
    convert = np.zeros((3,3))
    convert[0,0] = a
    convert[0,1] = b*np.cos(ga)
    convert[1,1] = b*np.sin(ga)
    convert[0,2] = c*np.cos(be)
    convert[1,2] = c*(np.cos(al)-np.cos(be)*np.cos(ga))/(np.sin(ga))
    convert[2,2] = omega/(a*b*np.sin(ga))
    coordm = np.transpose(coord)
    cartcoord = np.transpose(np.matmul(convert,coordm))
    return cartcoord

def Lattice_parameters_to_Crystal_matrix(lattice_parameters):
    """
    This function takes the lattice parameters and returns the crystal matrix

    **Required Inputs
    lattice_parameters = crystal lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    """
    # Computing pieces of the crystal lattice matrix
    Vxx = lattice_parameters[0]
    Vxy = lattice_parameters[1]*np.cos(np.radians(lattice_parameters[5]))
    Vxz = lattice_parameters[2]*np.cos(np.radians(lattice_parameters[4]))

    Vyy = lattice_parameters[1]*np.sin(np.radians(lattice_parameters[5]))
    Vyz = lattice_parameters[2]*(np.cos(np.radians(lattice_parameters[3])) - np.cos(np.radians(lattice_parameters[4]))*np.cos(np.radians(lattice_parameters[5])))/np.sin(np.radians(lattice_parameters[5]))

    Vzz = np.sqrt(lattice_parameters[2]**2 - Vxz**2 - Vyz**2)
    # Combining the pieces of the matrix together
    if np.absolute(Vxy) < 1e-10:
        Vxy = 0.
    if np.absolute(Vxz) < 1e-10:
        Vxz = 0.
    if np.absolute(Vyz) < 1e-10:
        Vyz = 0.
    crystal_matrix = np.matrix([[Vxx, Vxy, Vxz], [0., Vyy, Vyz], [0., 0., Vzz]])
    return crystal_matrix

def crystal_matrix_to_lattice_parameters(crystal_matrix):
    """
    This function takes any strained crystal lattice matrix and return the lattice parameters
    
    **Required Inputs
    crystal_matrix = crystal lattice matrix ([[Vxx,Vxy,Vxz],
                                              [Vyx,Vyy,Vyz],
                                              [Vzx,Vzy,Vzz]])
    """
    # Computing lattice parameters
    a = np.linalg.norm(crystal_matrix[:, 0])
    b = np.linalg.norm(crystal_matrix[:, 1])
    c = np.linalg.norm(crystal_matrix[:, 2])

    gamma = np.arccos(np.dot(np.squeeze(np.asarray(crystal_matrix[:, 0])), np.squeeze(np.asarray(crystal_matrix[:, 1])))
                      / (a * b)) * 180. / np.pi
    alpha = np.arccos(np.dot(np.squeeze(np.asarray(crystal_matrix[:, 1])), np.squeeze(np.asarray(crystal_matrix[:, 2])))
                      / (b * c)) * 180. / np.pi
    beta = np.arccos(np.dot(np.squeeze(np.asarray(crystal_matrix[:, 2])), np.squeeze(np.asarray(crystal_matrix[:, 0])))
                     / (c * a)) * 180. / np.pi

    # Creating an array of lattice parameters
    lattice_parameters = np.array([a, b, c, alpha, beta, gamma])
    return lattice_parameters

def Isotropic_Change_Lattice_Parameters(volume_fraction_change, Program, Coordinate_file):
    """
    This function returns the change in lattice parameters for isotropic expansion/compression based off of a given 
    change in volume fraction

    **Required Inputs
    volume_fraction_change = Volume of the new desired strucutre over the volume of the previous structure
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    Coordinate_file = file containing lattice parameters of the previous strucuture
    """
    # Calling the lattice parameters
    lattice_parameters = psf.Lattice_parameters(Program, Coordinate_file)

    # Calculating the new isotropic lattice parameters
    dlattice_parameters = lattice_parameters*volume_fraction_change**(1/3.) - lattice_parameters

    # Setting changes in angles to zero because they do not change in isotropic expansion
    dlattice_parameters[3:] = 0.
    return dlattice_parameters

def strain_matrix(strain):
    strain_mat = np.zeros((3, 3))
    # Principal strains
    strain_mat[0, 0] = strain[0]
    strain_mat[1, 1] = strain[1]
    strain_mat[2, 2] = strain[2]
    # Shear strains must conserve angular momentum
    strain_mat[0, 1] = strain[3]
    strain_mat[1, 0] = strain[3]
    strain_mat[0, 2] = strain[4]
    strain_mat[2, 0] = strain[4]
    strain_mat[1, 2] = strain[5]
    strain_mat[2, 1] = strain[5]
    return strain_mat

def triangle_crystal_matrix_to_array(crystal_matrix):
    return np.array([crystal_matrix[0, 0], crystal_matrix[1, 1], crystal_matrix[2, 2],
                     crystal_matrix[0, 1], crystal_matrix[0, 2], crystal_matrix[1, 2]])

def array_to_triangle_crystal_matrix(array):
    return np.matrix([[array[0], array[3], array[4]],
                      [0.,       array[1], array[5]],
                      [0.,       0.,       array[2]]])

##########################################
#            General Expansion           #
##########################################
def Expand_Structure(inputs, Coordinate_file, Expansion_type, output_file_name, **keyword_parameters):
    """
    This function expands a coordinate file either based off of an inputted change in lattice vectors or crystal 
        lattice matrix

    **Required Inputs
    Coordinate_file = file containing lattice parameters (and coordinates)
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    Expansion_type = 'lattice parameters' expanding the structure by lattice parameters ([a,b,c,alpha,beta,gamma])
                   = 'crystal_matrix' expanding the strucutre by changes in the crystal matrix
    molecules_in_coord = number of molecules in the coordinate file
    Output = string to name expanded coordinate file
    
    **Optional Inputs
    dlattice_vectors = Changes in lattice parameters
    dcrystal_matrix = changes in crystal matrix
    Parameter_file = program specific file containingforce field parameters
    """
    if inputs.program == 'Test':
        coordinates = ''
        lattice_parameters = psf.Lattice_parameters(inputs.program, Coordinate_file)
        crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)
        if Expansion_type == 'lattice_parameters':
            lattice_parameters = lattice_parameters + keyword_parameters['dlattice_parameters']
        elif Expansion_type == 'strain':
            crystal_matrix = np.dot((np.identity(3) + keyword_parameters['strain']), crystal_matrix)
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)
        elif Expansion_type == 'crystal_matrix':
            crystal_matrix = crystal_matrix + keyword_parameters['dcrystal_matrix']
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)

    else:
        # Grabbing the lattice parameters and coordiantes
        lattice_parameters = psf.Lattice_parameters(inputs.program, Coordinate_file)
        coordinates = psf.return_coordinates(inputs.program, Coordinate_file, lattice_parameters)

        # Converting the lattice parameters to the lattice tensor
        crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)

        # Setting up the number of atoms per molecule as specified by the user
        if type(inputs.multi_nmols) == type(None):
            coordinate_center_of_mass = np.zeros((inputs.number_of_molecules, 3))
            atoms_per_molecule = np.zeros(inputs.number_of_molecules)
            atoms_per_molecule[:] = len(coordinates[:, 0])//inputs.number_of_molecules
        else:
            coordinate_center_of_mass = np.zeros((sum(inputs.multi_nmols), 3))
            atoms_per_molecule = np.zeros(sum(inputs.multi_nmols))

            placement = 0
            for i in range(len(inputs.multi_nmols)):
                atoms_per_molecule[placement: placement+inputs.multi_nmols[i]] = inputs.multi_atomspermol[i]
                placement += inputs.multi_nmols[i]

        # Determining the molecules center and removing that to expand the crystal
        for i in range(len(atoms_per_molecule)):
            lb = int(sum(atoms_per_molecule[:i]))
            #lb = i*atoms_per_molecule
            ub = int(sum(atoms_per_molecule[:i+1]))
            #ub = (i+1)*atoms_per_molecule
            coordinate_center_of_mass[i, :] = np.mean(coordinates[lb:ub], axis=0)
            coordinates[lb:ub] = np.subtract(coordinates[lb:ub], coordinate_center_of_mass[i, :])
        
        # Center of mass coordinates converted to fractional coordinates
        coordinate_center_of_mass = np.dot(np.linalg.inv(crystal_matrix), coordinate_center_of_mass.T).T

        # Computing the new crystal matrix
        if Expansion_type == 'lattice_parameters':
            lattice_parameters = lattice_parameters + keyword_parameters['dlattice_parameters']
            crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)
        elif Expansion_type == 'strain':
            crystal_matrix = np.dot((np.identity(3) + keyword_parameters['strain']), crystal_matrix)
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)
            crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)
        elif Expansion_type == 'crystal_matrix':
            crystal_matrix = crystal_matrix + keyword_parameters['dcrystal_matrix']
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)

        # Converting the center of mass to cartesian coordinates, but expanded to the new crystal matrix
        coordinate_center_of_mass = np.dot(crystal_matrix, coordinate_center_of_mass.T).T

        # Adding the atoms back to the expanded center of mass
        for i in range(len(atoms_per_molecule)):
            lb = int(sum(atoms_per_molecule[:i]))
            #lb = i*atoms_per_molecule
            ub = int(sum(atoms_per_molecule[:i+1]))
            #ub = (i+1)*atoms_per_molecule
            coordinates[lb:ub] = np.subtract(coordinates[lb:ub], -1*coordinate_center_of_mass[i, :])

    # Outputing the new coordinate file
    psf.output_new_coordinate_file(inputs.program, Coordinate_file, inputs.tinker_parameter_file, coordinates,
                                   lattice_parameters, output_file_name, inputs.min_rms_gradient)



###################################################
#       Local Gradient of Thermal  Expansion      #
###################################################
def Isotropic_Local_Gradient(inputs, coordinate_file, temperature, LocGrd_dV, **keyword_parameters):
    """
    This function calculates the local gradient of isotropic expansion for a given coordinate file
    
    :param Coordinate_file: file containing lattice parameters (and coordinates)
    :param Program: 'Tinker' Tinker molecular modeling
                    'Test' Test case
    :param Temperature: in Kelvin
    :param Pressure: in atm
    :param LocGrd_Vol_FracStep: fractional volumetric step size for numerical gradient 
    :param molecules_in_coord: number of molecules in Coordinate_file
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param Method: 'GiQ' Gradient isotropic QHA
                   'GiQg' Gradient isotropic QHA with Gruneisen Parameter
    :param keyword_parameters: Parameter_file, Gruneisen, Wavenumber_reference, Volume_reference
    
    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen: isotropic Gruneisen parameter
    Wavenumber_reference: reference wavenumbers for the Gruneisen parameter
    Volume_reference: reference volume for the Gruneisen parameter
    """
    # Assigning general names for expanded and compressed structures
    file_ending = psf.assign_coordinate_file_ending(inputs.program)

    coordinate_plus = 'plus' + file_ending
    coordinate_minus = 'minus' + file_ending

    # Determining the volume of Coordinate_file
    volume = Pr.Volume(Program=inputs.program, Coordinate_file=coordinate_file)

    # Determining the change in lattice parameter for isotropic expansion
    dlattice_parameters_p = Isotropic_Change_Lattice_Parameters((volume + LocGrd_dV) / volume, inputs.program,
                                                                coordinate_file)
    dlattice_parameters_m = Isotropic_Change_Lattice_Parameters((volume - LocGrd_dV) / volume, inputs.program,
                                                                coordinate_file)

    # Building the isotropically expanded and compressed strucutres
    Expand_Structure(inputs, coordinate_file, 'lattice_parameters', 'plus', dlattice_parameters=dlattice_parameters_p)
    Expand_Structure(inputs, coordinate_file, 'lattice_parameters', 'minus', dlattice_parameters=dlattice_parameters_m)

    # Calculating wavenumbers coordinate_file, plus.*, and minus.*
    if inputs.method == 'GiQ':
        wavenumbers = Wvn.Call_Wavenumbers(inputs, Coordinate_file=coordinate_file)
        wavenumbers_plus = Wvn.Call_Wavenumbers(inputs, Coordinate_file=coordinate_plus)
        wavenumbers_minus = Wvn.Call_Wavenumbers(inputs, Coordinate_file=coordinate_minus)
    else:
        wavenumbers = Wvn.Call_Wavenumbers(inputs, Gruneisen=keyword_parameters['Gruneisen'],
                                           Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                           Volume_Reference=keyword_parameters['Volume_Reference'],
                                           New_Volume=volume)
        wavenumbers_plus = Wvn.Call_Wavenumbers(inputs, Gruneisen=keyword_parameters['Gruneisen'],
                                                Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                Volume_Reference=keyword_parameters['Volume_Reference'],
                                                New_Volume=volume + LocGrd_dV)
        wavenumbers_minus = Wvn.Call_Wavenumbers(inputs, Gruneisen=keyword_parameters['Gruneisen'],
                                                 Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                 Volume_Reference=keyword_parameters['Volume_Reference'],
                                                 New_Volume=volume - LocGrd_dV)

    # If temperature is zero, we assume that the local gradient is the same at 0.001K
    if temperature == 0.:
        temperature = 1e-03

    # Calculating the numerator of the local gradient -dS/dV
    dS = (Pr.Vibrational_Entropy(temperature, wavenumbers_plus, inputs.statistical_mechanics) /
          inputs.number_of_molecules - Pr.Vibrational_Entropy(temperature, wavenumbers_minus,
                                                              inputs.statistical_mechanics) /
          inputs.number_of_molecules) / (2 * LocGrd_dV)

    # Calculating the denominator of the local gradient d**2G/dV**2
    ddG = (Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_plus, coordinate_plus,
                                inputs.statistical_mechanics, inputs.number_of_molecules,
                                Parameter_file=inputs.tinker_parameter_file)[0] -
           2 * Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers, coordinate_file,
                                    inputs.statistical_mechanics, inputs.number_of_molecules,
                                    Parameter_file=inputs.tinker_parameter_file)[0] +
           Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_minus, coordinate_minus,
                                inputs.statistical_mechanics, inputs.number_of_molecules,
                                Parameter_file=inputs.tinker_parameter_file)[0]) / (LocGrd_dV ** 2)

    # Computing the backward, central, and forward finite difference of dG/dV
    dG = np.zeros(3)
    dG[0] = (Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers, coordinate_file,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0] -
             Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_minus, coordinate_minus,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0]) / (LocGrd_dV)

    dG[1] = (Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_plus, coordinate_plus,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0] -
             Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_minus, coordinate_minus,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0]) / (2 * LocGrd_dV)

    dG[2] = (Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_plus, coordinate_plus,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0] -
             Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers, coordinate_file,
                                  inputs.statistical_mechanics, inputs.number_of_molecules,
                                  Parameter_file=inputs.tinker_parameter_file)[0]) / (LocGrd_dV)


    # Saving numerical outputs
    left_minimum = NO.iso_gradient(dG, ddG, dS, dS/ddG)

    # Removing excess files
    subprocess.call(['rm', coordinate_plus, coordinate_minus])
    return dS/ddG, wavenumbers, volume, left_minimum


def Anisotropic_Local_Gradient(inputs, coordinate_file, temperature, LocGrd_dC, zeta=-1., **keyword_parameters):
    """
    This function calculates the local gradient of anisotropic expansion for a given coordinate file

    :param Coordinate_file: file containing lattice parameters (and coordinates)
    :param Program: 'Tinker' Tinker molecular modeling
                    'Test' Test case
    :param Temperature: in Kelvin
    :param Pressure: in atm
    :param LocGrd_LatParam_FracStep: 
    :param molecules_in_coord: number of molecules in Coordinate_file
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param Method: 'GaQ' Gradient anisotropic QHA
                   'GaQg' Gradient anisotropic QHA with Gruneisen Parameter
    :param Hessian_number: 73 Hessians to calculate the complete anistropic gradient
                           25 for d**2G_dUdU only calculating the diagonals and off-diags. of the upper left 3x3 matrix
                           19 for d**2G_dUdU only calculating the upper left 3x3 matrix
                           13 for d**2G_dUdU only calculating the diagonals
                           7  for d**2G_dUdU only calculating the upper left 3x3 matrix daigonals
    :param keyword_parameters: Parameter_file, Gruneisen, Wavenumber_reference, Volume_reference

    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen: isotropic Gruneisen parameter
    Wavenumber_reference: reference wavenumbers for the Gruneisen parameter
    Crystal_matrix_Reference
    """
    min_numerical_crystal_matrix = 1.0e-7

    # Determining the file ending of the coordinate files
    file_ending = psf.assign_coordinate_file_ending(inputs.program)

    # Preparing the matrix with each entry as d**2G/(dC*dC)
    ddG_ddC = np.zeros((6, 6))

    # Preparing the vector with each entry as dS/dC and ddG/dCd(zeta or T)
    # dS/dC will be more accurate than ddG/dCdT
    # ddG/dCd(zeta) must be used for the EZP expansion
    dS_dC = np.zeros(6)
    ddG_dCdzeta = np.zeros(6)

    # A place to save potential and helmholtz vibrational energies to output
    U = np.zeros((6, 2))
    Av = np.zeros((6, 2))
    PV = np.zeros((6, 2))

    # dictionaries for saved intermediate data
    dG_dict = dict()
    wavenumbers_dict = dict()

    # Making array for dG/dC
    dG_dC = np.zeros((6, 3))

    if temperature == 0. and zeta == -1. and inputs.statistical_mechanics == 'Classical':
        # If temperature is zero, we assume that the local gradient is the same at 0.1K
        temperature = 0.0001
    elif temperature == 0. and zeta == -1. and inputs.statistical_mechanics == 'Quantum':
        # If temperature is zero, we assume that the local gradient is the same at 0.1K
        temperature = 0.1
    elif zeta == 0.:
        zeta = 0.0001

    # Modified anisotropic Local Gradient
    if (inputs.anisotropic_type == '6D') or ((inputs.anisotropic_type == '1D') and not os.path.isfile(inputs.output + '_dC_' + inputs.method + '.npy')):
        diag_limit = 6
        off_diag_limit = 6
    elif inputs.anisotropic_type == '3D':
        diag_limit = 3
        off_diag_limit = 3
    else:
        print("Aniso_LocGrad_Type = ", inputs.anisotropic_type, " is not a valid option.")
        sys.exit()

    # Retrieving the wavenumbers of the initial structure
    wavenumbers = Wvn.Call_Wavenumbers(inputs, Coordinate_file=coordinate_file,
                                       ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                       Gruneisen=keyword_parameters['Gruneisen'],
                                       Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])

    G_hold, U_0, Av_0 = Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers,
                                                   coordinate_file, inputs.statistical_mechanics,
                                                   inputs.number_of_molecules,
                                                   Parameter_file=inputs.tinker_parameter_file)

    PV_0 = G_hold - U_0 - Av_0
    dG_dict['0'] = U_0 + Av_0*np.absolute(zeta) + PV_0

    Av_0 *= zeta
    for i in range(diag_limit):
        # Calculating the diagonals of ddG_ddeta and the vector dS_deta

        # Setting the additional strain to the input structure for the current diagonal element
        if LocGrd_dC[i] < min_numerical_crystal_matrix:
            continue  # they remain zero, we don't bother to calculate them

        cm_array = np.zeros(6)
        cm_array[i] = LocGrd_dC[i]

        # Straining the input structure by the current diagonal
        delta1 = ('p', 'm')
        for d in delta1:
            if d == 'm':
                cm_factor = -1.0
                out_factor = 0
            else:
                cm_factor = 1.0
                out_factor = 1
            Expand_Structure(inputs, coordinate_file, 'crystal_matrix', d,
                             dcrystal_matrix=array_to_triangle_crystal_matrix(cm_factor * cm_array))

            wavenumbers_dict[d] = Wvn.Call_Wavenumbers(inputs, Coordinate_file=d + file_ending,
                                                       ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                                       Gruneisen=keyword_parameters['Gruneisen'],
                                                       Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])
            G_hold, U[i, out_factor], Av[i, out_factor] = \
                Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_dict[d], d + file_ending,
                                     inputs.statistical_mechanics, inputs.number_of_molecules,
                                     Parameter_file=inputs.tinker_parameter_file)
            PV[i, out_factor] = G_hold - U[i, out_factor] - Av[i, out_factor]
            dG_dict[d] = U[i, out_factor] + Av[i, out_factor]*np.absolute(zeta) + PV[i, out_factor]
#            dG_dict[d] = G_hold + Av[i, out_factor] * (np.absolute(zeta) - 1)

        # Computing the derivative of entropy as a funciton of strain using a finite difference approach
        if zeta == -1.:
            if temperature < 1.0 and inputs.statistical_mechanics == 'Quantum':
                # The limit of quantum entropy blows up at low T so computing -d^2 G / d C^2
                Gpp,_,_ = Pr.Gibbs_Free_Energy(temperature + 0.05, inputs.pressure, inputs.program, wavenumbers_dict['p'], 'p' + file_ending,
                                               inputs.statistical_mechanics, inputs.number_of_molecules,
                                               Parameter_file=inputs.tinker_parameter_file)
                Gpm,_,_ = Pr.Gibbs_Free_Energy(temperature - 0.05, inputs.pressure, inputs.program, wavenumbers_dict['p'], 'p' + file_ending,
                                               inputs.statistical_mechanics, inputs.number_of_molecules,
                                               Parameter_file=inputs.tinker_parameter_file)
                Gmp,_,_ = Pr.Gibbs_Free_Energy(temperature + 0.05, inputs.pressure, inputs.program, wavenumbers_dict['m'], 'm' + file_ending,
                                               inputs.statistical_mechanics, inputs.number_of_molecules,
                                               Parameter_file=inputs.tinker_parameter_file)
                Gmm,_,_ = Pr.Gibbs_Free_Energy(temperature - 0.05, inputs.pressure, inputs.program, wavenumbers_dict['m'], 'm' + file_ending,
                                               inputs.statistical_mechanics, inputs.number_of_molecules,
                                               Parameter_file=inputs.tinker_parameter_file)
                dS_dC[i] = - (Gpp - Gpm - Gpm + Gmm) / (4 * (temperature * 0.01) * LocGrd_dC[i])
            else: 
                Sp = Pr.Vibrational_Entropy(temperature, wavenumbers_dict['p'], inputs.statistical_mechanics) / \
                     inputs.number_of_molecules
                Sm = Pr.Vibrational_Entropy(temperature, wavenumbers_dict['m'], inputs.statistical_mechanics) / \
                     inputs.number_of_molecules
                dS_dC[i] = (Sp - Sm) / (2 * LocGrd_dC[i])
        else:
            dzeta = inputs.zeta_numerical_step * 0.05
            ddG_dCdzeta[i] = ((U[i, 1] + PV[i, 1] + Av[i, 1]*(zeta + dzeta)) 
                               - (U[i, 1] + PV[i, 1] + Av[i, 1]*(zeta - dzeta))
                               - (U[i, 0] + PV[i, 0] + Av[i, 0]*(zeta + dzeta))
                               + (U[i, 0] + PV[i, 0] + Av[i, 0]*(zeta - dzeta))) / (4  * dzeta * LocGrd_dC[i])

        # Calculating the finite difference of dG/deta for forward, central, and backwards
        dG_dC[i, 0] = (dG_dict['0'] - dG_dict['m']) / (LocGrd_dC[i])
        dG_dC[i, 1] = (dG_dict['p'] - dG_dict['m']) / (2 * LocGrd_dC[i])
        dG_dC[i, 2] = (dG_dict['p'] - dG_dict['0']) / (LocGrd_dC[i])

        # Computing the second derivative Gibbs free energy as a function of strain using a finite difference approach
        ddG_ddC[i, i] = (dG_dict['p'] - 2 * dG_dict['0'] + dG_dict['m']) / (LocGrd_dC[i] ** 2)

        # otherwise, these stay as zero
        if i < off_diag_limit:
            # Computing the off diagonals of d**2 G/ (deta*deta)
            for j in np.arange(i + 1, off_diag_limit):
                if LocGrd_dC[j] < min_numerical_crystal_matrix:
                    continue  # don't bother to calculate them distance changed is too small.
                # Setting the strain of the second element for the off diagonal
                cm_array_2 = np.zeros(6)
                cm_array_2[j] = LocGrd_dC[j]

                # Expanding the structure for a 4 different strains
                delta2 = list()
                for di in delta1:
                    for dj in delta1:
                        d2 = di+dj
                        delta2.append(d2) # create the list of 2D changes as length 2 strings
                        if dj == 'm':  # if the second dimension is a minus, dcrystal_matrix is negative
                            cm_factor = -1.0
                        else:
                            cm_factor = 1.0

                        Expand_Structure(inputs, di + file_ending, 'crystal_matrix', d2,
                                         dcrystal_matrix=array_to_triangle_crystal_matrix(cm_factor * cm_array_2))

                for d in delta2:
                    wavenumbers_dict[d] = \
                        Wvn.Call_Wavenumbers(inputs, Coordinate_file=d + file_ending,
                                             ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                             Gruneisen=keyword_parameters['Gruneisen'],
                                             Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])

                    G_hold, ignore, Av_hold = \
                        Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_dict[d],
                                             d + file_ending, inputs.statistical_mechanics, inputs.number_of_molecules,
                                             Parameter_file=inputs.tinker_parameter_file)
                    dG_dict[d] =  G_hold + Av_hold * (np.absolute(zeta) - 1)

                # Calculating the diagonal elements of d**2 G/(deta*deta)
                ddG_ddC[i, j] = (dG_dict['pp'] - dG_dict['pm'] - dG_dict['mp'] + dG_dict['mm']) / \
                                (4 * LocGrd_dC[i] * LocGrd_dC[j])

                # Making d**2 G/(deta*deta) symetric
                ddG_ddC[j, i] = ddG_ddC[i, j]

                # Removing excess files
                for d in delta2:
                    subprocess.call(['rm', d + file_ending])
        for d in delta1:
            subprocess.call(['rm', d + file_ending])

    # Calculating deta/dT for all strains
    if zeta == -1.:
        dC_dT = np.linalg.lstsq(ddG_ddC, dS_dC, rcond=None)[0] 
        # Saving numerical outputs
        NO.raw_energies(np.array([U_0]), np.array([Av_0]), U, Av)
        left_minimum = NO.aniso_gradient(dG_dC, ddG_ddC, dS_dC, dC_dT)
    else:
        dC_dT = np.linalg.lstsq(ddG_ddC, - ddG_dCdzeta, rcond=None)[0]
        print(dC_dT)
        left_minimum = False
    return dC_dT, wavenumbers, left_minimum


def Anisotropic_Local_Gradient_1D(inputs, coordinate_file, temperature, LocGrd_dLambda, dC_dLambda, 
                                  **keyword_parameters):
    # Determining the file ending of the coordinate files
    file_ending = psf.assign_coordinate_file_ending(inputs.program)

    if temperature == 0.:
        # If temperature is zero, we assume that the local gradient is the same at 0.1K
        temperature = 1e-5

    # Retrieving the wavenumbers of the initial structure
    wavenumber_keywords = { 'Gruneisen': keyword_parameters['Gruneisen'],
                            'Wavenumber_Reference': keyword_parameters['Wavenumber_Reference'],
                            'ref_crystal_matrix': keyword_parameters['ref_crystal_matrix']}
    wavenumbers = Wvn.Call_Wavenumbers(inputs, Coordinate_file=coordinate_file, **wavenumber_keywords)

    G = np.zeros(3)
    U = np.zeros(3) 
    Av = np.zeros(3)
    S = np.zeros(3)

    G[1], U[1], Av[1] = Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers,
                                             coordinate_file, inputs.statistical_mechanics, inputs.number_of_molecules,
                                             Parameter_file=inputs.tinker_parameter_file)

    crystal_matrix_array = LocGrd_dLambda * dC_dLambda
    # Straining the input structure by the current diagonal
    delta = ('p', 'm')
    for d in delta:
        if d == 'm':
            cm_factor = -1.0
            out_factor = 0
        else:
            cm_factor = 1.0
            out_factor = 2

        # Expand the crystal structure
        Expand_Structure(inputs, coordinate_file, 'crystal_matrix', d,
                         dcrystal_matrix=array_to_triangle_crystal_matrix(cm_factor * crystal_matrix_array))

        # Compute the wavenumbers
        wavenumbers_hold = Wvn.Call_Wavenumbers(inputs, Coordinate_file=d + file_ending, **wavenumber_keywords)

        # Compute the energy
        G[out_factor], U[out_factor], Av[out_factor] = \
            Pr.Gibbs_Free_Energy(temperature, inputs.pressure, inputs.program, wavenumbers_hold, d + file_ending,
                                 inputs.statistical_mechanics, inputs.number_of_molecules,
                                 Parameter_file=inputs.tinker_parameter_file)

        # Compute the entropy
        S[out_factor] = Pr.Vibrational_Entropy(temperature, wavenumbers_hold, inputs.statistical_mechanics) / \
                        inputs.number_of_molecules

        # Remove excess files
        subprocess.call(['rm', d + file_ending])

    # Computing the numerical gradient of entropy wrt lambda
    dS_dLambda = (S[2] - S[0]) / (2 * LocGrd_dLambda)

    # Calculating the finite difference of dG/deta for forward, central, and backwards
    dG_dLambda = np.zeros(3)
    dG_dLambda[0] = (G[1] - G[0]) / (LocGrd_dLambda)
    dG_dLambda[1] = (G[2] - G[0]) / (2 * LocGrd_dLambda)
    dG_dLambda[2] = (G[2] - G[1]) / (LocGrd_dLambda)

    # Computing the second derivative Gibbs free energy as a function of strain using a finite difference approach
    ddG_ddLambda = (G[2] - 2 * G[1] + G[0]) / (LocGrd_dLambda ** 2)

    # Computing the expansion of lambda
    dLambda = dS_dLambda / ddG_ddLambda

    # Determining if the system is still at a free energy minima wrt lambda
    left_minimum = NO.aniso_gradient_1D(dG_dLambda, ddG_ddLambda, dS_dLambda, dLambda)
    return dLambda, wavenumbers, left_minimum


