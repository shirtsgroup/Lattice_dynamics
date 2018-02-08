#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import itertools as it
import numpy as np
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Numerical_Outputs as NO
import subprocess
import fileinput

##########################################
#                 Input                  #
##########################################
def Call_Expansion(Method, Purpose, Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
                   **keyword_parameters):
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
    # If there is no parameter_file, one is just inputted so there is no errors
    if 'Parmaeter_file' in keyword_parameters:
        keyword_parameters['Parameter_file'] == ''

    # Expanded a strucutre being input
    if Purpose == 'expand':
        if (Method == 'GiQ') or (Method == 'GiQg') or (Method == 'SiQ') or (Method == 'SiQg'):
            dlattice_parameters = Isotropic_Change_Lattice_Parameters(keyword_parameters['volume_fraction_change'],
                                                                      Program, Coordinate_file)
            Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord,
                             keyword_parameters['Output'], min_RMS_gradient, 
                             Parameter_file=keyword_parameters['Parameter_file'],
                             dlattice_parameters=dlattice_parameters)
        elif (Method == 'GaQ') or (Method == 'GaQg'):
            Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord, keyword_parameters['Output'],
                             min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                             dcrystal_matrix=keyword_parameters['dcrystal_matrix'])

    # Fining the local gradient of expansion for inputted strucutre
    elif Purpose == 'local_gradient':
        if Method == 'GiQ':
            isotropic_local_gradient, wavenumbers, volume = \
                Isotropic_Local_Gradient(Coordinate_file, Program, keyword_parameters['Temperature'],
                                         keyword_parameters['Pressure'], keyword_parameters['volume_fraction_change'],
                                         molecules_in_coord,
                                         keyword_parameters['Statistical_mechanics'], Method, min_RMS_gradient,
                                         Parameter_file=keyword_parameters['Parameter_file'])
            return isotropic_local_gradient, wavenumbers, volume
        elif Method == 'GiQg':
            isotropic_local_gradient, wavenumbers, volume = \
                Isotropic_Local_Gradient(Coordinate_file, Program, keyword_parameters['Temperature'],
                                         keyword_parameters['Pressure'], keyword_parameters['volume_fraction_change'],
                                         molecules_in_coord,
                                         keyword_parameters['Statistical_mechanics'], Method, min_RMS_gradient,
                                         Parameter_file=keyword_parameters['Parameter_file'],
                                         Gruneisen=keyword_parameters['Gruneisen'],
                                         Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                         Volume_Reference=keyword_parameters['Volume_Reference'])
            return isotropic_local_gradient, wavenumbers, volume

        elif (Method == 'GaQ') or (Method == 'GaQg'):
            strain_local_gradient, wavenumbers = \
                Anisotropic_Local_Gradient(Coordinate_file, Program, keyword_parameters['Temperature'],
                                           keyword_parameters['Pressure'],
                                           keyword_parameters['LocGrd_NormStrain'],
                                           keyword_parameters['LocGrd_ShearStrain'], molecules_in_coord,
                                           keyword_parameters['Statistical_mechanics'], Method,
                                           keyword_parameters['Aniso_LocGrad_Type'], min_RMS_gradient,
                                           keyword_parameters['ref_crystal_matrix'],
                                           Parameter_file=keyword_parameters['Parameter_file'],
                                           Gruneisen=keyword_parameters['Gruneisen'],
                                           Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'])
            return strain_local_gradient, wavenumbers

##########################################
#       TINKER MOLECULAR MODELING        #
##########################################
def Return_Tinker_Coordinates(Coordinate_file):
    """
    This function opens a Tinker .xyz for a crystal and returns the 3x(number of atoms) matrix

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    """
    with open(Coordinate_file) as f:
        # Opening xyz coordinate file to expand
        coordinates = np.array(list(it.izip_longest(*[lines.split() for lines in f], fillvalue=' '))).T
    coordinates = coordinates[2:, 2:5].astype(float)
    return coordinates


def Output_Tinker_New_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output, min_RMS_gradient):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    Tinker .xyz crystal file
    The structure is also minimized

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """   
    Ouput_Tinker_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output)
    
#    subprocess.call(['minimize', Output + '.xyz', '-k', Parameter_file, str(min_RMS_gradient)], stdout=open(os.devnull, 'wb'))
#    subprocess.call(['mv', Output + '.xyz_2', Output + '.xyz'])
    Tinker_minimization(Parameter_file, Output + '.xyz', Output, min_RMS_gradient)


def Ouput_Tinker_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    Tinker .xyz crystal file

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """
    with open(Coordinate_file) as f:
        # Opening xyz coordinate file to expand
        coordinates_template = np.array(list(it.zip_longest(*[lines.split() for lines in f], fillvalue=' '))).T

    coordinates_template[2:, 2:5] = np.around(coordinates, decimals=8).astype('str')
    coordinates_template[1, :6] = lattice_parameters.astype(str)
    string_coordinates = ''
    for i in range(len(coordinates_template[:, 0])):
        for j in range(len(coordinates_template[i, :])):
            string_coordinates = string_coordinates + '    ' + coordinates_template[i, j]
        string_coordinates = string_coordinates + '\n'

    with open(Output + '.xyz', 'w') as file_out:
        file_out.write(string_coordinates)


def Tinker_minimization(Parameter_file, Coordinate_file, Output, min_RMS_gradient):
    with open('minimization.out', 'a') as myfile:
        myfile.write("======================== New Minimization ========================\n")
    run_min = True
    count = 0
    subprocess.call(['cp', Coordinate_file, 'Temp_min_0.xyz'])
    while run_min == True:
        # Running minimization
        output = subprocess.check_output(['minimize', 'Temp_min_' + str(count) + '.xyz', '-k', Parameter_file, 
                                          str(min_RMS_gradient)])
        count = count + 1

        # Moving minimized structure to the next temporary file
        subprocess.call(['mv', 'Temp_min_' + str(count - 1) + '.xyz_2', 'Temp_min_' + str(count) + '.xyz'])

        # Writing minimization to output file
        with open('minimization.out', 'a') as myfile:
            myfile.write(output)

        # Checking output of minimization
        output = output.split('\n')
#        if count == 1:
#            # Having a standard to prevent minimization into a new well
#            U_hold = float(output[-4].split()[-1])
#        U = float(output[-4].split()[-1])

        if output[-6] == ' LBFGS  --  Normal Termination due to SmallGrad':
            run_min = False
            subprocess.call(['mv', 'Temp_min_' + str(count) + '.xyz', Output + '.xyz'])
        elif count == 10:
            run_min = False
            subprocess.call(['mv', 'Temp_min_10.xyz', Output + '.xyz'])
            print("      Could not minimize strucutre to tolerance after 10 shake cycles")
#        elif np.absolute(U_hold- U) > 0.01:
#            run_min = False
#            subprocess.call(['mv', 'Temp_min_' + str(count - 1) + '.xyz', Output + '.xyz'])
#            print "      Structure minimized into a different well, stopping minimization"
        else:
            if count == 1:
                print("   ... Structure did not minimze to tolerance, shaking molecule and re-minimizing")
            coordinates = Return_Tinker_Coordinates('Temp_min_' + str(count) + '.xyz')
            coordinates = coordinates + np.random.randint(0, 10, size=(len(coordinates), 3)) * 10 ** (-7)
            lattice_parameters = Pr.Tinker_Lattice_Parameters('Temp_min_' + str(count) + '.xyz')
            Ouput_Tinker_Coordinate_File('Temp_min_' + str(count) + '.xyz', Parameter_file, coordinates, 
                                         lattice_parameters, 'Temp_min_' + str(count))
    for i in xrange(11):
        if os.path.isfile('Temp_min_' + str(i) + '.xyz'):
            subprocess.call(['rm', 'Temp_min_' + str(i) + '.xyz'])


##########################################
#                  TEST                  #
##########################################
def Return_Test_Coordinates():
    """
    This funciton returns coordinates for the test system
    Because there are no real coordiantes, it just returns a matrix of ones as a holder for the coordinates
    """
    coordinates = np.ones((1, 3))
    return coordinates


def Output_Test_New_Coordinate_File(lattice_parameters, Output):
    """
    This function saves the lattice parameters in a .npy file for the test Program
    """
    np.save(Output, lattice_parameters)
    
##########################################
#       CP2K MOLECULAR MODELING        #
##########################################
def Return_CP2K_Coordinates(Coordinate_file):
    """
    This function opens a Tinker .xyz for a crystal and returns the 3x(number of atoms) matrix

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    """
    with open(Coordinate_file) as f:
        # Opening xyz coordinate file to expand
        coordinates = np.array(list(it.zip_longest(*[lines.split() for lines in f], fillvalue=' '))).T
    coords = np.zeros((len(coordinates)-3,3))
    coords[:,:] = coordinates[2:-1, 3:6].astype(float)
    return coords


def Output_CP2K_New_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output, min_RMS_gradient):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    Tinker .xyz crystal file
    The structure is also minimized

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """   
    Ouput_Tinker_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output)
    CP2K_minimization(Parameter_file, Output + '.xyz', Output, min_RMS_gradient)

def Ouput_CP2K_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    .pdb crystal file

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """
    xstr = []
    numatoms = np.shape(coordinates)[0]
    for d in range(3):                  
        xstr.append("%9.3f" % (lattice_param[d]))
    for d in range(3,6):                  
        xstr.append("%7.2f" % (lattice_param[d]))
    with open(Output + '.pdb', 'w') as file_out:
        file_out.write('REMARK'+ '\n')
        file_out.write('CRYST1    '+str(lattice_param[0])+xstr[1]+xstr[2]+xstr[3]+xstr[4]+xstr[5]+'\n')
        for x in range(numatoms):

            if (x+1) % 2 == 1:
                ty = 'C'
            else:
                ty = 'H'
            line =  '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'.format('ATOM',x+1, ty,'','','','','',coordinates[x,0],coordinates[x,0],coordinates[x,0], 0.0,0.0,ty,'')
            file_out.write(line+'\n')
        file_out.write('END')


def CP2K_minimization(Parameter_file, Coordinate_file, Output, min_RMS_gradient):
    new_coord_file = 'Temp_min_0.pdb'
    subprocess.call(['cp', Coordinate_file, new_coord_file])
    new_param_file = 'Temp_min_0.inp'
    subprocess.call(['cp', parameter_file, new_param_file])
    #lattice_param = Pr.CP2K_Lattice_Parameters(Coordinate_file)
    volume = '{:4.8s}'.format(str(lattice_param[0]*lattice_param[2]*lattice_param[1]))
    
    tempnamestr = 's/TEMPORARY/BNZ_VOL_'+volume+'/g'
    subprocess.call(["sed", "-i", "-e",  tempnamestr, new_param_file])
    
    tempcoordstr = 's/TEMPCOORDFILE/'+new_coord_file+'/g'
    subprocess.call(["sed", "-i", "-e",  tempcoordstr, new_param_file])
    
    tempalphastr = 's/ALPHA/{:4.5s}/g'.format(str(lattice_param[3]))
    subprocess.call(["sed", "-i", "-e",  tempalphastr, new_param_file])
    
    tempbetastr = 's/BETA/{:4.5s}/g'.format(str(lattice_param[4]))
    subprocess.call(["sed", "-i", "-e",  tempbetastr, new_param_file])

    tempgammastr = 's/GAMMA/{:4.5s}/g'.format(str(lattice_param[5]))
    subprocess.call(["sed", "-i", "-e",  tempgammastr, new_param_file])
    
    tempaaastr = 's/AAA/{:4.5s}/g'.format(str(lattice_param[0]))
    subprocess.call(["sed", "-i", "-e",  tempaaastr, new_param_file])
    
    tempbbbstr = 's/BBB/{:4.5s}/g'.format(str(lattice_param[1]))
    subprocess.call(["sed", "-i", "-e",  tempbbbstr, new_param_file])

    tempcccstr = 's/CCC/{:4.5s}/g'.format(str(lattice_param[2]))
    subprocess.call(["sed", "-i", "-e",  tempcccstr, new_param_file])

    temprunstr = 's/TEMPRUNTYPE/GEO_OPT/g'
    subprocess.call(["sed", "-i", "-e",  temprunstr, new_param_file])


    subprocess.call(['mpirun', '-np', '112', 'cp2k.popt', '-i', INPUTFILE, '>', OUTPUTNAME])


##########################################
#          Assistant Functions           #
##########################################
def assign_file_ending(program):
    if program == 'Tinker':
        return '.xyz'
    elif program == 'Test':
        return '.npy'
    else:
        print('ERROR: Program is not supported!')
        sys.exit()


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
    if Program == 'Tinker':
        lattice_parameters = Pr.Tinker_Lattice_Parameters(Coordinate_file)
    elif Program == 'Test':
        lattice_parameters = Pr.Test_Lattice_Parameters(Coordinate_file)

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
def Expand_Structure(Coordinate_file, Program, Expansion_type, molecules_in_coord, Output, min_RMS_gradient,
                     **keyword_parameters):
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
    if Program == 'Test':
        lattice_parameters = Pr.Test_Lattice_Parameters(Coordinate_file)
        crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)
        if Expansion_type == 'lattice_parameters':
            lattice_parameters = lattice_parameters + keyword_parameters['dlattice_parameters']
        elif Expansion_type == 'strain':
            crystal_matrix = np.dot((np.identity(3) + keyword_parameters['strain']), crystal_matrix)
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)
        elif Expansion_type == 'crystal_matrix':
            crystal_matrix = crystal_matrix + keyword_parameters['dcrystal_matrix']
            lattice_parameters = crystal_matrix_to_lattice_parameters(crystal_matrix)
        Output_Test_New_Coordinate_File(lattice_parameters, Output)

    else:
        if Program == 'Tinker':
            coordinates = Return_Tinker_Coordinates(Coordinate_file)
            lattice_parameters = Pr.Tinker_Lattice_Parameters(Coordinate_file)

        crystal_matrix = Lattice_parameters_to_Crystal_matrix(lattice_parameters)

        coordinate_center_of_mass = np.zeros((molecules_in_coord, 3))
        atoms_per_molecule = len(coordinates[:, 0])/molecules_in_coord

        for i in range(int(molecules_in_coord)):
            coordinate_center_of_mass[i, :] = np.mean(coordinates[i*atoms_per_molecule:(i+1)*atoms_per_molecule],
                                                      axis=0)
            coordinates[i*atoms_per_molecule:(i+1)*atoms_per_molecule] = \
                np.subtract(coordinates[i*atoms_per_molecule:(i+1)*atoms_per_molecule], coordinate_center_of_mass[i, :])

        # Center of mass coordinates converted to fractional coordinates
        coordinate_center_of_mass = np.dot(np.linalg.inv(crystal_matrix), coordinate_center_of_mass.T).T
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
        coordinate_center_of_mass = np.dot(crystal_matrix, coordinate_center_of_mass.T).T
        for i in range(int(molecules_in_coord)):
            coordinates[i*atoms_per_molecule:(i+1)*atoms_per_molecule] = \
                np.subtract(coordinates[i*atoms_per_molecule:(i+1)*atoms_per_molecule],
                            -1*coordinate_center_of_mass[i, :])

        if Program == 'Tinker':
            Output_Tinker_New_Coordinate_File(Coordinate_file, keyword_parameters['Parameter_file'], coordinates,
                                              lattice_parameters, Output, min_RMS_gradient)

###################################################
#       Local Gradient of Thermal  Expansion      #
###################################################
def Isotropic_Local_Gradient(Coordinate_file, Program, Temperature, Pressure, LocGrd_Vol_FracStep,
                             molecules_in_coord, Statistical_mechanics, Method, min_RMS_gradient, **keyword_parameters):
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
    if Program == 'Tinker':
        coordinate_plus = 'plus.xyz'
        coordinate_minus = 'minus.xyz'
    elif Program == 'Test':
        coordinate_plus = 'plus.npy'
        coordinate_minus = 'minus.npy'
        keyword_parameters['Parameter_file'] = ''

    # Determining the change in lattice parameter for isotropic expansion
    dlattice_parameters = Isotropic_Change_Lattice_Parameters(1 + LocGrd_Vol_FracStep, Program, Coordinate_file)

    # Building the isotropically expanded and compressed strucutres
    Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, 'plus', min_RMS_gradient,
                     dlattice_parameters=dlattice_parameters, Parameter_file=keyword_parameters['Parameter_file'])
    Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, 'minus', min_RMS_gradient,
                     dlattice_parameters=-1*dlattice_parameters, Parameter_file=keyword_parameters['Parameter_file'])

    # Determining the volume of Coordinate_file
    volume = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)
    # Calculating wavenumbers coordinate_file, plus.*, and minus.*
    if Method == 'GiQ':
        wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file=Coordinate_file,
                                           Parameter_file=keyword_parameters['Parameter_file'],
                                           Program=Program)
        wavenumbers_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file=coordinate_plus,
                                                Parameter_file=keyword_parameters['Parameter_file'],
                                                Program=Program)
        wavenumbers_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file=coordinate_minus,
                                                 Parameter_file=keyword_parameters['Parameter_file'],
                                                 Program=Program)
    else:
        wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=keyword_parameters['Gruneisen'],
                                           Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                           Volume_Reference=keyword_parameters['Volume_Reference'],
                                           New_Volume=volume)
        wavenumbers_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=keyword_parameters['Gruneisen'],
                                                Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                Volume_Reference=keyword_parameters['Volume_Reference'],
                                                New_Volume=volume + volume*LocGrd_Vol_FracStep)
        wavenumbers_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=keyword_parameters['Gruneisen'],
                                                 Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                 Volume_Reference=keyword_parameters['Volume_Reference'],
                                                 New_Volume=volume - volume*LocGrd_Vol_FracStep)

    # If temperature is zero, we assume that the local gradient is the same at 0.1K
    if Temperature == 0.:
        Temperature = 0.1

    # Calculating the numerator of the local gradient -dS/dV
    dS = (Pr.Vibrational_Entropy(Temperature, wavenumbers_plus, Statistical_mechanics)/molecules_in_coord -
                  Pr.Vibrational_Entropy(Temperature, wavenumbers_minus, Statistical_mechanics)/molecules_in_coord) / \
                 (2*LocGrd_Vol_FracStep*volume)

    # Calculating the denominator of the local gradient d**2G/dV**2
    ddG = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, coordinate_plus,
                                Statistical_mechanics, molecules_in_coord,
                                Parameter_file=keyword_parameters['Parameter_file']) -
           2*Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file']) +
           Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, coordinate_minus,
                                Statistical_mechanics, molecules_in_coord,
                                Parameter_file=keyword_parameters['Parameter_file'])) / \
          ((volume*LocGrd_Vol_FracStep)**2)

    # Computing the backward, central, and forward finite difference of dG/dV
    dG = np.zeros(3)
    dG[0] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file']) -
             Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, coordinate_minus,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file'])) / \
            (volume*LocGrd_Vol_FracStep)

    dG[1] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, coordinate_plus,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file']) -
             Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, coordinate_minus,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file'])) / \
            (2*volume*LocGrd_Vol_FracStep)

    dG[2] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, coordinate_plus,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file']) -
             Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                  Statistical_mechanics, molecules_in_coord,
                                  Parameter_file=keyword_parameters['Parameter_file'])) / \
            (volume*LocGrd_Vol_FracStep)


    # Saving numerical outputs
    NO.iso_gradient(dG, ddG, dS, dS/ddG)

    # Removing excess files
    os.system('rm '+coordinate_plus+' '+coordinate_minus)
    return dS/ddG, wavenumbers, volume


def Anisotropic_Local_Gradient(Coordinate_file, Program, Temperature, Pressure, LocGrd_NormStrain,
                               LocGrd_ShearStrain, molecules_in_coord, Statistical_mechanics, Method, Hessian_number,
                               min_RMS_gradient, ref_crystal_matrix, **keyword_parameters):
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
    # Setting up an array for changes in the crystal matrix
    crystal_matrix = Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file))
    numerical_crystal_matrix_step = np.absolute(triangle_crystal_matrix_to_array(crystal_matrix))
    numerical_crystal_matrix_step[:3] = LocGrd_NormStrain * numerical_crystal_matrix_step[:3]
    numerical_crystal_matrix_step[3:] = LocGrd_ShearStrain * numerical_crystal_matrix_step[3:]
    for i in range(3,6):
        if numerical_crystal_matrix_step[i] < 0.005:
            numerical_crystal_matrix_step[i] = 0.005
    # Determining the file ending of the coordinate files
    file_ending = assign_file_ending(Program)

    # Preparing the matrix with each entry as d**2G/(dC*dC)
    ddG_ddC = np.zeros((6, 6))

    # Preparing the vector with each entry as d*S/dC
    dS_dC = np.zeros(6)

    # Modified anisotropic Local Gradient
# This needs to be re-thought
    if Hessian_number == 73:
        diag_limit = 6
        off_diag_limit = 6
    elif Hessian_number == 25:
        diag_limit = 6
        off_diag_limit = 3
    elif Hessian_number == 19:
        diag_limit = 3
        off_diag_limit = 3
    elif Hessian_number == 13:
        diag_limit = 6
        off_diag_limit = 0
    elif Hessian_number == 7:
        diag_limit = 3
        off_diag_limit = 0

    # Retrieving the wavenumbers of the initial structure
### Start Change
    wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file=Coordinate_file,
                                       Parameter_file=keyword_parameters['Parameter_file'],
                                       Gruneisen=keyword_parameters['Gruneisen'],
                                       Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                       ref_crystal_matrix=ref_crystal_matrix, Program=Program)
#    if Method == 'GaQ':
#        if Program == 'Tinker':
#            wavenumbers = Wvn.Tinker_Wavenumber(Coordinate_file, Parameter_file=keyword_parameters['Parameter_file'])
#        elif Program == 'Test':
#            wavenumbers = Wvn.Test_Wavenumber(Coordinate_file, Applied_strain)
#    elif Method == 'GaQg':
#        wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=keyword_parameters['Gruneisen'],
#                                           Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
#                                           Applied_strain=Applied_strain)
### End Change

    # Making array for dG/dC
    dG_dC = np.zeros((6, 3))

    if Temperature == 0.:
        # If temperature is zero, we assume that the local gradient is the same at 0.1K
        Temperature = 0.1

    for i in range(diag_limit):
        # Calculating the diagonals of ddG_ddeta and the vector dS_deta

        # Setting the additional strain to the input strucutre for the current diagonal element
        cm_array = np.zeros(6)
        cm_array[i] = numerical_crystal_matrix_step[i]

        # Straining the input structure by the current diagonal
        Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord, 'p', min_RMS_gradient,
                         dcrystal_matrix=array_to_triangle_crystal_matrix(cm_array),
                         Parameter_file=keyword_parameters['Parameter_file'])

        Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord, 'm', min_RMS_gradient,
                         dcrystal_matrix=array_to_triangle_crystal_matrix(-1.*cm_array),
                         Parameter_file=keyword_parameters['Parameter_file'])

### Start Change
        wavenumbers_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file='p' + file_ending,
                                                Program=Program,
                                                Parameter_file=keyword_parameters['Parameter_file'],
                                                Gruneisen=keyword_parameters['Gruneisen'],
                                                Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                ref_crystal_matrix=ref_crystal_matrix)
        wavenumbers_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file='m' + file_ending,
                                                 Program=Program,
                                                 Parameter_file=keyword_parameters['Parameter_file'],
                                                 Gruneisen=keyword_parameters['Gruneisen'],
                                                 Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                 ref_crystal_matrix=ref_crystal_matrix)

#        if Method == 'GaQ':
#            # Computing the wavenumbers for the strained strucutres with the mass-weighted Hessian
#            wavenumbers_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file='p' + file_ending,
#                                                    Program=Program,
#                                                    Parameter_file=keyword_parameters['Parameter_file'],
#                                                    Applied_strain=Applied_strain + strain_array)
#            wavenumbers_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Coordinate_file='m' + file_ending,
#                                                     Program=Program,
#                                                     Parameter_file=keyword_parameters['Parameter_file'],
#                                                     Applied_strain=Applied_strain - strain_array)
#        elif Method == 'GaQg':
#            # Computing the wavenumbers for the strained strucutres with the Gruneisen parameter
#            wavenumbers_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=keyword_parameters['Gruneisen'],
#                                                    Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
#                                                    Applied_strain=Applied_strain + strain_array)
#            wavenumbers_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                     Gruneisen=keyword_parameters['Gruneisen'],
#                                                     Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
#                                                     Applied_strain=Applied_strain - strain_array)
#### End change

        # Computing the derivative of entropy as a funciton of strain using a finite difference approach
        dS_dC[i] = (Pr.Vibrational_Entropy(Temperature, wavenumbers_plus, Statistical_mechanics) / molecules_in_coord
                      - Pr.Vibrational_Entropy(Temperature, wavenumbers_minus, Statistical_mechanics) /
                      molecules_in_coord) / (2 * numerical_crystal_matrix_step[i])

        # Computing the second derivative Gibbs free energy as a funciton of strain using a finite difference approach
        ddG_ddC[i, i] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, 'p' + file_ending,
                                              Statistical_mechanics, molecules_in_coord,
                                              Parameter_file=keyword_parameters['Parameter_file']) -
                         2 * Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                                  Statistical_mechanics, molecules_in_coord,
                                                  Parameter_file=keyword_parameters['Parameter_file']) +
                         Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, 'm' + file_ending,
                                               Statistical_mechanics, molecules_in_coord,
                                               Parameter_file=keyword_parameters['Parameter_file'])) / \
                        (numerical_crystal_matrix_step[i] ** 2)

        # Calculating the finite difference of dG/deta for forward, central, and backwards
        dG_dC[i, 0] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file']) -
                       Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, 'm' + file_ending,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file'])) / \
                      (numerical_crystal_matrix_step[i])

        dG_dC[i, 1] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, 'p' + file_ending,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file']) -
                       Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus, 'm' + file_ending,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file'])) / \
                      (2*numerical_crystal_matrix_step[i])

        dG_dC[i, 2] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus, 'p' + file_ending,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file']) -
                       Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file,
                                            Statistical_mechanics, molecules_in_coord,
                                            Parameter_file=keyword_parameters['Parameter_file'])) / \
                      (numerical_crystal_matrix_step[i])



        if i < off_diag_limit:
            # Computing the off diagonals of d**2 G/ (deta*deta)
            for j in np.arange(i + 1, off_diag_limit):
                # Setting the strain of the second element for the off diagonal
                cm_array_2 = np.zeros(6)
                cm_array_2[j] = numerical_crystal_matrix_step[j]

                # Expanding the strucutres for a varying number of strains
                Expand_Structure('p' + file_ending, Program, 'crystal_matrix', molecules_in_coord, 'pp',
                                 min_RMS_gradient, dcrystal_matrix=array_to_triangle_crystal_matrix(cm_array_2),
                                 Parameter_file=keyword_parameters['Parameter_file'])
                Expand_Structure('p' + file_ending, Program, 'crystal_matrix', molecules_in_coord, 'pm',
                                 min_RMS_gradient, dcrystal_matrix=array_to_triangle_crystal_matrix(-1.*cm_array_2),
                                 Parameter_file=keyword_parameters['Parameter_file'])
                Expand_Structure('m' + file_ending, Program, 'crystal_matrix', molecules_in_coord, 'mm',
                                 min_RMS_gradient, dcrystal_matrix=array_to_triangle_crystal_matrix(-1.*cm_array_2),
                                 Parameter_file=keyword_parameters['Parameter_file'])
                Expand_Structure('m' + file_ending, Program, 'crystal_matrix', molecules_in_coord, 'mp',
                                 min_RMS_gradient, dcrystal_matrix=array_to_triangle_crystal_matrix(cm_array_2),
                                 Parameter_file=keyword_parameters['Parameter_file'])

### Start Change
                wavenumbers_plus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
                                                             Coordinate_file='pp' + file_ending, Program=Program,
                                                             Parameter_file=keyword_parameters['Parameter_file'],
                                                             Gruneisen=keyword_parameters['Gruneisen'],
                                                             Wavenumber_Reference=
                                                             keyword_parameters['Wavenumber_Reference'],
                                                             ref_crystal_matrix=ref_crystal_matrix)
                wavenumbers_plus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
                                                              Coordinate_file='pm' + file_ending, Program=Program,
                                                              Parameter_file=keyword_parameters['Parameter_file'],
                                                              Gruneisen=keyword_parameters['Gruneisen'],
                                                              Wavenumber_Reference=
                                                              keyword_parameters['Wavenumber_Reference'],
                                                              ref_crystal_matrix=ref_crystal_matrix)
                wavenumbers_minus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
                                                               Coordinate_file='mm' + file_ending, Program=Program,
                                                               Parameter_file=keyword_parameters['Parameter_file'],
                                                               Gruneisen=keyword_parameters['Gruneisen'],
                                                               Wavenumber_Reference=
                                                               keyword_parameters['Wavenumber_Reference'],
                                                               ref_crystal_matrix=ref_crystal_matrix)
                wavenumbers_minus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
                                                              Coordinate_file='mp' + file_ending, Program=Program,
                                                              Parameter_file=keyword_parameters['Parameter_file'],
                                                              Gruneisen=keyword_parameters['Gruneisen'],
                                                              Wavenumber_Reference=
                                                              keyword_parameters['Wavenumber_Reference'],
                                                              ref_crystal_matrix=ref_crystal_matrix)

#                if Method == 'GaQ':
#                    # Computing the wavenumbers for the strained strucutres with the mass-weighted Hessian
#                    wavenumbers_plus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                 Coordinate_file='pp' + file_ending, Program=Program,
#                                                                 Parameter_file=keyword_parameters['Parameter_file'],
#                                                                 Applied_strain=Applied_strain + strain_array +
#                                                                                strain_array_2)
#                    wavenumbers_plus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                  Coordinate_file='pm' + file_ending, Program=Program,
#                                                                  Parameter_file=keyword_parameters['Parameter_file'],
#                                                                  Applied_strain=Applied_strain + strain_array -
#                                                                                 strain_array_2)
#                    wavenumbers_minus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                   Coordinate_file='mm' + file_ending, Program=Program,
#                                                                   Parameter_file=keyword_parameters['Parameter_file'],
#                                                                   Applied_strain=Applied_strain - strain_array -
#                                                                                  strain_array_2)
#                    wavenumbers_minus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                  Coordinate_file='mp' + file_ending, Program=Program,
#                                                                  Parameter_file=keyword_parameters['Parameter_file'],
#                                                                  Applied_strain=Applied_strain - strain_array +
#                                                                                 strain_array_2)
#                elif Method == 'GaQg':
#                    # Computing the wavenumbers for the strained strucutres with the Gruneisen parameter
#                    wavenumbers_plus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                 Gruneisen=keyword_parameters['Gruneisen'],
#                                                                 Wavenumber_Reference=
#                                                                 keyword_parameters['Wavenumber_Reference'],
#                                                                 Applied_strain=Applied_strain + strain_array +
#                                                                                strain_array_2)
#                    wavenumbers_plus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                  Gruneisen=keyword_parameters['Gruneisen'],
#                                                                  Wavenumber_Reference=
#                                                                  keyword_parameters['Wavenumber_Reference'],
#                                                                  Applied_strain=Applied_strain + strain_array -
#                                                                                 strain_array_2)
#                    wavenumbers_minus_minus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                   Gruneisen=keyword_parameters['Gruneisen'],
#                                                                   Wavenumber_Reference=
#                                                                   keyword_parameters['Wavenumber_Reference'],
#                                                                   Applied_strain=Applied_strain - strain_array -
#                                                                                  strain_array_2)
#                    wavenumbers_minus_plus = Wvn.Call_Wavenumbers(Method, min_RMS_gradient,
#                                                                  Gruneisen=keyword_parameters['Gruneisen'],
#                                                                  Wavenumber_Reference=
#                                                                  keyword_parameters['Wavenumber_Reference'],
#                                                                  Applied_strain=Applied_strain - strain_array +
#                                                                                 strain_array_2)
### End Change

                # Calculating the diagonal elements of d**2 G/(deta*deta)
                ddG_ddC[i, j] = (Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus_plus,
                                                      'pp' + file_ending, Statistical_mechanics, molecules_in_coord,
                                                      Parameter_file=keyword_parameters['Parameter_file']) -
                                 Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_plus_minus, 'pm' +
                                                      file_ending, Statistical_mechanics, molecules_in_coord,
                                                      Parameter_file=keyword_parameters['Parameter_file']) -
                                 Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus_plus, 'mp' +
                                                      file_ending, Statistical_mechanics, molecules_in_coord,
                                                      Parameter_file=keyword_parameters['Parameter_file']) +
                                 Pr.Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers_minus_minus, 'mm' +
                                                      file_ending, Statistical_mechanics, molecules_in_coord,
                                                      Parameter_file=keyword_parameters['Parameter_file'])) / \
                                (4 * numerical_crystal_matrix_step[i] * numerical_crystal_matrix_step[j])

                # Making d**2 G/(deta*deta) symetric
                ddG_ddC[j, i] = ddG_ddC[i, j]

                # Removing excess files
                os.system('rm pm' + file_ending + ' mp' + file_ending + ' pp' + file_ending + ' mm' + file_ending)

        # Removing excess files
        os.system('rm p' + file_ending + ' m' + file_ending)

    crystal_matrix_array = np.absolute(triangle_crystal_matrix_to_array(crystal_matrix))
    for i in range(3,6):
        if crystal_matrix_array[i] < 1.e-02:
            for j in [k for k in range(6) if k != i]:
                ddG_ddC[i, j] = 0.
                ddG_ddC[j, i] = 0.

    # Calculating deta/dT for all strains
    dC_dT = np.linalg.lstsq(ddG_ddC, dS_dC)[0]

    # Saving numerical outputs
    NO.aniso_gradient(dG_dC, ddG_ddC, dS_dC, dC_dT)
    return dC_dT, wavenumbers
