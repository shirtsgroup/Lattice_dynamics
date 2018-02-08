#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import subprocess
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr


##########################################
#                 Input                  #
##########################################
def Call_Wavenumbers(Method, min_RMS_gradient, **keyword_parameters):
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
    if (Method == 'SiQ') or (Method == 'GiQ') or (Method == 'GaQ') or (Method == 'HA'):
        # Directly computing the wavenumbers for a specific program, given a coordinate file
        if keyword_parameters['Program'] == 'Tinker':
            wavenumbers = Tinker_Wavenumber(keyword_parameters['Coordinate_file'], keyword_parameters['Parameter_file'])
        elif keyword_parameters['Program'] == 'Test':
            if Method == 'GaQ':
                wavenumbers = Test_Wavenumber(keyword_parameters['Coordinate_file'],
                                              keyword_parameters['ref_crystal_matrix'])
            else:
                wavenumbers = Test_Wavenumber(keyword_parameters['Coordinate_file'], True)
        elif keyword_parameters['Program'] == 'CP2K':
            return "Natalie add this in"
        return wavenumbers

    elif (Method == 'SiQg') or (Method == 'GiQg'):
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
            if os.path.isfile(keyword_parameters['Output'] + '_GRUwvn_' + Method + '.npy') and os.path.isfile(
                                            keyword_parameters['Output'] + '_GRU_' + Method + '.npy'):
                print("   ...Using Gruneisen parameters from: " + keyword_parameters['Output'] + '_GRU_' \
                      + Method + '.npy')
                Gruneisen = np.load(keyword_parameters['Output'] + '_GRU_' + Method + '.npy')
                Wavenumber_Reference = np.load(keyword_parameters['Output'] + '_GRUwvn_' + Method + '.npy')
                Volume_Reference = Pr.Volume(Coordinate_file=keyword_parameters['Coordinate_file'],
                                             Program=keyword_parameters['Program'],
                                             Parameter_file=keyword_parameters['Parameter_file'])
            # If the Gruneisen parameter has yet to be determined, here it will be calculated
            # It is assumed that the input Coordinate_file is the lattice minimum strucutre
            else:
                Gruneisen, Wavenumber_Reference, Volume_Reference = \
                    Setup_Isotropic_Gruneisen(keyword_parameters['Coordinate_file'],
                                              keyword_parameters['Program'],
                                              keyword_parameters['Gruneisen_Vol_FracStep'],
                                              keyword_parameters['molecules_in_coord'], min_RMS_gradient,
                                              Parameter_file=keyword_parameters['Parameter_file'])
                print("   ... Saving reference wavenumbers and Gruneisen parameters to: " + \
                      keyword_parameters['Output'] + '_GRU_' + Method + '.npy')
                np.save(keyword_parameters['Output'] + '_GRU_' + Method, Gruneisen)
                np.save(keyword_parameters['Output'] + '_GRUwvn_' + Method, Wavenumber_Reference)
            return Gruneisen, Wavenumber_Reference, Volume_Reference

    elif Method == 'GaQg':
        if ('Gruneisen' in keyword_parameters) and ('Wavenumber_Reference' in keyword_parameters) and \
                ('ref_crystal_matrix' in keyword_parameters):
            # Calculating the wavenumbers of the new anisotropically expanded structure
            # The Gruniesen parameter and reference wavenumbers have already been calculated
            wavenumbers = Get_Aniso_Gruneisen_Wavenumbers(keyword_parameters['Gruneisen'],
                                                          keyword_parameters['Wavenumber_Reference'],
                                                          keyword_parameters['ref_crystal_matrix'],
                                                          keyword_parameters['Coordinate_file'],
                                                          keyword_parameters['Program'])
            return wavenumbers

        else:
            if os.path.isfile(keyword_parameters['Output'] + '_GRUwvn_' + Method + '.npy') and os.path.isfile(
                                            keyword_parameters['Output'] + '_GRU_' + Method + '.npy'):
                # If the current directory has saved Gruneisen outputs, it will open those and use them
                print("   ...Using Gruneisen parameters from: " + keyword_parameters['Output'] + '_GRU_' \
                      + Method + '.npy')
                Gruneisen = np.load(keyword_parameters['Output'] + '_GRU_' + Method + '.npy')
                Wavenumber_Reference = np.load(keyword_parameters['Output'] + '_GRUwvn_' + Method + '.npy')
            else:
                # Calculating the Gruneisen parameter and wavenumbers
                Gruneisen, Wavenumber_Reference = \
                    Setup_Anisotropic_Gruneisen(keyword_parameters['Coordinate_file'], keyword_parameters['Program'],
                                                keyword_parameters['Gruneisen_Lat_FracStep'],
                                                keyword_parameters['molecules_in_coord'], min_RMS_gradient,
                                                Parameter_file=keyword_parameters['Parameter_file'])

                # Saving the wavenumbers for future use
                print("   ... Saving reference wavenumbers and Gruneisen parameters to: " + \
                      keyword_parameters['Output'] + '_GRU_/_GRUwvn' + Method + '.npy')
                np.save(keyword_parameters['Output'] + '_GRU_' + Method, Gruneisen)
                np.save(keyword_parameters['Output'] + '_GRUwvn_' + Method, Wavenumber_Reference)
            return Gruneisen, Wavenumber_Reference


##########################################
#       TINKER MOLECULAR MODELING        #
##########################################
def Tinker_Wavenumber(Coordinate_file, Parameter_file):
    """
    Calls the vibrate executable of Tinker Molecular Modeling and extracts the wavenumbers

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    Parameter_file = Tinker .key file specifying the force field parameter
    """
    # Calling Tinker's vibrate executable and extracting the eigenvalues and wavenumbers of the respective
    # Hessian and mass-weighted Hessian
    eigenvalues_and_wavenumbers = subprocess.check_output("vibrate %s -k %s  CR |  grep -oP '[-+]*[0-9]*\.[0-9]{2,9}'"
                                                          % (Coordinate_file, Parameter_file), shell=True).decode("utf-8")
    # Splitting the outputs into array form
    eigenvalues_and_wavenumbers = eigenvalues_and_wavenumbers.split('\n')
    eigenvalues_and_wavenumbers_hold = []
    for i in eigenvalues_and_wavenumbers:
        if i == '':
            pass
        else:
            eigenvalues_and_wavenumbers_hold.append(float(i))

    # Extracting the wavenumbers and assuring they're sorted from lowest to highest
    wavenumbers = np.sort(np.array(eigenvalues_and_wavenumbers_hold[len(eigenvalues_and_wavenumbers_hold)/2:]))
    return wavenumbers

##########################################
#                  CP2K                  #
##########################################

def CP2K_Wavenumber(coordinatefile, parameter_file, cp2kroot):
    wavenumbers = np.zeros((3,))
    wavenumfile = open(cp2kroot+'-VIBRATIONS-1.mol','r')
    lines = wavenumfile.readlines()
    iter = 2
    while '[FR-COORD]' not in lines[iter]:
        wave = lines[iter].split()
        wavenumbers = np.append(wavenumbers, float(wave[0]))
        iter = iter+1
    return wavenumbers

	 
    

##########################################
#                  Test                  #
##########################################
def Test_Wavenumber(Coordinate_file, ref_crystal_matrix, function='Test3'):
    """
    This function takes a set of lattice parameters in a .npy file and returns a set of wavenumbers
    Random functions can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters and atom coordinates
    """
    lattice_parameters = np.load(Coordinate_file)
    if function == 'Test1':
        wavenumbers = np.array([0., 0., 0., 52., 380., 1570., 3002.])
        for i in np.arange(3, len(wavenumbers[3:])+3):  # probably should be 3?
            wavenumbers[i] = wavenumbers[i]*(1/3.)*(((lattice_parameters[0]-16)/6)**2 + ((lattice_parameters[1] -
                                                                                          12)/5)**2 +
                                                    ((lattice_parameters[2] - 23)/11)**2)
    elif function == 'Test2':
        wavenumbers = np.arange(1, 200)
        wavenumbers = wavenumbers**(5.0/3.0)  # get something in the right range = 400^(4/3) = 2941
        wavenumbers[0:3] = [0, 0, 0]  # zero translation
        [refx, refy, refz] = [lattice_parameters[0]/10., lattice_parameters[1]/12., lattice_parameters[2]/15.]
#        [refx,refy,refz] = [lattice_parameters[0]/5,lattice_parameters[1]/6,lattice_parameters[2]/8]
        for i in range(3,len(wavenumbers[3:])+3):
            wavenumbers[i] = wavenumbers[i]*(1.0/15.0)*(2*refx**4.8 + 10*refy**4.2 + 4*np.sin(2*np.pi*refx) +
                                                        3*refz**4.8)
    elif function == 'Test3':
        if os.path.isfile('wvn0_test.npy') and os.path.isfile('wvnChange_test.npy'):
            if np.all(ref_crystal_matrix == True):
                strain = np.zeros(6)
                strain[:3] = (Pr.Volume(Program='Test', Coordinate_file=Coordinate_file)/ 400.47001361725802)**(1./3.) - 1.
            else:
                new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters('Test', Coordinate_file))
                strain = Pr.RotationFree_StrainArray_from_CrystalMatrix(ref_crystal_matrix, new_crystal_matrix)
            wvn0 = np.load('wvn0_test.npy')
            change = np.load('wvnChange_test.npy')
            wavenumbers = np.zeros(len(wvn0))
            for i in range(3, len(wavenumbers)):
                wavenumbers[i] = wvn0[i]*np.exp(-1.*np.sum(np.dot(strain, change[i]))) # + strain**2*(change[i]**2)))
        else:
            # Setting random wavenumbers
            wavenumbers = np.zeros(303)
            wavenumbers[3:241] = np.random.uniform(10., 2000., len(wavenumbers[3:241]))
            wavenumbers[241:] = np.random.uniform(2800., 3200., len(wavenumbers[241:]))
            wavenumbers = np.sort(wavenumbers)
            np.save('wvn0_test', wavenumbers)

            # Setting gruneisen parameters
            change = np.zeros((len(wavenumbers), 6))
            for i in range(3,len(wavenumbers)):
                change[i, :3] = np.random.uniform(-2*np.exp(-wavenumbers[i]/200.) - 0.001, 7*np.exp(-wavenumbers[i]/500.) + 0.001)
                for j in range(3, 6):
                    change[j] = np.random.uniform(-0.5*np.absolute(lattice_parameters[j] - 90.)*np.exp(-wavenumbers[i]/100.) - 0.01,
                                                  0.5*np.absolute(lattice_parameters[j] - 90.)*np.exp(-wavenumbers[i]/100.) + 0.01)
            np.save('wvnChange_test.npy', change)


    return wavenumbers


##########################################
#     Isotropic Gruneisen Parameter      #
##########################################
def Setup_Isotropic_Gruneisen(Coordinate_file, Program, Gruneisen_Vol_FracStep, molecules_in_coord, min_RMS_gradient,
                              **keyword_parameters):
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
    dLattice_Parameters = Ex.Isotropic_Change_Lattice_Parameters((1+Gruneisen_Vol_FracStep), Program, Coordinate_file)

    # Determining wavenumbers of lattice strucutre and expanded strucutre
    # Also, assigning a file ending name for the nex coordinate file (program dependent)
    if Program == 'Tinker':
        Ex.Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, 'temp', min_RMS_gradient,
                            dlattice_parameters=dLattice_Parameters,
                            Parameter_file=keyword_parameters['Parameter_file'])
        Organized_wavenumbers = Tinker_Gru_organized_wavenumbers('Isotropic', Coordinate_file, 'temp.xyz', keyword_parameters['Parameter_file'])
        Wavenumber_Reference = Organized_wavenumbers[0] 
        Wavenumber_expand = Organized_wavenumbers[1]
        lattice_parameters = Pr.Tinker_Lattice_Parameters(Coordinate_file)
        file_ending = '.xyz'
    if Program == 'CP2K':
        Ex.Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, 'temp', min_RMS_gradient,
                            dlattice_parameters=dLattice_Parameters,
                            Parameter_file=keyword_parameters['Parameter_file'], cp2kroot = keyword_parameters['cp2kroot'])
        Organized_wavenumbers = CP2K_Gru_organized_wavenumbers('Isotropic', Coordinate_file, 'temp.xyz', keyword_parameters['Parameter_file'])
        Wavenumber_Reference = Organized_wavenumbers[0] 
        Wavenumber_expand = Organized_wavenumbers[1]
        lattice_parameters = Pr.CP2K_Lattice_Parameters(Coordinate_file)
        file_ending = '.pdb'
    elif Program == 'Test':
        Ex.Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord, 'temp', min_RMS_gradient,
                            dlattice_parameters=dLattice_Parameters)
        Wavenumber_Reference = Test_Wavenumber(Coordinate_file, True)
        Wavenumber_expand = Test_Wavenumber('temp.npy', True)
        lattice_parameters = Pr.Test_Lattice_Parameters(Coordinate_file)
        file_ending = '.npy'

    # Calculating the volume of the lattice minimum and expanded structure
    Volume_Reference = Pr.Volume(lattice_parameters=lattice_parameters)
    Volume_expand = Volume_Reference + Gruneisen_Vol_FracStep*Volume_Reference

    # Calculating the Gruneisen parameter and zeroing out the parameters for the translational modes
    Gruneisen = np.zeros(len(Wavenumber_Reference))
    Gruneisen[3:] = -(np.log(Wavenumber_Reference[3:]) - np.log(Wavenumber_expand[3:]))/(np.log(Volume_Reference) -
                                                                                         np.log(Volume_expand))

    # Removing extra files created in process
    os.system('rm temp'+file_ending)
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
def Setup_Anisotropic_Gruneisen(Coordinate_file, Program, strain, molecules_in_coord, min_RMS_gradient,
                                **keyword_parameters):
    # Starting by straining the crystal in the six principal directions
    for i in range(6):
        # Making expanded structures in th direction of the six principal strains
        applied_strain = np.zeros(6)
        applied_strain[i] = strain
        Ex.Expand_Structure(Coordinate_file, Program, 'strain', molecules_in_coord, 'temp_' + str(i), min_RMS_gradient,
                            strain=Ex.strain_matrix(applied_strain),
                            crystal_matrix=Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file)),
                            Parameter_file=keyword_parameters['Parameter_file'])

    # Setting an array of the names of expanded strucutres
    expanded_coordinates = ['temp_0', 'temp_1', 'temp_2', 'temp_3', 'temp_4', 'temp_5']

    if Program == 'Tinker':
        file_ending = '.xyz'

        # Organizing the Tinker wavenumbers
        Organized_wavenumbers = Tinker_Gru_organized_wavenumbers('Anisotropic', Coordinate_file, 
                                                                 [s + '.xyz' for s in expanded_coordinates],
                                                                 keyword_parameters['Parameter_file'])

        # Setting aside the reference wavenumbers
        Wavenumber_Reference = Organized_wavenumbers[0]

        # Setting a blank matrix to save the Gruneisen parameters in
        Gruneisen = np.zeros((len(Wavenumber_Reference), 6))

        for i in range(6):
            # Calculating the Gruneisen parameters
            Gruneisen[3:, i] = -(np.log(Organized_wavenumbers[i + 1, 3:]) - np.log(Wavenumber_Reference[3:])) / strain
            os.system('rm ' + expanded_coordinates[i] + file_ending)

    elif Program == 'Test':
        file_ending = '.npy'
        Wavenumber_Reference = Test_Wavenumber(Coordinate_file, True)
        Gruneisen = np.zeros((len(Wavenumber_Reference), 6))
        for i in range(6):
            applied_strain = np.zeros(6)
            applied_strain[i] = strain
            Wavenumber_expand = Test_Wavenumber(expanded_coordinates[i] + file_ending,
                                                Ex.Lattice_parameters_to_Crystal_matrix(Pr.Test_Lattice_Parameters(Coordinate_file)))
            Gruneisen[3:, i] = -(np.log(Wavenumber_expand[3:]) - np.log(Wavenumber_Reference[3:])) / strain
            os.system('rm ' + expanded_coordinates[i] + file_ending)
    return Gruneisen, Wavenumber_Reference


def Get_Aniso_Gruneisen_Wavenumbers(Gruneisen, Wavenumber_Reference, ref_crystal_matrix, Coordinate_file, Program):
    # Setting a blank array for new wavenumbers
    new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file))
    applied_strain = Pr.RotationFree_StrainArray_from_CrystalMatrix(ref_crystal_matrix, new_crystal_matrix)

    wavenumbers = np.zeros(len(Wavenumber_Reference))

    for i in np.arange(3, len(wavenumbers), 1):
        # Computing the change to each wavenumber due to the curren strain
        hold = 0
        for j in range(6):
            hold = hold + -1 * applied_strain[j] * Gruneisen[i, j]
        wavenumbers[i] = Wavenumber_Reference[i] * np.exp(hold)
    return wavenumbers

##########################################
#     Organizing Wavenumbers for Gru     #
##########################################
def Tinker_Gru_organized_wavenumbers(Expansion_type, Coordinate_file, Expanded_Coordinate_file, Parameter_file):
    from munkres import Munkres, print_matrix
    m = Munkres()

    number_of_modes = 3*Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)

    if Expansion_type == 'Isotropic':
        wavenumbers = np.zeros((2, number_of_modes))
        eigenvectors = np.zeros((2, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = Tinker_Wavenumber_and_Vectors(Coordinate_file, Parameter_file)
        wavenumbers[1], eigenvectors[1] = Tinker_Wavenumber_and_Vectors(Expanded_Coordinate_file, Parameter_file)
    elif Expansion_type == 'Anisotropic':
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = Tinker_Wavenumber_and_Vectors(Coordinate_file, Parameter_file)
        for i in xrange(1,7):
            wavenumbers[i], eigenvectors[i] = Tinker_Wavenumber_and_Vectors(Expanded_Coordinate_file[i-1], Parameter_file)

    # Weighting the modes matched together
    wavenumbers_out = np.zeros((len(wavenumbers[:, 0]), number_of_modes))
    wavenumbers_out[0] = wavenumbers[0]
    for k in xrange(1, len(wavenumbers[:, 0])):
        weight = np.zeros((number_of_modes - 3, number_of_modes - 3))
        for i in xrange(3, number_of_modes):
            diff = np.dot(eigenvectors[0, i], eigenvectors[k, i])/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, i]))
            if np.absolute(diff) > 0.95:
                weight[i - 3] = 10000000.
                weight[i - 3, i - 3] = 1. - diff
            else:
                for j in xrange(3, number_of_modes):
                    hold_weight = np.zeros(4)
                    hold_weight[0] = 1 - np.dot(-1*eigenvectors[0, i], eigenvectors[k, j])/(np.linalg.norm(-1*eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, j]))
                    hold_weight[1] = 1 - np.dot(eigenvectors[0, i], -1*eigenvectors[k, j])/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(-1*eigenvectors[k, j]))
                    hold_weight[2] = 1 - np.dot(eigenvectors[0, i], eigenvectors[k, j])/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, j]))
                    hold_weight[3] = 1 - np.dot(-1*eigenvectors[0, i], -1*eigenvectors[k, j])/(np.linalg.norm(-1*eigenvectors[0, i])*np.linalg.norm(-1*eigenvectors[k, j]))
                    weight[i - 3, j - 3] = min(hold_weight)
        # Using the Hungarian algorithm to match wavenumbers
        Wgt = m.compute(weight)
        x,y = zip(*Wgt)
        z = np.column_stack((x,y))
        z = z +3

    # Re-organizing the expanded wavenumbers
        for i in z:
            wavenumbers_out[k, i[0]] = wavenumbers[k, i[1]]
    return wavenumbers_out


def CP2K_Gru_organized_wavenumbers(Expansion_type, Coordinate_file, Expanded_Coordinate_file, Parameter_file):
    from munkres import Munkres, print_matrix
    m = Munkres()

    number_of_modes = 3*Pr.CP2K_atoms_per_molecule(Coordinate_file, 1)

    if Expansion_type == 'Isotropic':
        wavenumbers = np.zeros((2, number_of_modes))
        eigenvectors = np.zeros((2, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = CP2K_Wavenumber_and_Vectors(Coordinate_file, Parameter_file)
        wavenumbers[1], eigenvectors[1] = CP2K_Wavenumber_and_Vectors(Expanded_Coordinate_file, Parameter_file)
    elif Expansion_type == 'Anisotropic':
        wavenumbers = np.zeros((7, number_of_modes))
        eigenvectors = np.zeros((7, number_of_modes, number_of_modes))
        wavenumbers[0], eigenvectors[0] = CP2K_Wavenumber_and_Vectors(Coordinate_file, Parameter_file)
        for i in xrange(1,7):
            wavenumbers[i], eigenvectors[i] = CP2K_Wavenumber_and_Vectors(Expanded_Coordinate_file[i-1], Parameter_file)


    # Weighting the modes matched together
    wavenumbers_out = np.zeros((len(wavenumbers[:, 0]), number_of_modes))
    wavenumbers_out[0] = wavenumbers[0]
    for k in xrange(1, len(wavenumbers[:, 0])):
        weight = np.zeros((number_of_modes - 3, number_of_modes - 3))
        for i in xrange(3, number_of_modes):
            diff = np.linalg.norm(np.dot(eigenvectors[0, i], eigenvectors[k, i]))/(np.linalg.norm(eigenvectors[0, i])*np.linalg.norm(eigenvectors[k, i]))
            if diff > 0.95:
                weight[i - 3] = 10000000.
                weight[i - 3, i - 3] = 1. - diff
            else:
                for j in xrange(3, number_of_modes):
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


def Tinker_Wavenumber_and_Vectors(Coordinate_file, Parameter_file):
    # Calling Tinker's vibrate executable and extracting the eigenvectors and wavenumbers of the respective
    # Hessian and mass-weighted Hessian
    os.system('cp ' + Coordinate_file + ' vector_temp.xyz')
    output = subprocess.check_output("vibrate vector_temp.xyz -k %s  A |  grep -oP '[-+]*[0-9]*\.[0-9]{2,9}'"
                                                          % (Parameter_file), shell=True)

    os.system('rm vector_temp.*')

    # Finding the number modes in the system
    number_of_modes = 3*Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)

    # Splitting the outputs into array form
    output = output.split('\n')
    output.remove('')
    output = np.array(output).astype(float)

    # Grabbing the wavenumbers
    wavenumbers = np.array(output[number_of_modes: number_of_modes*2]).astype(float)

    # Grabbing the eigenvectors
    eigenvectors = np.zeros((number_of_modes, number_of_modes))
    for i in range(number_of_modes):
        start = number_of_modes*(i + 2) + i + 1
        finish = start + number_of_modes
        eigenvectors[i] = output[start: finish] /np.sqrt(np.sum(output[start: finish]**2))
    return wavenumbers, eigenvectors


def CP2K_Wavenumber_and_Vectors(Coordinate_file, Parameter_file):
    # Calling Tinker's vibrate executable and extracting the eigenvectors and wavenumbers of the respective
    # Hessian and mass-weighted Hessian
    os.system('cp ' + Coordinate_file + ' vector_temp.xyz')
    output = subprocess.check_output("vibrate vector_temp.xyz -k %s  A |  grep -oP '[-+]*[0-9]*\.[0-9]{2,9}'"
                                                          % (Parameter_file), shell=True)

    os.system('rm vector_temp.*')

    # Finding the number modes in the system
    number_of_modes = 3*Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)

    # Splitting the outputs into array form
    output = output.split('\n')
    output.remove('')
    output = np.array(output).astype(float)

    # Grabbing the wavenumbers
    wavenumbers = np.array(output[number_of_modes: number_of_modes*2]).astype(float)

    # Grabbing the eigenvectors
    eigenvectors = np.zeros((number_of_modes, number_of_modes))
    for i in range(number_of_modes):
        start = number_of_modes*(i + 2) + i + 1
        finish = start + number_of_modes
        eigenvectors[i] = output[start: finish] /np.sqrt(np.sum(output[start: finish]**2))
    return wavenumbers, eigenvectors


