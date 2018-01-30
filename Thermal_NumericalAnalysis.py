#!/usr/bin/env python
import os
import subprocess
import sys
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Numerical_Outputs as NO

##########################################
#           Numerical Methods            #
##########################################
def Runge_Kutta_Fourth_Order(Method, Coordinate_file, Program, Temperature, Pressure, molecules_in_coord,
                             Statistical_mechanics, RK4_stepsize, min_RMS_gradient, **keyword_parameters):
    """
    This function determines the gradient of thermal expansion of a strucutre between two temperatures using
    a forth order Runge-Kutta numerical analysis
    :param Method: Gradient Isotropic QHA ('GiQ');
                   Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
                   Gradient Anisotropic QHA ('GaQ');
    :param Coordinate_file: file containing the lattice parameters (and coordinates)
    :param Program: 'Tinker' for Tinker Molecular Modeling
                    'Test' for a test run
    :param Temperature: in Kelvin
    :param Pressure: in atm
    :param molecules_in_coord: number of molecules in coordinate file
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param RK4_stepsize: stepsize for runge-kutta 4th order
    :param keyword_parameters: Parameter_file, LocGrd_Vol_FracStep, LocGrd_LatParam_FracStep, Gruneisen, 
    Wavenumber_Reference, Volume_Reference, Aniso_LocGrad_Type

    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    LocGrd_Vol_FracStep: isotropic volume fractional stepsize for local gradient
    LocGrd_LatParam_FracStep: anisotropic crystal matrix fractional stepsize for local gradient
    Gruneisen: Gruneisen parameters found with Setup_Isotropic_Gruneisen
    Wavenumber_Reference: Reference wavenumbers for Gruneisen parameter
    Volume_Reference: Reference volume of structure for Wavenumber_Reference
    Aniso_LocGrad_Type: 73 Hessians to calculate the complete anistropic gradient
                        25 for d**2G_dhdh only calculating the diagonals and off-diags. of the upper left 3x3 matrix
                        19 for d**2G_dhdh only calculating the uppder left 3x3 matrix
                        13 for d**2G_dhdh only calculating the diagonals
                        7  for d**2G_dhdh only calculating the upper left 3x3 matrix daigonals
    Crystal_matrix_Reference:
    """
    # Setting up program specific file endings and giving parameter files blank names to avoid errors
    file_ending = Ex.assign_file_ending(Program)

    # Output of numerical analysis
    NO.start_RK(Temperature, RK4_stepsize)

    # Final array for weights on slopes
    RK_multiply = np.array([1. / 6., 1. / 3., 1. / 3., 1. / 6.])

    # Copying the coordinate file to a seperate file to work with
    os.system('cp ' + Coordinate_file + ' RK4' + file_ending)

    # Setting the different temperature stepsizes
    temperature_steps = np.array([0., RK4_stepsize / 2., RK4_stepsize / 2., RK4_stepsize])

    # Setting RK_4 array/matix and general parameters that aren't required for specific methods
    if (Method == 'GiQ') or (Method == 'GiQg'):
        # Setting array to save 4 gradients in dV/dT
        RK_grad = np.zeros(4)
        if Method == 'GiQ':
### These next parts shouldn't be needed`
            # Setting parameters for non-Gruneisen methods
            keyword_parameters['Gruneisen'] = 0.
            keyword_parameters['Wavenumber_Reference'] = 0.
            keyword_parameters['Volume_Reference'] = 0.
    elif (Method == 'GaQ') or (Method == 'GaQg'):
        # Setting arra to save 4 gradients for the six different strains d\eta/dT
        RK_grad = np.zeros((4, 6))
        if Method == 'GaQ':
            # Setting parameters for non-Gruneisen methods
            keyword_parameters['Gruneisen'] = 0.
            keyword_parameters['Wavenumber_Reference'] = 0.
            keyword_parameters['Volume_Reference'] = 0.

    # Setting a varaible to track strain during RK4
    RK_strain = np.zeros(6)

    # Calculating the RK gradients for the overall numerical gradient
    for i in range(4):
        # Outputting numerical analysis
        NO.step_RK(i, Temperature + temperature_steps[i], Program, 'RK4' + file_ending)

        print "   + Performing Runge-Kutta step " + str(i + 1)
        if (Method == 'GiQ') or (Method == 'GiQg'):
            # Determining the slope at the current RK step
            RK_grad[i], wavenumbers_hold, volume_hold = \
                Ex.Call_Expansion(Method, 'local_gradient', Program, 'RK4' + file_ending, molecules_in_coord,
                                  min_RMS_gradient, Temperature=Temperature + temperature_steps[i], Pressure=Pressure,
                                  volume_fraction_change=keyword_parameters['LocGrd_Vol_FracStep'],
                                  Statistical_mechanics=Statistical_mechanics,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  Gruneisen=keyword_parameters['Gruneisen'],
                                  Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                  Volume_Reference=keyword_parameters['Volume_Reference'])
        elif (Method == 'GaQ') or (Method == 'GaQg'):
            # Determining the slope at the current RK step
            RK_grad[i], wavenumbers_hold = \
                Ex.Call_Expansion(Method, 'local_gradient', Program, 'RK4' + file_ending, molecules_in_coord,
                                  min_RMS_gradient, Temperature=Temperature + temperature_steps[i], Pressure=Pressure,
                                  LocGrd_NormStrain=keyword_parameters['LocGrd_NormStrain'],
                                  LocGrd_ShearStrain=keyword_parameters['LocGrd_ShearStrain'],
                                  Statistical_mechanics=Statistical_mechanics,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  Gruneisen=keyword_parameters['Gruneisen'],
                                  Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                  ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'],
                                  Aniso_LocGrad_Type=keyword_parameters['Aniso_LocGrad_Type'])
            volume_hold = 0.

        if i == 0:
            # Saving outputs to be passed to the earlier code (local gradient and wavenumbers of initial strucuture)
            wavenumbers = 1. * wavenumbers_hold
            volume = 1. * volume_hold
            k1 = 1. * RK_grad[0]

        if i != 3:
            if (Method == 'GiQ') or (Method == 'GiQg'):
                # For isotropic expansion, determining the volume fraction change of the input strucutre (V_new/V_input)
                volume_fraction_change = (volume + RK_grad[i] * temperature_steps[i + 1]) / volume

                # Expanding the crystal to the next step size
                Ex.Call_Expansion(Method, 'expand', Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  volume_fraction_change=volume_fraction_change, Output='RK4')

            elif (Method == 'GaQ') or (Method == 'GaQg'):
                # For anisotropic expansion, determining the strain of th input strucutre for the next step
                RK_crystal_matrix = RK_grad[i] * temperature_steps[i + 1]

                # Expanding the crystal to the next step size
                Ex.Call_Expansion(Method, 'expand', Program, Coordinate_file,
                                  molecules_in_coord, min_RMS_gradient,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  dcrystal_matrix=Ex.array_to_triangle_crystal_matrix(RK_crystal_matrix),
                                  Output='RK4')

        # Multiplying the found gradient by the fraction it will contribute to the overall gradient
        RK_grad[i] = RK_grad[i] * RK_multiply[i]

    # Summing all RK gradients for the overall numerical gradient
    numerical_gradient = np.sum(RK_grad, axis=0)

    # Removign excess files
    os.system('rm RK4' + file_ending)
    return numerical_gradient, wavenumbers, volume, k1

def RK_Dense_Output(theta, y_0, y_1, f_0, f_1, h):
    return (1 - theta) * y_0 + theta * y_1 + theta * (theta - 1) * ((1 - 2 * theta) * (y_1 - y_0) +
                                                                    (theta - 1) * h * f_0 + theta * h * f_1)


def Spline_Intermediate_Points(Output, Method, Program, properties, Temperature, molecules_in_coord, Pressure,
                               Statistical_mechanics, min_RMS_gradient, **keyword_parameters):
    """
    This funciton determines intermediate
    :param Output: string for outputted files
    :param Method: Gradient Isotropic QHA ('GiQ');
                   Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
                   Gradient Anisotropic QHA ('GaQ');
                   Gradient Anisotropic QHA w/ Gruneisen Parameter ('GaQg');
    :param Program: 'Tinker' for Tinker Molecular Modeling
                    'Test' for a test run
    :param properties: Properties previously calculated with gradient approach
    :param Temperature: temperatures that were not computed with gradient approach in K
    :param molecules_in_coord: number of molecules in the coordinate file
    :param Pressure: pressure in atm
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param keyword_parameters: Parameter_file
    :return: 

    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen
    Wavenumber_Reference
    Volume_Reference
    Crystal_matrix_Reference
    """
    print "Using cubic spline to determine intermediate temperature steps."
    # Setting file endings
    if Program == 'Tinker':
        file_ending = '.xyz'
    elif Program == 'Test':
        file_ending = '.npy'
        keyword_parameters['Parameter_file'] = ''

    # Making temperature array for wanted output values (user specified)
    Temperature = np.sort(np.unique(Temperature))

    # Setting step points and tangents/gradients at those points
    if (Method == 'GiQ') or (Method == 'GiQg'):
        # Loading in the local gradients found form numerical analysis
        tangent = np.load(Output + '_dV_' + Method + '.npy')
        # Setting the isotropic volumes for the gradients loaded in
        y = properties[:, 6]
    elif (Method == 'GaQ') or (Method == 'GaQg'):
        # Loading in the local gradients found form numerical analysis
        tangent = np.load(Output + '_dC_' + Method + '.npy')
        # Setting the anisotropic strain for the gradients loaded in
        y = np.zeros((len(properties[:, 0]), 6))
        for i in range(1, len(properties[:, 0] + 1)):
            # Using the slopes from RK to solve for the strains in numerical analysis
            y[i] = y[i - 1] + tangent[i - 1, 0, 1:] * (properties[i, 0] - properties[i - 1, 0])

    # Setting up an array to store the output properties
    properties_out = np.zeros((len(np.unique(np.append(Temperature, properties[:, 0]))), len(properties[0, :])))

    # Setting a count on where to place the properites in the output matrix
    count = 0

    # Pulling up starting structure
    subprocess.call(['cp', 'Cords/' + Output + '_' + Method + 'T' + str(properties[0, 0]) + file_ending, 
                     './temp' + file_ending])

    # Setting variables for loop
#    temp_strain = np.zeros(6)
    new_strain = np.zeros(6)
    new_volume = 1.
    temp_volume = Pr.Volume(Program=Program, Coordinate_file='temp' + file_ending)
#    temp_volume = 1.

    for i in range(len(properties[:, 0]) - 1):
        # Stepsize used for numerical analysis
        h = properties[i + 1, 0] - properties[i, 0]

        # Adding properties found for lower bound temperature in Numerical analysis to output matrix
        properties_out[count] = properties[i]
        count = count + 1

        for j in range(len(Temperature)):
            if Temperature[j] == properties[i + 1, 0]:
                pass
#                # Copying up the new reference strucutre
#                subprocess.call(['cp', 'Cords/' + Output + '_' + Method + 'T' + str(properties[i + 1, 0]) + file_ending,
#                                 './temp' + file_ending])
#                if (Method == 'GaQ') or (Method == 'GaQg'):
#                    # Determining the new strain of the reference strucutre
#                    temp_strain = y[i + 1]

            elif properties[i, 0] < Temperature[j] < properties[i + 1, 0]:
                print "   Using a Spline, adding intermediate temperature at:" + str(Temperature[j]) + " K"
                # Computing the properties for all intermediate temperature points
                theta = (Temperature[j] - properties[i, 0]) / h
                if (Method == 'GiQ') or (Method == 'GiQg'):
                    # Computing the volume we are currently at and the next intermediate step
                    new_volume = RK_Dense_Output(theta, y[i], y[i + 1], tangent[i, 2], tangent[i+1, 2], h)
#                    temp_volume = Pr.Volume(Program=Program, Coordinate_file='temp' + file_ending)
                elif (Method == 'GaQ') or (Method == 'GaQg'):
                    # Computing the strain from the lattice minimum structure to the next intermediate step
                    new_strain = np.zeros(6)
                    for k in range(6):
                        new_strain[k] = RK_Dense_Output(theta, y[i, k], y[i + 1, k], tangent[i, 1, k + 1],
                                                        tangent[i + 1, 1, k + 1], h)

                # Expanding the strucutre, from the structure at the last temperature to the next intermediate step
                if os.path.isfile('Cords/' + Output + '_' + Method + 'T' + str(Temperature[j]) + file_ending):
                     pass
#                    print "   ...Using previous coordinate file"
#                    subprocess.call(['cp', 'Cords/' + Output + '_' + Method + 'T' + str(Temperature[j]) + file_ending, './temp' + file_ending])
                else:
                    Ex.Call_Expansion(Method, 'expand', Program, 'temp' + file_ending, molecules_in_coord, min_RMS_gradient,
                                      Parameter_file=keyword_parameters['Parameter_file'],
                                      volume_fraction_change=new_volume / temp_volume, strain=new_strain,
                                      Output='hold')

                # Computing the wavenumbers for the new structure
                wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program,
                                                   Coordinate_file='hold' + file_ending,
                                                   Parameter_file=keyword_parameters['Parameter_file'],
                                                   Gruneisen=keyword_parameters['Gruneisen'],
                                                   Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                   Volume_Reference=properties[0, 6], Applied_strain=new_strain,
                                                   New_Volume=new_volume)

                # Computing the properites
                properties_out[count] = Pr.Properties('hold' + file_ending, wavenumbers, Temperature[j], Pressure,
                                                      Program, Statistical_mechanics, molecules_in_coord,
                                                      Parameter_file=keyword_parameters['Parameter_file'])

                # Moving new intermediate structure to the Cords directory for storage
                subprocess.call(['mv', 'hold' + file_ending, 'Cords/' + Output + '_' + Method + 'T' +
                                 str(Temperature[j]) + file_ending])

                count = count + 1
                temp_strain = new_strain

    # Setting the last numerical step to the output matrix
    properties_out[-1] = properties[-1]

    # Removing the temporary strucutre
    os.system('rm temp' + file_ending)
    return properties_out


##########################################
#      Stepwise Isotropic Expansion      #
##########################################
def Isotropic_Stepwise_Expansion(StepWise_Vol_StepFrac, StepWise_Vol_LowerFrac, StepWise_Vol_UpperFrac, Coordinate_file,
                                 Program, Temperature, Pressure, Output, Method, molecules_in_coord, Wavenum_Tol,
                                 Statistical_mechanics, min_RMS_gradient, **keyword_parameters):
    """
    This function performs stepwise isotropic QHA either with or without the gruneisen parameter
    :param StepWise_Vol_StepFrac: volumetric fraction step
    :param StepWise_Vol_LowerFrac: lower bound on the fraction to compress to
    :param StepWise_Vol_UpperFrac: uppder gound on the fraction to expand to
    :param Coordinate_file: file containing the lattice parameters (and coordinates)
    :param Program: 'Tinker' for Tinker Molecular Modeling
                    'Test' for a test run
    :param Temperature: array of temperatures in Kelvin
    :param Pressure: in atm
    :param Output: string for outputted files
    :param Method: Stepwise Isotropic QHA ('SiQ');
                   Stepwise Isotropic QHA w/ Gruneisen Parameter ('SiQg');
    :param molecules_in_coord: number of molecules in coordinate file
    :param Wavenum_Tol: lowest tollerable wavenumbers (some small negative wavenumbers are okay)
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param keyword_parameters: Parameter_file, Gruneisen_Vol_FracStep
    
    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen_Vol_FracStep: volume fraction step used to determine the Gruneisen parameter 
    """
    # Setting file endings and determining how many wavenumbers there will be
    if Program == 'Tinker':
        file_ending = '.xyz'
        number_of_wavenumbers = Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)*3
    elif Program == 'Test':
        file_ending = '.npy'
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file, True))
        keyword_parameters['Parameter_file'] = ''
    elif Program =='CP2L':
	file_ending = '.pdb'
	number_of_wavenumbers = Pr.CP2K_atoms_per_molecule(Coordinate_file,1)*3

    # Setting up array of volume fractions from the lattice structure
    lower_volume_fraction = np.arange(StepWise_Vol_LowerFrac, 1.0, StepWise_Vol_StepFrac)[::-1]
    if len(lower_volume_fraction) > 0:
        if lower_volume_fraction[0] != 1.0:
            lower_volume_fraction = np.insert(lower_volume_fraction, 0, 1.0)
    upper_volume_fraction = np.arange(1.0 + StepWise_Vol_StepFrac, StepWise_Vol_UpperFrac, StepWise_Vol_StepFrac)
    volume_fraction = np.append(lower_volume_fraction, upper_volume_fraction)

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(volume_fraction), number_of_wavenumbers+1))
    wavenumbers[:, 0] = volume_fraction

    # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
#    existing_wavenumbers = False
    if Method == 'SiQg':
        print "   Calculating the isotropic Gruneisen parameter"
        Gruneisen, Wavenumber_Reference, Volume_Reference = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Output=Output,
                                                                                 Coordinate_file=Coordinate_file,
                                                                                 Program=Program,
                                                                                 Gruneisen_Vol_FracStep=
                                                                                 keyword_parameters[
                                                                                     'Gruneisen_Vol_FracStep'],
                                                                                 molecules_in_coord=molecules_in_coord,
                                                                                 Parameter_file=
                                                                                 keyword_parameters['Parameter_file'], cp2kroot=keyword_parameters['cp2kroot'])
    elif Method == 'SiQ':
        Gruneisen = 0.
        Wavenumber_Reference = 0.
        Volume_Reference = 0.
        if os.path.isfile(Output + '_WVN_' + Method + '.npy'):
            old_wavenumbers = np.load(Output + '_WVN_' + Method + '.npy')
#            existing_wavenumbers = True
            for i in range(len(old_wavenumbers[:, 0])):
                if any(volume_fraction == old_wavenumbers[i,0]):
                    loc = np.where(volume_fraction == old_wavenumbers[i,0])[0][0]
                    wavenumbers[loc] = old_wavenumbers[i]

    # setting a matrix for properties versus temperature and pressure
    properties = np.zeros((len(volume_fraction), len(Temperature), 14))

    # Finding all expanded structures
    previous_volume = 1.0
    lattice_volume = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)
    os.system('cp ' + Coordinate_file + ' ' + Output + '_' + Method + str(previous_volume) + file_ending)
    for i in range(len(volume_fraction)):
        print "   Performing volume fraction of: " + str(volume_fraction[i])
        if os.path.isfile('Cords/' + Output + '_' + Method + str(volume_fraction[i]) + file_ending):
            print "   ... Coordinate file Cords/" + Output + "_" + Method + str(volume_fraction[i]) + file_ending + \
                  "already exists"
            # Skipping structures if they've already been constructed
            os.system('cp Cords/' + Output + '_' + Method + str(volume_fraction[i]) + file_ending + ' ./')
        else:
            Ex.Call_Expansion(Method, 'expand', Program, Output + '_' + Method + str(previous_volume) + file_ending,
                              molecules_in_coord, min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                              volume_fraction_change=(volume_fraction[i]/previous_volume),
                              Output=Output + '_' + Method + str(volume_fraction[i]))

        # Calculating wavenumbers of new expanded strucutre
        find_wavenumbers = True
        if any(wavenumbers[i, 1:] > 0.):
            pass
#        if existing_wavenumbers == True:
#            for j in range(len(old_wavenumbers[:, 0])):
#                if old_wavenumbers[j, 0] == volume_fraction[i]:
#                    print "   ... Using wavenumbers already computed in: " + Output + "_WVN_" + Method + ".npy"
#                    wavenumbers[i, 1:] = old_wavenumbers[j, 1:]
#                    find_wavenumbers = False
#
#        if find_wavenumbers == True:
        else:
            wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program, Gruneisen=Gruneisen,
                                                      Wavenumber_Reference=Wavenumber_Reference,
                                                      Volume_Reference=Volume_Reference,
                                                      New_Volume=volume_fraction[i]*lattice_volume,
                                                      Coordinate_file=(Output + '_' + Method + str(volume_fraction[i])
                                                                       + file_ending),
                                                      Parameter_file=keyword_parameters['Parameter_file'], cp2kroot=keyword_parameters['cp2kroot'])

        # Saving the wavenumbers if the Gruneisen parameter is not being used
        if Method == 'SiQ':
            print "   ... Saving wavenumbers in: " + Output + "_WVN_" + Method + ".npy"
            np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

        # Calculating properties of systems with wavenumbers above user specified tollerance
        if all(wavenumbers[i, 1:] > Wavenum_Tol):
            print "   ... Wavenumbers are greater than tolerance of: " + str(Wavenum_Tol) + " cm^-1"
            properties[i, :, :] = Pr.Properties_with_Temperature(Output + '_' + Method + str(volume_fraction[i]) +
                                                                 file_ending, wavenumbers[i, 1:], Temperature, Pressure,
                                                                 Program, Statistical_mechanics, molecules_in_coord,
                                                                 keyword_parameters['cp2kroot'],
                                                                 Parameter_file=keyword_parameters['Parameter_file'])
        else:
            print "   ... WARNING: wavenumbers are lower than tolerance of: " + str(Wavenum_Tol) + " cm^-1"
            print "      ... Properties will be bypassed for this paricular strucutre."
            properties[i, :, :] = np.nan

    os.system('mv ' + Output + '_' + Method + '*' + file_ending + ' Cords/')

    # Saving the raw data before minimizing
    print "   All properties have been saved in " + Output + "_raw.npy"
    np.save(Output+'_raw', properties)

    # Building matrix for minimum Gibbs Free energy data across temperature range
    minimum_gibbs_properties = np.zeros((len(Temperature), 14))
    for i in range(len(properties[0, :, 0])):
        for j in range(len(properties[:, 0, 0])):
            if properties[j, i, 2] == np.nanmin(properties[:, i, 2]):
                minimum_gibbs_properties[i, :] = properties[j, i, :]
    return minimum_gibbs_properties

##########################################
#      Gradient Isotropic Expansion      #
##########################################
def Isotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method, Gradient_MaxTemp,
                                 Pressure, LocGrd_Vol_FracStep, Statistical_mechanics,
                                 NumAnalysis_step, NumAnalysis_method, Temperature, min_RMS_gradient, **keyword_parameters):
    """
    This function calculated the isotropic gradient for thermal expansion and returns the properties along that path
    :param Coordinate_file: file containing the lattice parameters (and coordinates)
    :param Program: 'Tinker' for Tinker Molecular Modeling
                    'Test' for a test run
    :param molecules_in_coord: number of molecules in coordinate file
    :param Output: string for outputted files
    :param Method: Gradient Isotropic QHA ('GiQ');
                   Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
    :param Gradient_MaxTemp: Maximum temperature in gradient method
    :param Pressure: in atm
    :param LocGrd_Vol_FracStep:  isotropic volume fractional stepsize for local gradient
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param NumAnalysis_step: stepsize for numerical method
    :param NumAnalysis_method: 'RK4' Runge-Kutta 4th order
    :param keyword_parameters: Parameter_file, Gruneisen_Vol_FracStep
    
    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen_Vol_FracStep: volume fraction step used to determine the Gruneisen parameter 
    """
    # Setting file endings and determining how many wavenumbers there will be
    if Program == 'Tinker':
        file_ending = '.xyz'
        number_of_wavenumbers = Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)*3
    elif Program == 'Test':
        file_ending = '.npy'
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file, True))
        keyword_parameters['Parameter_file'] = ''
    elif Program =='CP2K':
        file_ending = '.pdb'
        number_of_wavenumbers = PR.CP2K_atoms_per_molecule(Coordinate_file, 1)*3
    # Setting the temperature array
    temperature = np.arange(0, Gradient_MaxTemp + 1., NumAnalysis_step)

    # Setting the volume gradient array to be filled
    volume_gradient = np.zeros((len(temperature), 3))
    volume_gradient[:, 0] = temperature[:len(temperature)]
    if os.path.isfile(Output + '_dV_' + Method + '.npy'):
        print "Using volume gradients in: " + Output + "_dV_" + Method + ".npy"
        volume_gradient_hold = np.load(Output + '_dV_' + Method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(volume_gradient_hold[:, 0]) <= len(volume_gradient[:, 0]):
            if all(volume_gradient_hold[:, 0] == volume_gradient[:len(volume_gradient_hold[:, 0]), 0]):
                volume_gradient[:len(volume_gradient_hold[:, 0]), :] = volume_gradient_hold

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_wavenumbers+1))
    wavenumbers[:, 0] = temperature

    # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
    if Method == 'GiQg':
        print "   Calculating the isotropic Gruneisen parameter"
        Gruneisen, Wavenumber_Reference, Volume_Reference = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Output=Output,
                                                                                 Coordinate_file=Coordinate_file,
                                                                                 Program=Program,
                                                                                 Gruneisen_Vol_FracStep=
                                                                                 keyword_parameters[
                                                                                     'Gruneisen_Vol_FracStep'],
                                                                                 molecules_in_coord=molecules_in_coord,
                                                                                 Parameter_file=
                                                                                 keyword_parameters['Parameter_file'], cp2kroot=keyword_parameters['cp2kroot'])
    elif Method == 'GiQ':
        Gruneisen = 0.
        Wavenumber_Reference = 0.
        Volume_Reference = 0.
        if os.path.isfile(Output + '_WVN_' + Method + '.npy'):
            wavenumbers_hold = np.load(Output + '_WVN_' + Method + '.npy')
            # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    print "Using wavenumbers previously computed in: " + Output + "_WVN_" + Method + ".npy"
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Holding lattice structures as the structure at 0K
    os.system('cp ' + Coordinate_file + ' ' + Output + '_' + Method + 'T' + str(temperature[0]) + file_ending)

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        print "   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K"
        if any(wavenumbers[i, 4:] != 0.) or Method == 'GiQg' and (volume_gradient[i, 1] != 0.) and \
                (os.path.isfile('Cords/' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending)):
            print "   ... Using expansion gradient and wavenumbers previously found"
            os.system('mv ' 'Cords/' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending + ' ./')
            volume = Pr.Volume(Program=Program, Coordinate_file=Output + '_' + Method + 'T' + str(temperature[i])
                               + file_ending)
            if Method == 'GiQg':
                wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen,
                                                          Wavenumber_Reference=Wavenumber_Reference,
                                                          Volume_Reference=Volume_Reference, New_Volume=volume)
            pass
        else:
            if NumAnalysis_method == 'RK4':
                volume_gradient[i, 1], wavenumbers[i, 1:], volume, volume_gradient[i, 2] = \
                    Runge_Kutta_Fourth_Order(Method, Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                             Program, temperature[i], Pressure,
                                             molecules_in_coord, Statistical_mechanics, NumAnalysis_step, min_RMS_gradient,
                                             Parameter_file=keyword_parameters['Parameter_file'],
                                             Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                             Volume_Reference=Volume_Reference, LocGrd_Vol_FracStep=LocGrd_Vol_FracStep)
            elif NumAnalysis_method == 'Euler':
                volume_gradient[i, 1], wavenumbers[i, 1:], volume = \
                    Ex.Call_Expansion(Method, 'local_gradient', Program, Output + '_' + Method + 'T' +
                                      str(temperature[i]) + file_ending, molecules_in_coord, min_RMS_gradient,
                                      Temperature=temperature[i], Pressure=Pressure,
                                      volume_fraction_change=LocGrd_Vol_FracStep,
                                      Statistical_mechanics=Statistical_mechanics,
                                      Parameter_file=keyword_parameters['Parameter_file'],
                                      Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                      Volume_Reference=Volume_Reference)

                volume_gradient[i, 2] = volume_gradient[i, 1]

        # Saving wavenumbers and local gradient information
        if Method == 'GiQ':
            np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])
        np.save(Output + '_dV_' + Method, volume_gradient[~np.all(volume_gradient == 0, axis=1)])

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                         wavenumbers[i, 1:], temperature[i], Pressure, Program, Statistical_mechanics,
                                         molecules_in_coord, keyword_parameters['cp2kroot'],
                                         Parameter_file=keyword_parameters['Parameter_file'])

        # Expanding to the next strucutre
        print "   Expanding to strucutre at: " + str(temperature[i + 1]) + " K"
        Ex.Call_Expansion(Method, 'expand', Program, Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                          molecules_in_coord, min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                          volume_fraction_change=(volume + volume_gradient[i, 1]*NumAnalysis_step)/volume,
                          Output=Output + '_' + Method + 'T' + str(temperature[i + 1]))
        os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending + ' Cords/')
        if temperature[i + 1] == temperature[-1]:
            print "   Determining local gradient and thermal properties at: " + str(temperature[i+1]) + " K"
            volume_gradient[i + 1, 2], wavenumbers[i+1, 1:], volume = \
                Ex.Call_Expansion(Method, 'local_gradient', Program, Output + '_' + Method + 'T' +
                                  str(temperature[i + 1]) + file_ending, molecules_in_coord, min_RMS_gradient,
                                  Temperature=temperature[i + 1], Pressure=Pressure,
                                  volume_fraction_change=LocGrd_Vol_FracStep,
                                  Statistical_mechanics=Statistical_mechanics,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                  Volume_Reference=Volume_Reference)
            properties[i+1, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending,
                                               wavenumbers[i + 1, 1:], temperature[i + 1], Pressure, Program,
                                               Statistical_mechanics, molecules_in_coord, keyword_parameters['cp2kroot'],
                                               Parameter_file=keyword_parameters['Parameter_file'])
            os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending + ' Cords/')
            if Method == 'GiQ':
                np.save(Output + '_WVN_' + Method, wavenumbers)
            np.save(Output + '_dV_' + Method, volume_gradient)

    # Saving the raw data before minimizing
    properties = Spline_Intermediate_Points(Output, Method, Program, properties, Temperature, molecules_in_coord,
                                      Pressure, Statistical_mechanics, min_RMS_gradient,
                                      Parameter_file=keyword_parameters['Parameter_file'],
                                      Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                      Volume_Reference=Volume_Reference)
    print "   All properties have been saved in " + Output + "_raw.npy"
    np.save(Output+'_raw', properties)
    return properties
        
########################################################
#     Gradient Anisotropic Expansion Due to Strain     #
########################################################
def Anisotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method, Gradient_MaxTemp,
                                   Pressure, LocGrd_NormStrain, LocGrd_ShearStrain, Statistical_mechanics,
                                   NumAnalysis_step, NumAnalysis_method, Aniso_LocGrad_Type, Temperature,
                                   min_RMS_gradient,
                                   **keyword_parameters):
    """
    This function calculated the anisotropic gradient for thermal expansion and returns the properties along that path
    :param Coordinate_file: file containing the lattice parameters (and coordinates)
    :param Program: 'Tinker' for Tinker Molecular Modeling
                    'Test' for a test run
    :param molecules_in_coord: number of molecules in coordinate file
    :param Output: string for outputted files
    :param Method: Gradient Anisotropic QHA ('GaQ');
                   Gradient Anisotropic QHA w/ Gruneisen Parameter ('GaQg');
    :param Gradient_MaxTemp: Maximum temperature in gradient method
    :param Pressure: in atm
    ***:param LocGrd_LatParam_FracStep: anisotropic crystal matrix fractional stepsize for local gradient
    :param Statistical_mechanics: 'Classical' Classical mechanics
                                  'Quantum' Quantum mechanics
    :param NumAnalysis_step: stepsize for numerical method
    :param NumAnalysis_method: 'RK4' Runge-Kutta 4th order
    :param Aniso_LocGrad_Type: 73 Hessians to calculate the complete anistropic gradient
                               25 for d**2G_dhdh only calculating the diagonals and off-diags. of the upper left 3x3 matrix
                               19 for d**2G_dhdh only calculating the uppder left 3x3 matrix
                               13 for d**2G_dhdh only calculating the diagonals
                               7  for d**2G_dhdh only calculating the upper left 3x3 matrix daigonals
    :param keyword_parameters: Parameter_file

    Optional Parameters
    Parameter_file: program specific file containing force field parameters
    Gruneisen_Lat_FracStep
    """
    # Setting file endings and determining how many wavenumbers there will be
    file_ending = Ex.assign_file_ending(Program)
    if Program == 'Tinker':
        number_of_wavenumbers = Pr.Tinker_atoms_per_molecule(Coordinate_file, 1) * 3
    elif Program == 'Test':
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file, True))
        keyword_parameters['Parameter_file'] = ''

    # Setting the temperature array
    temperature = np.arange(0, Gradient_MaxTemp + 1, NumAnalysis_step)

    # Setting the volume gradient array to be filled
    crystal_matrix_gradient = np.zeros((len(temperature), 2, 7))
    crystal_matrix_gradient[:, 0, 0] = temperature[:len(temperature)]
    if os.path.isfile(Output + '_dC_' + Method + '.npy'):
        crystal_matrix_gradient_hold = np.load(Output + '_dC_' + Method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(crystal_matrix_gradient_hold[:, 0, 0]) <= len(crystal_matrix_gradient[:, 0, 0]):
            if all(crystal_matrix_gradient_hold[:, 0, 0] == crystal_matrix_gradient[:len(crystal_matrix_gradient_hold[:, 0, 0]), 0, 0]):
                print "   Using lattice gradients in: " + Output + "_dC_" + Method + ".npy"
                crystal_matrix_gradient[:len(crystal_matrix_gradient_hold[:, 0, 0]), :, :] = crystal_matrix_gradient_hold

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_wavenumbers + 1))
    wavenumbers[:, 0] = temperature

    if Method == 'GaQg':
        # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
        Gruneisen, Wavenumber_Reference = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Output=Output,
                                                               Coordinate_file=Coordinate_file,
                                                               Parameter_file=keyword_parameters['Parameter_file'],
                                                               Program=Program, molecules_in_coord=molecules_in_coord,
                                                               Gruneisen_Lat_FracStep=
                                                               keyword_parameters['Gruneisen_Lat_FracStep'])

    elif Method == 'GaQ':
        # Setting parameters for Gruneisen parameter that won't be used (blank varaible)
        Gruneisen = 0.
        Wavenumber_Reference = 0.
        if os.path.isfile(Output + '_WVN_' + Method + '.npy'):
            # If a npy file of wavenumbers exist, pulling those in to use
            wavenumbers_hold = np.load(Output + '_WVN_' + Method + '.npy')
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
                    print "   Using wavenumbers already computed in: " + Output + "_WVN_" + Method + ".npy"
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Original coordinate file is used for 0K
    os.system('cp ' + Coordinate_file + ' ' + Output + '_' + Method + 'T' + str(temperature[0]) + file_ending)

    # Keeping track of the strain applied to the system [3 diagonal, 3 off-diagonal]
    ref_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file))

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        if (any(wavenumbers[i, 4:] != 0.) or (Method == 'GaQg')) and any(crystal_matrix_gradient[i, 0, 1:] != 0.):
            print "   Using previous data for the local gradient at: " + str(temperature[i]) + " K"
            if Method == 'GaQg':
                wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen, 
                                                          Wavenumber_Reference=Wavenumber_Reference, 
                                                          Coordinate_file=Output + '_' + Method + 'T' + 
                                                          str(temperature[i]) + file_ending,
                                                          ref_crystal_matrix=ref_crystal_matrix)

        else:
            print "   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K"
            # Using a numerical method to determine the gradient of the strains with thermal expansion
            if NumAnalysis_method == 'RK4':
                crystal_matrix_gradient[i, 0, 1:], wavenumbers[i, 1:], ignore, crystal_matrix_gradient[i, 1, 1:] = \
                        Runge_Kutta_Fourth_Order(Method, Output + '_' + Method + 'T' + str(temperature[i]) +
                                                 file_ending, Program, temperature[i], Pressure,
                                                 molecules_in_coord, Statistical_mechanics, NumAnalysis_step,
                                                 min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                                                 LocGrd_NormStrain=LocGrd_NormStrain,
                                                 LocGrd_ShearStrain=LocGrd_ShearStrain,
                                                 Aniso_LocGrad_Type=Aniso_LocGrad_Type, Gruneisen=Gruneisen,
                                                 Wavenumber_Reference=Wavenumber_Reference,
                                                 ref_crystal_matrix=ref_crystal_matrix)

            elif NumAnalysis_method == 'Euler':
                crystal_matrix_gradient[i, 0, 1:], wavenumbers[i, 1:] = \
                        Ex.Call_Expansion(Method, 'local_gradient', Program,
                                          Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                          molecules_in_coord, min_RMS_gradient, Temperature=temperature[i],
                                          Pressure=Pressure, LocGrd_NormStrain=LocGrd_NormStrain,
                                          LocGrd_ShearStrain=LocGrd_ShearStrain,
                                          Statistical_mechanics=Statistical_mechanics,
                                          Parameter_file=keyword_parameters['Parameter_file'], Gruneisen=Gruneisen,
                                          Wavenumber_Reference=Wavenumber_Reference,
                                          ref_crystal_matrix=ref_crystal_matrix,
                                          Aniso_LocGrad_Type=Aniso_LocGrad_Type)

                # Setting the local gradient equal to the step gradient
                crystal_matrix_gradient[i, 1, 1:] = crystal_matrix_gradient[i, 0, 1:]

            # Saving wavenumbers for non-Gruneisen methods
            if Method == 'GaQ':
                np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

            # Saving the strain gradient
            np.save(Output + '_dC_' + Method, crystal_matrix_gradient)

        # Changing the applied strain to the new expanded strucutre
        if os.path.isfile('Cords/' + Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending):
            subprocess.call(['cp', 'Cords/' + Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending, './'])
        else:
            # Expanding to the next strucutre using the strain gradient to the next temperature step
            Ex.Call_Expansion(Method, 'expand', Program,
                              Output + '_' + Method + 'T' + str(temperature[i]) + file_ending, molecules_in_coord,
                              min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                              dcrystal_matrix=Ex.array_to_triangle_crystal_matrix(crystal_matrix_gradient[i, 0, 1:]*NumAnalysis_step),
                              Output=Output + '_' + Method + 'T' + str(temperature[i + 1]))

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                         wavenumbers[i, 1:], temperature[i], Pressure, Program, Statistical_mechanics,
                                         molecules_in_coord, keyword_parameters['cp2kroot'],
                                         Parameter_file=keyword_parameters['Parameter_file'])

        # Moving the current strucutre to the Cords directory
        os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending + ' Cords/')

        if temperature[i + 1] == temperature[-1]:
            # Completing the run for the final structure
            if (any(wavenumbers[i + 1, 4:] != 0.) or (Method == 'GaQg')) and any(crystal_matrix_gradient[i + 1, 1, 1:] != 0.):
                print "   Using previous data for the local gradient at: " + str(temperature[i + 1]) + " K"
                if Method == 'GaQg':
                    wavenumbers[i + 1, 1:] = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen,
                                                                  Wavenumber_Reference=Wavenumber_Reference,
                                                                  ref_crystal_matrix=ref_crystal_matrix, Program=Program,
                                                                  Coordinate_file=Output + '_' + Method + 'T' +
                                                                  str(temperature[i + 1]) + file_ending)

            else:
                print "   Determining local gradient and thermal properties at: " + str(temperature[i + 1]) + " K"

                # Determinin the local gradient at the final structure (Used for finding intermediate temperatures)
                crystal_matrix_gradient[i + 1, 1, 1:], wavenumbers[i + 1, 1:] = \
                    Ex.Call_Expansion(Method, 'local_gradient', Program, Output + '_' + Method + 'T' +
                                      str(temperature[i + 1]) + file_ending, molecules_in_coord, min_RMS_gradient,
                                      Temperature=temperature[i + 1], Pressure=Pressure,
                                      LocGrd_NormStrain=LocGrd_NormStrain, LocGrd_ShearStrain=LocGrd_ShearStrain,
                                      Statistical_mechanics=Statistical_mechanics,
                                      Parameter_file=keyword_parameters['Parameter_file'], Gruneisen=Gruneisen,
                                      Wavenumber_Reference=Wavenumber_Reference, ref_crystal_matrix=ref_crystal_matrix,
                                      Aniso_LocGrad_Type=Aniso_LocGrad_Type)

            # Computing the properties at the final temperature
            properties[i + 1, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending,
                                                 wavenumbers[i + 1, 1:], temperature[i + 1], Pressure, Program,
                                                 Statistical_mechanics, molecules_in_coord, keyword_parameters['cp2kroot'],
                                                 Parameter_file=keyword_parameters['Parameter_file'])

            # Moving the final strucutre to the Cords directory
            os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending + ' Cords/')

            if Method == 'GaQ':
                # Saving the wavenumbers for the non-Gruneisen methods 
                np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

            # Saving the gradients for the temperature range
            np.save(Output + '_dC_' + Method, crystal_matrix_gradient)

    # Calculatin the properties for intermediate points
#    properties = Spline_Intermediate_Points(Output, Method, Program, properties, Temperature, molecules_in_coord,
#                                            Pressure, Statistical_mechanics, min_RMS_gradient,
#                                            Parameter_file=keyword_parameters['Parameter_file'],
#                                            Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,)

    # Saving the raw data before minimizing
    print "   All properties have been saved in " + Output + "_raw.npy"
    np.save(Output + '_raw', properties)
    return properties

