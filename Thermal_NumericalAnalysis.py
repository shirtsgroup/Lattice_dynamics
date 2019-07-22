#!/usr/bin/env python
from __future__ import print_function
import os
import subprocess
import sys
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Numerical_Outputs as NO
import System_sensitivity as Ss
import shutil
import volume_constrained_minimization as vcm

##########################################
#           Numerical Methods            #
##########################################
def Runge_Kutta_Fourth_Order(inputs, coordinate_file, temperature, **keyword_parameters):
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
    file_ending = Ex.assign_file_ending(inputs.program)

    # Output of numerical analysis
    NO.start_RK(temperature, inputs.gradient_numerical_step)

    # Final array for weights on slopes
    RK_multiply = np.array([1. / 6., 1. / 3., 1. / 3., 1. / 6.])

    # Copying the coordinate file to a separate file to work with
    subprocess.call(['cp', coordinate_file, 'RK4' + file_ending])

    if inputs.program == 'QE':
        print(coordinate_file, 'copying bv file')
        os.system('cp ' + coordinate_file + 'bv' + ' RK4' + file_ending + 'bv')

    # Setting the different temperature stepsizes
    temperature_steps = np.array([0., inputs.gradient_numerical_step / 2., inputs.gradient_numerical_step / 2.,
                                  inputs.gradient_numerical_step])

    # Setting RK_4 array/matix and general parameters that aren't required for specific methods
    if (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
        # Setting array to save 4 gradients in dV/dT
        RK_grad = np.zeros(4)
    elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
        # Setting array to save 4 gradients for the six different strains d\eta/dT
        if inputs.anisotropic_type == '1D':
            RK_grad = np.zeros(6)
        else:
            RK_grad = np.zeros((4, 6))

    # Calculating the RK gradients for the overall numerical gradient
    for i in range(4):
        # Outputting numerical analysis
        NO.step_RK(i, temperature + temperature_steps[i], inputs.program, 'RK4' + file_ending)

        print("   + Performing Runge-Kutta step " + str(i + 1))
        if (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
            # Determining the slope at the current RK step
            RK_grad[i], wavenumbers_hold, volume_hold, left_minimum = \
                Ex.Call_Expansion(inputs, 'local_gradient', 'RK4' + file_ending, Temperature=temperature +
                                                                                             temperature_steps[i],
                                  LocGrd_dV=keyword_parameters['LocGrd_dV'], Gruneisen=keyword_parameters['Gruneisen'],
                                  Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                  Volume_Reference=keyword_parameters['Volume_Reference'])

        elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
            if inputs.anisotropic_type != '1D':
                # Determining the slope at the current RK step
                RK_grad[i], wavenumbers_hold, left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', 'RK4' + file_ending, Temperature=temperature +
                                                                                                 temperature_steps[i],
                                      LocGrd_dC=keyword_parameters['LocGrd_dC'],
                                      Gruneisen=keyword_parameters['Gruneisen'],
                                      Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                      ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'])
            else:
                # Determining the slope at the current RK step
                RK_grad[i], wavenumbers_hold, left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', 'RK4' + file_ending, Temperature=temperature +
                                                                                                 temperature_steps[i],
                                      LocGrd_dLambda=keyword_parameters['LocGrd_dLambda'],
                                      dC_dLambda=keyword_parameters['dC_dLambda'],
                                      Gruneisen=keyword_parameters['Gruneisen'],
                                      Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                      ref_crystal_matrix=keyword_parameters['ref_crystal_matrix'])
            volume_hold = 0.

        if i == 0:
            # Saving outputs to be passed to the earlier code (local gradient and wavenumbers of initial strucuture)
            wavenumbers = 1. * wavenumbers_hold
            volume = 1. * volume_hold
            k1 = 1. * RK_grad[0]
            if left_minimum == True:
                subprocess.call(['rm', 'RK4' + file_ending])
                return np.nan, np.nan, np.nan, np.nan

        if i != 3:
            if (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
                # For isotropic expansion, determining the volume fraction change of the input strucutre (V_new/V_input)
                volume_fraction_change = (volume + RK_grad[i] * temperature_steps[i + 1]) / volume

                # Expanding the crystal to the next step size
                Ex.Call_Expansion(inputs, 'expand', coordinate_file, volume_fraction_change=volume_fraction_change,
                                  output_file='RK4')

            elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
                if inputs.anisotropic_type != '1D':
                    # For anisotropic expansion, determining the strain of th input strucutre for the next step
                    RK_crystal_matrix = Ex.array_to_triangle_crystal_matrix(RK_grad[i] * temperature_steps[i + 1])
                else:
                    # For anisotropic expansion, determining the strain of th input strucutre for the next step
                    RK_crystal_matrix = Ex.array_to_triangle_crystal_matrix(RK_grad[i] * temperature_steps[i + 1] *
                                                                            keyword_parameters['dC_dLambda'])

                # Expanding the crystal to the next step size
                Ex.Call_Expansion(inputs, 'expand', coordinate_file, dcrystal_matrix=RK_crystal_matrix,
                                  output_file='RK4')

        # Multiplying the found gradient by the fraction it will contribute to the overall gradient
        RK_grad[i] = RK_grad[i] * RK_multiply[i]

    # Summing all RK gradients for the overall numerical gradient
    numerical_gradient = np.sum(RK_grad, axis=0)

    # Removing excess files
    subprocess.call(['rm', 'RK4' + file_ending])
    return numerical_gradient, wavenumbers, volume, k1

def RK_Dense_Output(theta, y_0, y_1, f_0, f_1, h):
    return (1 - theta) * y_0 + theta * y_1 + theta * (theta - 1) * ((1 - 2 * theta) * (y_1 - y_0) +
                                                                    (theta - 1) * h * f_0 + theta * h * f_1)


def Spline_Intermediate_Points(inputs, properties, **keyword_parameters):
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
    print("Using cubic spline to determine intermediate temperature steps.")
    # Setting file endings
    file_ending = Ex.assign_file_ending(inputs.program)

    # Making temperature array for wanted output values (user specified)
#NSA: This should probably be done when read in
    temperature = np.sort(np.unique(inputs.temperature))

    # Setting step points and tangents/gradients at those points
    if (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
        # Loading in the local gradients found form numerical analysis
        tangent = np.load(inputs.output + '_dV_' + inputs.method + '.npy')
        # Setting the isotropic volumes for the gradients loaded in
        y = properties[:, 6]
    elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
        # Loading in the local gradients found form numerical analysis
        if inputs.anisotropic_type == '1D':
            dLambda_dT = np.load(inputs.output + '_dLAMBDA_' + inputs.method + '.npy')
            tangent = np.zeros((len(dLambda_dT[:, 0]), 2, 7))
            dC_dT = np.load(inputs.output + '_dC_' + inputs.method + '.npy')
            for i in range(len(dLambda_dT[:, 0])):
                tangent[i, 0, 1:] = dC_dT * dLambda_dT[i, 1]
                tangent[i, 1, 1:] = dC_dT * dLambda_dT[i, 2] 
        else:
            tangent = np.load(inputs.output + '_dC_' + inputs.method + '.npy')
        # Assigning an array of crystal matricies at each temperature recorded
        y = np.zeros((len(properties[:, 0]), 6))

        for i in range(len(properties[:, 0])):
            # Pulling the crystal matrix from the original properites
            y[i] = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(properties[i, 7:13]))

    # Setting up an array to store the output properties
    properties_out = np.zeros((len(temperature), len(properties[0, :])))

    # Setting a count on where to place the properties in the output matrix
    count = 0

    # Setting variables for loop
    new_volume = 1.
    new_cm_array = np.ones(6)

    for i in range(len(properties[:, 0]) - 1):
        # Pulling up structure to expand from
        subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(properties[i, 0]) +
                         file_ending, './temp' + file_ending])

        # Stepsize used for numerical analysis
        h = properties[i + 1, 0] - properties[i, 0]

        # Adding properties found for lower bound temperature in Numerical analysis to output matrix

        for j in range(len(temperature)):
#            if temperature[j] == properties[i, 0]:
#                properties_out[count] = properties[i]
#                count += 1
#
            if (properties[i, 0] <= temperature[j] < properties[i + 1, 0]) or (temperature[j] == properties[i+1, 0] == properties[-1, 0]):
                print("   Using a Spline, adding intermediate temperature at:" + str(temperature[j]) + " K")
                # Computing the properties for all intermediate temperature points
                theta = (temperature[j] - properties[i, 0]) / h
                if (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
                    # Computing the volume we are currently at and the next intermediate step
                    new_volume = RK_Dense_Output(theta, y[i], y[i + 1], tangent[i, 2], tangent[i + 1, 2], h)
                elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
                    # Computing the strain from the lattice minimum structure to the next intermediate step
                    new_cm_array = np.zeros(6)
                    for k in range(6):
                        new_cm_array[k] = RK_Dense_Output(theta, y[i, k], y[i + 1, k], tangent[i, 1, k + 1],
                                                        tangent[i + 1, 1, k + 1], h)
    
                # Expanding the strucutre, from the structure at the last temperature to the next intermediate step
                if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[j]) +
                                          file_ending):
                    subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[j])
                                      + file_ending, 'hold' + file_ending])
                else:
                    Ex.Call_Expansion(inputs, 'expand', 'temp' + file_ending, volume_fraction_change=new_volume / y[i],
                                      dcrystal_matrix=Ex.array_to_triangle_crystal_matrix(new_cm_array - y[i]),
                                      output_file='hold')
    
                # Computing the wavenumbers for the new structure
                wavenumbers = Wvn.Call_Wavenumbers(inputs, Coordinate_file='hold' + file_ending,
                                                   Gruneisen=keyword_parameters['Gruneisen'],
                                                   Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                                   Volume_Reference=properties[0, 6], 
                                                   ref_crystal_matrix=Ex.Lattice_parameters_to_Crystal_matrix(properties[0, 7:13]),
                                                   New_Volume=new_volume)
    
                # Computing the properites
                properties_out[count] = Pr.Properties(inputs, 'hold' + file_ending, wavenumbers, temperature[j])

                # Moving new intermediate structure to the Cords directory for storage
                subprocess.call(['mv', 'hold' + file_ending, 'Cords/' + inputs.output + '_' + inputs.method + 'T' +
                                 str(temperature[j]) + file_ending])
    
                count += 1

    # Setting the last numerical step to the output matrix
    if properties[-1, 0] == temperature[-1]:
        properties_out[-1] = properties[-1]

    # Removing the temporary strucutre
    subprocess.call(['rm', 'temp' + file_ending])
    return properties_out


##########################################
#      Stepwise Isotropic Expansion      #
##########################################
def Isotropic_Stepwise_Expansion(inputs):
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
    runeisen_Vol_FracStep: volume fraction step used to determine the Gruneisen parameter 
    """
    # Setting file endings and determining how many wavenumbers there will be
    file_ending = Ex.assign_file_ending(inputs.program)
    number_of_modes = int(Pr.atoms_count(inputs.program, inputs.coordinate_file) * 3)

    # Setting up array of volume fractions from the lattice structure
    lower_volume_fraction = np.arange(inputs.stepwise_volume_fraction_lower, 1.0,
                                      inputs.stepwise_volume_fraction_stepsize)[::-1]
    if len(lower_volume_fraction) > 0:
        if lower_volume_fraction[0] != 1.0:
            lower_volume_fraction = np.insert(lower_volume_fraction, 0, 1.0)
    upper_volume_fraction = np.arange(1.0 + inputs.stepwise_volume_fraction_stepsize,
                                      inputs.stepwise_volume_fraction_upper, inputs.stepwise_volume_fraction_stepsize)
    volume_fraction = np.append(lower_volume_fraction, upper_volume_fraction)

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(volume_fraction), number_of_modes + 1))
    wavenumbers[:, 0] = volume_fraction

    # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
    if inputs.method == 'SiQg':
        print("   Calculating the isotropic Gruneisen parameter")
        gruneisen, wavenumber_reference, volume_reference = Wvn.Call_Wavenumbers(inputs)
    elif inputs.method == 'SiQ':
        gruneisen = 0.
        wavenumber_reference = 0.
        volume_reference = 0.
        if os.path.isfile(inputs.output + '_WVN_' + inputs.method + '.npy'):
            old_wavenumbers = np.load(inputs.output + '_WVN_' + inputs.method + '.npy')
            for i in range(len(old_wavenumbers[:, 0])):
                if any(volume_fraction == old_wavenumbers[i, 0]):
                    loc = np.where(volume_fraction == old_wavenumbers[i, 0])[0][0]
                    wavenumbers[loc] = old_wavenumbers[i]

    # setting a matrix for properties versus temperature and pressure
    properties = np.zeros((len(volume_fraction), len(inputs.temperature), 14))

    # Finding all expanded structures
    previous_volume = 1.0
    lattice_volume = Pr.Volume(Program=inputs.program, Coordinate_file=inputs.coordinate_file)

    if inputs.program == 'QE':
        lattfile = inputs.coordinate_file + 'bv' 
        newlattfile = inputs.output + '_' + inputs.method + str(previous_volume) + file_ending + 'bv'
        shutil.copyfile(lattfile, newlattfile)
    subprocess.call(['cp', inputs.coordinate_file, inputs.output + '_' + inputs.method + str(previous_volume) +
                     file_ending])

    for i in range(len(volume_fraction)):
        print("   Performing volume fraction of: " + str(volume_fraction[i]))
        if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + str(volume_fraction[i]) + file_ending):
            print("   ... Coordinate file Cords/" + inputs.output + "_" + inputs.method + str(volume_fraction[i]) +
                  file_ending + "already exists")

            # Skipping structures if they've already been constructed
            subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + str(volume_fraction[i]) +
                             file_ending, './'])
            if inputs.program == 'QE':
                os.system('cp Cords/' + inputs.output + '_' + inputs.method + str(volume_fraction[i]) + file_ending +
                          'bv' + ' ./')

        else:
            Ex.Call_Expansion(inputs, 'expand', inputs.output + '_' + inputs.method + str(previous_volume)
                              + file_ending, volume_fraction_change=(volume_fraction[i] / previous_volume),
                              output_file=inputs.output + '_' + inputs.method + str(volume_fraction[i]))

        # Calculating wavenumbers of new expanded strucutre
        find_wavenumbers = True
        if any(wavenumbers[i, 1:] > 0.):
            pass
        else:
            wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                      Wavenumber_Reference=wavenumber_reference,
                                                      Volume_Reference=volume_reference,
                                                      New_Volume=volume_fraction[i] * lattice_volume,
                                                      Coordinate_file=(inputs.output + '_' + inputs.method +
                                                                       str(volume_fraction[i]) + file_ending))

        # Saving the wavenumbers if the Gruneisen parameter is not being used
        if inputs.method == 'SiQ':
            print("   ... Saving wavenumbers in: " + inputs.output + "_WVN_" + inputs.method + ".npy")
            np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

        # Calculating properties of systems with wavenumbers above user specified tollerance
        if all(wavenumbers[i, 1:4] < inputs.wavenumber_tolerance) and all(wavenumbers[i, 1:4] > -1. * inputs.wavenumber_tolerance): 
            print("   ... Wavenumbers are greater than tolerance of: " + str(inputs.wavenumber_tolerance) + " cm^-1")
            properties[i, :, :] = Pr.Properties_with_Temperature(inputs, inputs.output + '_' + inputs.method +
                                                                 str(volume_fraction[i]) + file_ending,
                                                                 wavenumbers[i, 1:])

        else:
            print("   ... WARNING: wavenumbers are lower than tolerance of: " +
                  str(inputs.wavenumber_tolerance) + " cm^-1")
            print("      ... Properties will be bypassed for this paricular strucutre.")
            properties[i, :, :] = np.nan

    subprocess.call(['mv ' + inputs.output + '_' + inputs.method + '*' + file_ending + ' Cords/'], shell=True)
    if inputs.program == 'QE':
        os.system('mv ' + inputs.output + '_' + inputs.method + '*' + file_ending + 'bv' + ' Cords/')

    # Saving the raw data before minimizing
    print("   All properties have been saved in " + inputs.output + "_raw.npy")
    np.save(inputs.output + '_raw', properties)

    # Building matrix for minimum Gibbs Free energy data across temperature range
    minimum_gibbs_properties = np.zeros((len(inputs.temperature), 14))
    for i in range(len(properties[0, :, 0])):
        for j in range(len(properties[:, 0, 0])):
            if properties[j, i, 2] == np.nanmin(properties[:, i, 2]):
                minimum_gibbs_properties[i, :] = properties[j, i, :]
    return minimum_gibbs_properties

##########################################
#    generalized stepwise expansion      #
##########################################
def stepwise_expansion(inputs):
    # Setting file endings and determining how many wavenumbers there will be
    file_ending = Ex.assign_file_ending(inputs.program)
    number_of_modes = int(Pr.atoms_count(inputs.program, inputs.coordinate_file) * 3)

    # Setting up array of volume fractions from the lattice structure
    V0 = Pr.Volume(Program=inputs.program, Coordinate_file=inputs.coordinate_file)
    dV = inputs.stepwise_volume_fraction_stepsize * V0
    volumes = np.arange(inputs.stepwise_volume_fraction_lower * V0,inputs.stepwise_volume_fraction_upper * V0
                        + 0.000001, inputs.stepwise_volume_fraction_stepsize * V0)

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(volumes), number_of_modes + 1))
    wavenumbers[:, 0] = volumes
    placement = np.where(np.around(volumes, 3) == np.around(V0, 3))
    if inputs.method == 'SaQply':
        eigenvectors = np.zeros((len(volumes), number_of_modes, number_of_modes))
        wavenumbers_hold = \
            Wvn.Call_Wavenumbers(inputs, Coordinate_file=inputs.coordinate_file)

    # setting a matrix for properties versus temperature and pressure
    properties = np.zeros((len(volumes), len(inputs.temperature), 14))
    properties[placement, :] = Pr.Properties_with_Temperature(inputs, inputs.coordinate_file, wavenumbers_hold[0])

    for i in [-1, 1]:
        V = V0
        subprocess.call(['cp', inputs.coordinate_file, 'hold' + file_ending])
        while volumes[0] <= V + i * dV <= volumes[-1]:
            placement = np.where(np.around(V + i * dV, 3) == np.around(volumes, 3))
            if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4))
                              + file_ending):
                subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4))
                                 + file_ending, './'])
                subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4))
                                 + file_ending, 'hold' + file_ending])
            else:
                Ex.Call_Expansion(inputs, 'expand', 'hold' + file_ending, volume_fraction_change=(V + i * dV) / V,
                                  output_file=inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4)))
    
                if inputs.method == 'SaQply':
                    vcm.constrained_minimization(inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4))
                                                 + file_ending, inputs.program,
                                                 molecules_in_coord=inputs.number_of_molecules,
                                                 min_RMS_gradient=inputs.min_rms_gradient,
                                                 Parameter_file=inputs.tinker_parameter_file)
    
                subprocess.call(['cp', inputs.output + '_' + inputs.method + str(round((V + i * dV) / V0, 4))
                                 + file_ending, 'hold' + file_ending])

            wavenumbers_hold = Wvn.Call_Wavenumbers(inputs, Coordinate_file=inputs.output + '_' + inputs.method
                                                                            + str(round((V + i * dV) / V0, 4))
                                                                            + file_ending)
            if inputs.method == 'SaQply':
                wavenumbers[placement, 1:] = wavenumbers_hold[0]
                eigenvectors[placement] = wavenumbers_hold[1]
            else:
                wavenumbers[placement, 1:] = wavenumbers_hold
            properties[placement, :] = Pr.Properties_with_Temperature(inputs, inputs.output + '_' + inputs.method +
                                                                      str(round((V + i * dV) / V0, 4)) + file_ending,
                                                                      wavenumbers_hold[0])
            V = V + i * dV
    subprocess.call(['mv ' + inputs.output + '_' + inputs.method + '* ./Cords'], shell=True)
    subprocess.call(['rm', 'hold' + file_ending])

    np.save('out_raw', properties)
    if inputs.method == 'SaQply':
        print("polynomial fit is not working yet")
#NSA: This is working yet
#        minimum_gibbs_properties = Pr.polynomial_properties_optimize(properties[:, 0, 6], V0, wavenumbers, eigenvectors,
#                                                                     inputs.number_of_molecules,
#                                                                     inputs.statistical_mechanics, inputs.temperature,
#                                                                     inputs.pressure, inputs.eq_of_state,
#                                                                     inputs.poly_order, properties[:, 0, :],
#                                                                     inputs.output, inputs.program)
    else:
        minimum_gibbs_properties = np.zeros((len(inputs.temperature), 14))
        for i in range(len(properties[0, :, 0])):
            for j in range(len(properties[:, 0, 0])):
                if properties[j, i, 2] == np.min(properties[:, i, 2]):
                    minimum_gibbs_properties[i, :] = properties[j, i, :]
    return minimum_gibbs_properties



##########################################
#      Gradient Isotropic Expansion      #
##########################################
def Isotropic_Gradient_Expansion(inputs, LocGrd_dV):
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
    file_ending = Ex.assign_file_ending(inputs.program)
    number_of_modes = int(Pr.atoms_count(inputs.program, inputs.coordinate_file) * 3)

    # Setting the temperature array
    temperature = np.arange(0, inputs.gradient_max_temperature + 1., inputs.gradient_numerical_step)

    # Setting the volume gradient array to be filled
    volume_gradient = np.zeros((len(temperature), 3))
    volume_gradient[:, 0] = temperature[:len(temperature)]
    if os.path.isfile(inputs.output + '_dV_' + inputs.method + '.npy'):
        print("Using volume gradients in: " + inputs.output + "_dV_" + inputs.method + ".npy")
        volume_gradient_hold = np.load(inputs.output + '_dV_' + inputs.method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(volume_gradient_hold[:, 0]) <= len(volume_gradient[:, 0]):
            if all(volume_gradient_hold[:, 0] == volume_gradient[:len(volume_gradient_hold[:, 0]), 0]):
                volume_gradient[:len(volume_gradient_hold[:, 0]), :] = volume_gradient_hold

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_modes + 1))
    wavenumbers[:, 0] = temperature

    # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
    if inputs.method == 'GiQg':
        print("   Calculating the isotropic Gruneisen parameter")
        gruneisen, wavenumber_reference, volume_reference = Wvn.Call_Wavenumbers(inputs)
    elif inputs.method == 'GiQ':
        gruneisen = 0.
        wavenumber_reference = 0.
        volume_reference = 0.
        if os.path.isfile(inputs.output + '_WVN_' + inputs.method + '.npy'):
            wavenumbers_hold = np.load(inputs.output + '_WVN_' + inputs.method + '.npy')
            # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    print("Using wavenumbers previously computed in: " + inputs.output + "_WVN_" + inputs.method +
                          ".npy")
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Holding lattice structures as the structure at 0K
    subprocess.call(['cp', inputs.coordinate_file, inputs.output + '_' + inputs.method + 'T' + str(temperature[0]) +
                     file_ending])
    if inputs.program == 'QE':
        os.system('cp ' + inputs.coordinate_file + 'bv' + ' ' + inputs.output + '_' + inputs.method + 'T' +
                  str(temperature[0]) + file_ending + 'bv')

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        left_minimum = False
        print("   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K")
        if any(wavenumbers[i, 4:] != 0.) or inputs.method == 'GiQg' and (volume_gradient[i, 1] != 0.):
            print("   ... Using expansion gradient and wavenumbers previously found")
            volume = Pr.Volume(Program=inputs.program, Coordinate_file=inputs.output + '_' + inputs.method + 'T' +
                                                                       str(temperature[i]) + file_ending)
            if inputs.method == 'GiQg':
                wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                          Wavenumber_Reference=wavenumber_reference,
                                                          Volume_Reference=volume_reference, New_Volume=volume)
            pass
        else:
            if inputs.gradient_numerical_method == 'RK4':
                volume_gradient[i, 1], wavenumbers[i, 1:], volume, volume_gradient[i, 2] = \
                    Runge_Kutta_Fourth_Order(inputs, inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) +
                                             file_ending, temperature[i], Gruneisen=gruneisen,
                                             Wavenumber_Reference=wavenumber_reference,
                                             Volume_Reference=volume_reference, LocGrd_dV=LocGrd_dV)
                if np.isnan(volume_gradient[i, 1]):
                    left_minimum = True
            elif inputs.gradient_numerical_method == 'Euler':
                NO.gradient_output(temperature[i], inputs.program,  inputs.output + '_' + inputs.method + 'T' +
                                   str(temperature[i]) + file_ending)
                volume_gradient[i, 1], wavenumbers[i, 1:], volume, left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T' +
                                      str(temperature[i]) + file_ending, Temperature=temperature[i],
                                      LocGrd_dV=LocGrd_dV, Gruneisen=gruneisen,
                                      Wavenumber_Reference=wavenumber_reference, Volume_Reference=volume_reference)
                if left_minimum == True:
                    volume_gradient[i, 1:] = np.nan 
                    wavenumbers[i, 1:] = np.nan 
                    volume = np.nan
                else:
                    volume_gradient[i, 2] = volume_gradient[i, 1]

        # Exiting if the structure has left the minimum
        if left_minimum == True:
            inputs.gradient_max_temperature = temperature[i] - inputs.gradient_numerical_step
            inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
            print("Warning: system left minimum, stoping integration at: ", temperature[i], " K")
            properties = properties[:i]
            subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending, 
                             'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + "_NotAMin" + file_ending])
            break

        # Saving wavenumbers and local gradient information
        if inputs.method == 'GiQ':
            np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])
        np.save(inputs.output + '_dV_' + inputs.method, volume_gradient[~np.all(volume_gradient == 0, axis=1)])

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(inputs, inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) +
                                         file_ending, wavenumbers[i, 1:], temperature[i])

        # Expanding to the next strucutre
        if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending):
            print("   Using expanded structure in 'Cords/' at: " + str(temperature[i + 1]) + " K")
            subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) +
                             file_ending, './'])
        else:
            print("   Expanding to strucutre at: " + str(temperature[i + 1]) + " K")
            Ex.Call_Expansion(inputs, 'expand', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) +
                              file_ending, volume_fraction_change=(volume + volume_gradient[i, 1] *
                                                                   inputs.gradient_numerical_step) / volume,
                              output_file=inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]))
        subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending, 'Cords/'])

        if temperature[i + 1] == temperature[-1]:
            if any(wavenumbers[i + 1, 4:] != 0.) or inputs.method == 'GiQg' and (volume_gradient[i + 1, 2] != 0.):
                print("   ... Using expansion gradient and wavenumbers previously found")
                volume = Pr.Volume(Program=inputs.program, Coordinate_file=inputs.output + '_' + inputs.method + 'T' +
                                                                           str(temperature[i + 1]) + file_ending)
                if inputs.method == 'GiQg':
                    wavenumbers[i + 1, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                                  Wavenumber_Reference=wavenumber_reference,
                                                                  Volume_Reference=volume_reference, New_Volume=volume)
            else:
                print("   Determining local gradient and thermal properties at: " + str(temperature[i + 1]) + " K")
                NO.gradient_output(temperature[i + 1], inputs.program, inputs.output + '_' + inputs.method + 'T' +
                                   str(temperature[i + 1]) + file_ending)
                volume_gradient[i + 1, 2], wavenumbers[i+1, 1:], volume, left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T' +
                                      str(temperature[i + 1]) + file_ending, Temperature=temperature[i + 1],
                                      LocGrd_dV=LocGrd_dV, Gruneisen=gruneisen,
                                      Wavenumber_Reference=wavenumber_reference, Volume_Reference=volume_reference)
                if left_minimum == True:
                    inputs.gradient_max_temperature = temperature[i + 1] - inputs.gradient_numerical_step
                    inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
                    print("Warning: system left minimum, stoping integration at: ", temperature[i+1], " K")
                    properties = properties[:i+1]
                    subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                                     'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + "_NotAMin"+ file_ending])
                    break

            properties[i+1, :] = Pr.Properties(inputs, inputs.output + '_' + inputs.method + 'T' +
                                               str(temperature[i + 1]) + file_ending, wavenumbers[i + 1, 1:],
                                               temperature[i + 1])

            subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                             'Cords/'])
            if inputs.method == 'GiQ':
                np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers)
            np.save(inputs.output + '_dV_' + inputs.method, volume_gradient)

    # Saving the raw data before minimizing
    properties = Spline_Intermediate_Points(inputs, properties, Gruneisen=gruneisen,
                                            Wavenumber_Reference=wavenumber_reference,
                                            Volume_Reference=volume_reference)
    print("   All properties have been saved in " + inputs.output + "_raw.npy")
    np.save(inputs.output + '_raw', properties)
    return properties
        
########################################################
#     Gradient Anisotropic Expansion Due to Strain     #
########################################################
def Anisotropic_Gradient_Expansion(inputs, LocGrd_dC):
    # Setting file endings and determining how many wavenumbers there will be
    file_ending = Ex.assign_file_ending(inputs.program)
    number_of_modes = int(Pr.atoms_count(inputs.program, inputs.coordinate_file) * 3)

    # Setting the temperature array
    temperature = np.arange(0, inputs.gradient_max_temperature + 1, inputs.gradient_numerical_step)

    # Setting the volume gradient array to be filled
    crystal_matrix_gradient = np.zeros((len(temperature), 2, 7))
    crystal_matrix_gradient[:, 0, 0] = temperature[:len(temperature)]
    if os.path.isfile(inputs.output + '_dC_' + inputs.method + '.npy'):
        crystal_matrix_gradient_hold = np.load(inputs.output + '_dC_' + inputs.method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(crystal_matrix_gradient_hold[:, 0, 0]) <= len(crystal_matrix_gradient[:, 0, 0]):
            if all(crystal_matrix_gradient_hold[:, 0, 0] ==
                   crystal_matrix_gradient[:len(crystal_matrix_gradient_hold[:, 0, 0]), 0, 0]):
                print("   Using lattice gradients in: " + inputs.output + "_dC_" + inputs.method + ".npy")
                crystal_matrix_gradient[:len(crystal_matrix_gradient_hold[:, 0, 0]), :, :] = \
                    crystal_matrix_gradient_hold

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_modes + 1))
    wavenumbers[:, 0] = temperature

    if inputs.method == 'GaQg':
        # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
        gruneisen, wavenumber_reference = Wvn.Call_Wavenumbers(inputs)

    elif inputs.method == 'GaQ':
        # Setting parameters for Gruneisen parameter that won't be used (blank variable)
        gruneisen = 0.
        wavenumber_reference = 0.
        if os.path.isfile(inputs.output + '_WVN_' + inputs.method + '.npy'):
            # If a npy file of wavenumbers exist, pulling those in to use
            wavenumbers_hold = np.load(inputs.output + '_WVN_' + inputs.method + '.npy')
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
                    print("   Using wavenumbers already computed in: " + inputs.output + "_WVN_" + inputs.method
                          + ".npy")
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Original coordinate file is used for 0K
    subprocess.call(['cp', inputs.coordinate_file, inputs.output + '_' + inputs.method + 'T' + str(temperature[0]) +
                     file_ending])

    # Keeping track of the strain applied to the system [3 diagonal, 3 off-diagonal]
    ref_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(inputs.program,
                                                                                       inputs.coordinate_file))

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        left_minimum = False
        if (any(wavenumbers[i, 4:] != 0.) or (inputs.method == 'GaQg')) and \
                any(crystal_matrix_gradient[i, 0, 1:] != 0.):
            print("   Using previous data for the local gradient at: " + str(temperature[i]) + " K")
            if inputs.method == 'GaQg':
                wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                          Wavenumber_Reference=wavenumber_reference,
                                                          Coordinate_file=inputs.output + '_' + inputs.method + 'T' +
                                                          str(temperature[i]) + file_ending,
                                                          ref_crystal_matrix=ref_crystal_matrix)

        else:
            print("   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K")
            # Using a numerical method to determine the gradient of the strains with thermal expansion
            if inputs.gradient_numerical_method == 'RK4':
                crystal_matrix_gradient[i, 0, 1:], wavenumbers[i, 1:], ignore, crystal_matrix_gradient[i, 1, 1:] = \
                        Runge_Kutta_Fourth_Order(inputs, inputs.output + '_' + inputs.method + 'T' +
                                                 str(temperature[i]) + file_ending, temperature[i], LocGrd_dC=LocGrd_dC,
                                                 Gruneisen=gruneisen, Wavenumber_Reference=wavenumber_reference,
                                                 ref_crystal_matrix=ref_crystal_matrix)
                if np.isnan(ignore):
                    left_minimum = True
            elif inputs.gradient_numerical_method == 'Euler':
                NO.gradient_output(temperature[i], inputs.program, inputs.output + '_' + inputs.method + 'T' +
                                   str(temperature[i]) + file_ending)
                crystal_matrix_gradient[i, 0, 1:], wavenumbers[i, 1:], left_minimum = \
                        Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T' +
                                          str(temperature[i]) + file_ending, Temperature=temperature[i],
                                          LocGrd_dC=LocGrd_dC, Gruneisen=gruneisen,
                                          Wavenumber_Reference=wavenumber_reference,
                                          ref_crystal_matrix=ref_crystal_matrix)
                if left_minimum == True:
                    crystal_matrix_gradient[i, :, 1:] = np.nan
                    wavenumbers[i, 1:] = np.nan
                # Setting the local gradient equal to the step gradient
                else:
                    crystal_matrix_gradient[i, 1, 1:] = crystal_matrix_gradient[i, 0, 1:]

            # Exiting if the structure has left the minimum
            if left_minimum == True:
                nputs.gradient_max_temperature = temperature[i] - inputs.gradient_numerical_step
                inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
                print("Warning: system left minimum, stoping integration at: ", temperature[i], " K")
                properties = properties[:i]
                subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending,
                                 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + "_NotAMin"+ file_ending])
                break

            # Saving wavenumbers for non-Gruneisen methods
            if inputs.method == 'GaQ':
                np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

            # Saving the strain gradient
            np.save(inputs.output + '_dC_' + inputs.method, crystal_matrix_gradient)

        # Changing the applied strain to the new expanded structure
        if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending):
            print("   Using expanded structure in 'Cords/' at: " + str(temperature[i + 1]) + " K")
            subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) +
                             file_ending, './'])
        else:
            # Expanding to the next structure using the strain gradient to the next temperature step
            print("   Expanding to structure at: " + str(temperature[i + 1]) + " K")
            Ex.Call_Expansion(inputs, 'expand', inputs.output + '_' + inputs.method + 'T' + str(temperature[i])
                              + file_ending,
                              dcrystal_matrix=Ex.array_to_triangle_crystal_matrix(crystal_matrix_gradient[i, 0, 1:] *
                                                                                  inputs.gradient_numerical_step),
                              output_file=inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]))

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(inputs, inputs.output + '_' + inputs.method + 'T' + str(temperature[i])
                                         + file_ending, wavenumbers[i, 1:], temperature[i])

        # Moving the current structure to the Cords directory
        subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending, 'Cords/'])

        if temperature[i + 1] == temperature[-1]:
            # Completing the run for the final structure
            if (any(wavenumbers[i + 1, 4:] != 0.) or (inputs.method == 'GaQg')) and \
                    any(crystal_matrix_gradient[i + 1, 1, 1:] != 0.):
                print("   Using previous data for the local gradient at: " + str(temperature[i + 1]) + " K")
                if inputs.method == 'GaQg':
                    wavenumbers[i + 1, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                                  Wavenumber_Reference=wavenumber_reference,
                                                                  ref_crystal_matrix=ref_crystal_matrix,
                                                                  Coordinate_file=inputs.output + '_' + inputs.method
                                                                                  + 'T' + str(temperature[i + 1])
                                                                                  + file_ending)

            else:
                print("   Determining local gradient and thermal properties at: " + str(temperature[i + 1]) + " K")
                NO.gradient_output(temperature[i + 1], inputs.program, inputs.output + '_' + inputs.method + 'T'
                                   + str(temperature[i + 1]) + file_ending)
                # Determining the local gradient at the final structure (Used for finding intermediate temperatures)
                crystal_matrix_gradient[i + 1, 1, 1:], wavenumbers[i + 1, 1:], left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T' +
                                      str(temperature[i + 1]) + file_ending, Temperature=temperature[i + 1],
                                      LocGrd_dC=LocGrd_dC, Gruneisen=gruneisen,
                                      Wavenumber_Reference=wavenumber_reference, ref_crystal_matrix=ref_crystal_matrix)

                if left_minimum == True:
                    inputs.gradient_max_temperature = temperature[i+1] - inputs.gradient_numerical_step
                    inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
                    print("Warning: system left minimum, stoping integration at: ", temperature[i+1], " K")
                    properties = properties[:i+1]
                    subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                                     'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + "_NotAMin"+ file_ending])
                    break


            # Computing the properties at the final temperature
            properties[i + 1, :] = Pr.Properties(inputs, inputs.output + '_' + inputs.method + 'T'
                                                 + str(temperature[i + 1]) + file_ending, wavenumbers[i + 1, 1:],
                                                 temperature[i + 1])

            # Moving the final structure to the Cords directory
            subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                             'Cords/'])

            if inputs.method == 'GaQ':
                # Saving the wavenumbers for the non-Gruneisen methods 
                np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])

            # Saving the gradients for the temperature range
            np.save(inputs.output + '_dC_' + inputs.method, crystal_matrix_gradient)

    # Calculating the properties for intermediate points
    properties = Spline_Intermediate_Points(inputs, properties, Gruneisen=gruneisen,
                                            Wavenumber_Reference=wavenumber_reference)

    # Saving the raw data before minimizing
    print("   All properties have been saved in " + inputs.output + "_raw.npy")
    np.save(inputs.output + '_raw', properties)
    return properties



def Anisotropic_Gradient_Expansion_1D(inputs, LocGrd_dC):
    # Setting file endings and determining how many wavenumbers there will be
    file_ending = Ex.assign_file_ending(inputs.program)
    number_of_modes = int(Pr.atoms_count(inputs.program, inputs.coordinate_file) * 3)

    # Setting the temperature array
    temperature = np.arange(0, inputs.gradient_max_temperature + 1, inputs.gradient_numerical_step)

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_modes + 1))
    wavenumbers[:, 0] = temperature

    if inputs.method == 'GaQg':
        # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
        gruneisen, wavenumber_reference = Wvn.Call_Wavenumbers(inputs)

    elif inputs.method == 'GaQ':
        # Setting parameters for Gruneisen parameter that won't be used (blank variable)
        gruneisen = 0.
        wavenumber_reference = 0.
        if os.path.isfile(inputs.output + '_WVN_' + inputs.method + '.npy'):
            # If a npy file of wavenumbers exist, pulling those in to use
            wavenumbers_hold = np.load(inputs.output + '_WVN_' + inputs.method + '.npy')
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
                    print("   Using wavenumbers already computed in: " + inputs.output + "_WVN_" + inputs.method
                          + ".npy")
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Keeping track of the strain applied to the system [3 diagonal, 3 off-diagonal]
    ref_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(inputs.program,
                                                                                       inputs.coordinate_file))

    # Setting the volume gradient array to be filled
    if os.path.isfile(inputs.output + '_dC_' + inputs.method + '.npy'):
        print("   Using lattice gradients in: " + inputs.output + "_dC_" + inputs.method + ".npy")
        dC_dLambda = np.load(inputs.output + '_dC_' + inputs.method + '.npy')
    else:
        NO.gradient_output(temperature[0], inputs.program, inputs.coordinate_file)
        
        dC_dLambda, wavenumbers[0, 1:], left_minimum = Ex.Anisotropic_Local_Gradient(inputs, inputs.coordinate_file, 0., LocGrd_dC,
                                                                       Gruneisen=gruneisen,
                                                                       Wavenumber_Reference=wavenumber_reference,
                                                                       ref_crystal_matrix=ref_crystal_matrix)
        if left_minimum == True:
            print("ERROR: Starting structure is no longer at a minimum")
            sys.exit()

        np.save(inputs.output + '_dC_' + inputs.method, dC_dLambda)

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Original coordinate file is used for 0K
    subprocess.call(['cp', inputs.coordinate_file, inputs.output + '_' + inputs.method + 'T' + str(temperature[0])
                     + file_ending])

    LocGrd_dLambda = Ss.anisotropic_gradient_settings_1D(inputs, dC_dLambda)

    # Setting a place to store dLambda/dT
    dLambda_dT = np.zeros((len(temperature), 3))
    dLambda_dT[:, 0] = temperature
    if os.path.isfile(inputs.output + '_dLAMBDA_' + inputs.method + '.npy'):
        print("Using lambda gradients in: " + inputs.output + "_dLAMBDA_" + inputs.method + ".npy")
        dLambda_dT_hold = np.load(inputs.output + '_dLAMBDA_' + inputs.method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(dLambda_dT_hold[:, 0]) <= len(dLambda_dT[:, 0]):
            if all(dLambda_dT_hold[:, 0] == dLambda_dT[:len(dLambda_dT_hold[:, 0]), 0]):
                dLambda_dT[:len(dLambda_dT_hold[:, 0]), :] = dLambda_dT_hold

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        left_minimum = False
        print("   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K")
        if (any(wavenumbers[i, 4:] != 0.) or inputs.method == 'GaQg') and (dLambda_dT[i, 1] != 0.):
            print("   ... Using expansion gradient and wavenumbers previously found")
            if inputs.method == 'GaQg':
                wavenumbers[i + 1, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                              Wavenumber_Reference=wavenumber_reference,
                                                              ref_crystal_matrix=ref_crystal_matrix,
                                                              Coordinate_file=inputs.output + '_' + inputs.method + 'T'
                                                                              + str(temperature[i]) + file_ending)
            pass
        else:
            if inputs.gradient_numerical_method == 'RK4':
                dLambda_dT[i, 1], wavenumbers[i, 1:], ignore, dLambda_dT[i, 2] = \
                    Runge_Kutta_Fourth_Order(inputs, inputs.output + '_' + inputs.method + 'T' + str(temperature[i])
                                             + file_ending, temperature[i], Gruneisen=gruneisen,
                                             Wavenumber_Reference=wavenumber_reference,
                                             ref_crystal_matrix=ref_crystal_matrix, LocGrd_dLambda=LocGrd_dLambda,
                                             dC_dLambda=dC_dLambda)
                if np.isnan(ignore):
                    left_minimum = True
            elif inputs.gradient_numerical_method == 'Euler':
                NO.gradient_output(temperature[i], inputs.program,  inputs.output + '_' + inputs.method + 'T'
                                   + str(temperature[i]) + file_ending)
                dLambda_dT[i, 1], wavenumbers[i, 1:], left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T'
                                      + str(temperature[i]) + file_ending, Temperature=temperature[i],
                                      LocGrd_dLambda=LocGrd_dLambda, dC_dLambda=dC_dLambda,
                                      Gruneisen=gruneisen, Wavenumber_Reference=wavenumber_reference,
                                      ref_crystal_matrix=ref_crystal_matrix)
                if left_minimum == True:
                    dLambda_dT[i, 1:] = np.nan
                    wavenumbers[i, 1:] = np.nan
                else:
                    dLambda_dT[i, 2] = dLambda_dT[i, 1]

        # Exiting if the structure has left the minimum
        if left_minimum == True:
            inputs.gradient_max_temperature = temperature[i] - inputs.gradient_numerical_step
            inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
            print("Warning: system left minimum, stoping integration at: ", temperature[i], " K")
            properties = properties[:i]
            subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending,
                             'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + "_NotAMin"+ file_ending])
            break

        # Saving wavenumbers and local gradient information
        if inputs.method == 'GaQ':
            np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])
        np.save(inputs.output + '_dLAMBDA_' + inputs.method, dLambda_dT[~np.all(dLambda_dT == 0, axis=1)])

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(inputs,  inputs.output + '_' + inputs.method + 'T' + str(temperature[i])
                                         + file_ending, wavenumbers[i, 1:], temperature[i])

        # Expanding to the next strucutre
        if os.path.isfile('Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending):
            print("   Using expanded structure in 'Cords/' at: " + str(temperature[i + 1]) + " K")
            subprocess.call(['cp', 'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1])
                             + file_ending, './'])
        else:
            print("   Expanding to strucutre at: ", str(temperature[i + 1]), " K")
            Ex.Call_Expansion(inputs, 'expand', inputs.output + '_' + inputs.method + 'T' + str(temperature[i])
                              + file_ending,
                              dcrystal_matrix=Ex.array_to_triangle_crystal_matrix(inputs.gradient_numerical_step *
                                                                                  dLambda_dT[i, 1] * dC_dLambda),
                              output_file=inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]))

        subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i]) + file_ending, 'Cords/'])

        if temperature[i + 1] == temperature[-1]:
            if any(wavenumbers[i + 1, 4:] != 0.) or inputs.method == 'GaQg' and (dLambda_dT[i + 1, 2] != 0.):
                if inputs.method == 'GaQg':
                    wavenumbers[i + 1, 1:] = Wvn.Call_Wavenumbers(inputs, Gruneisen=gruneisen,
                                                                  Wavenumber_Reference=wavenumber_reference,
                                                                  ref_crystal_matrix=ref_crystal_matrix,
                                                                  Coordinate_file=inputs.output + '_' + inputs.method
                                                                                  + 'T' + str(temperature[i + 1])
                                                                                  + file_ending)
            else:
                print("   Determining local gradient and thermal properties at: " + str(temperature[i+1]) + " K")
                NO.gradient_output(temperature[i + 1], inputs.program, inputs.output + '_' + inputs.method + 'T'
                                   + str(temperature[i + 1]) + file_ending)
                dLambda_dT[i + 1, 2], wavenumbers[i+1, 1:], left_minimum = \
                    Ex.Call_Expansion(inputs, 'local_gradient', inputs.output + '_' + inputs.method + 'T'
                                      + str(temperature[i + 1]) + file_ending, Temperature=temperature[i + 1],
                                      LocGrd_dLambda=LocGrd_dLambda, dC_dLambda=dC_dLambda,
                                      Gruneisen=gruneisen, Wavenumber_Reference=wavenumber_reference,
                                      ref_crystal_matrix=ref_crystal_matrix)
                if left_minimum == True:
                    inputs.gradient_max_temperature = temperature[i+1] - inputs.gradient_numerical_step
                    inputs.temperature = inputs.temperature[np.where(inputs.temperature <= inputs.gradient_max_temperature)]
                    print("Warning: system left minimum, stoping integration at: ", temperature[i+1], " K")
                    properties = properties[:i+1]
                    subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                                     'Cords/' + inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + "_NotAMin"+ file_ending])
                    break
            properties[i+1, :] = Pr.Properties(inputs, inputs.output + '_' + inputs.method + 'T'
                                               + str(temperature[i + 1]) + file_ending, wavenumbers[i + 1, 1:],
                                               temperature[i + 1])
            subprocess.call(['mv', inputs.output + '_' + inputs.method + 'T' + str(temperature[i + 1]) + file_ending,
                             'Cords/'])
            if inputs.method == 'GaQ':
                np.save(inputs.output + '_WVN_' + inputs.method, wavenumbers)
            np.save(inputs.output + '_dLAMBDA_' + inputs.method, dLambda_dT)

    # Saving the raw data before minimizing
    properties = Spline_Intermediate_Points(inputs, properties, Gruneisen=gruneisen,
                                            Wavenumber_Reference=wavenumber_reference, ref_crystal_matrix=ref_crystal_matrix)
    print("   All properties have been saved in " + inputs.output + "_raw.npy")
    np.save(inputs.output+'_raw', properties)
    return properties
