#!/usr/bin/env python
import os
import sys
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn

##########################################
#           Numerical Methods            #
##########################################
def Runge_Kutta_Fourth_Order(Method, Coordinate_file, Program, Temperature, Pressure,
                             molecules_in_coord, Statistical_mechanics, RK4_stepsize, min_RMS_gradient, **keyword_parameters):
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
    if Program == 'Tinker':
        file_ending = '.xyz'
    elif Program == 'Test':
        file_ending = '.npy'
        keyword_parameters['Parameter_file'] = ''

    RK_multiply = np.array([1./6., 1./3., 1./3., 1./6.])

    # Copying the coordinate file to a seperate file to work with
    os.system('cp ' + Coordinate_file + ' RK4' + file_ending)

    # Setting the different temperature stepsizes
    temperature_steps = np.array([0., RK4_stepsize/2., RK4_stepsize/2., RK4_stepsize])

    # Setting RK_4 array/matix and general parameters that aren't required for specific methods
    if (Method == 'GiQ') or (Method == 'GiQg'):
        RK_grad = np.zeros(4)
        if Method == 'GiQ':
            keyword_parameters['Gruneisen'] = 0.
            keyword_parameters['Wavenumber_Reference'] = 0.
            keyword_parameters['Volume_Reference'] = 0.
            keyword_parameters['Crystal_matrix_Reference'] = 0.
    elif (Method == 'GaQ') or (Method == 'GaQg'):
        RK_grad = np.zeros((4, 3, 3))
        if Method == 'GaQ':
            keyword_parameters['Gruneisen'] = 0.
            keyword_parameters['Wavenumber_Reference'] = 0.
            keyword_parameters['Volume_Reference'] = 0.
            keyword_parameters['Crystal_matrix_Reference'] = 0.

    # Calculating the RK gradients for the overall numerical gradient
    for i in range(4):
        print "   + Performing Runge-Kutta step " + str(i + 1)
        if (Method == 'GiQ') or (Method == 'GiQg'):
            RK_grad[i], wavenumbers_hold, volume_hold = Ex.Call_Expansion(Method, 'local_gradient', Program,
                                                                          'RK4' + file_ending, molecules_in_coord,
                                                                          min_RMS_gradient,
                                                                          Temperature=Temperature, Pressure=Pressure,
                                                                          volume_fraction_change=keyword_parameters[
                                                                               'LocGrd_Vol_FracStep'],
                                                                          Statistical_mechanics=Statistical_mechanics,
                                                                          Parameter_file=
                                                                          keyword_parameters['Parameter_file'],
                                                                          Gruneisen=keyword_parameters['Gruneisen'],
                                                                          Wavenumber_Reference=
                                                                          keyword_parameters['Wavenumber_Reference'],
                                                                          Volume_Reference=
                                                                          keyword_parameters['Volume_Reference'])
        elif (Method == 'GaQ') or (Method == 'GaQg'):
            RK_grad[i], wavenumbers_hold = Ex.Call_Expansion(Method, 'local_gradient', Program, 'RK4' + file_ending,
                                                             molecules_in_coord, min_RMS_gradient, Temperature=Temperature,
                                                             Pressure=Pressure, matrix_parameters_fraction_change=
                                                             keyword_parameters['LocGrd_LatParam_FracStep'],
                                                             Statistical_mechanics=Statistical_mechanics,
                                                             Parameter_file=keyword_parameters['Parameter_file'],
                                                             Gruneisen=keyword_parameters['Gruneisen'],
                                                             Wavenumber_Reference=
                                                             keyword_parameters['Wavenumber_Reference'],
                                                             Crystal_matrix_Reference=
                                                             keyword_parameters['Crystal_matrix_Reference'],
                                                             Aniso_LocGrad_Type=
                                                             keyword_parameters['Aniso_LocGrad_Type'])
            volume_hold = 0.
        if i == 0:
            wavenumbers = 1.*wavenumbers_hold
            volume = 1.*volume_hold
            k1 = 1.*RK_grad[0]
        if i != 3:
            if (Method == 'GiQ') or (Method == 'GiQg'):
                dcrystal_matrix = 0.
                volume_fraction_change = (volume + RK_grad[i]*temperature_steps[i+1])/volume
            elif (Method == 'GaQ') or (Method == 'GaQg'):
                dcrystal_matrix = RK_grad[i]*temperature_steps[i+1]
                volume_fraction_change = 0.
            # Expanding the crystal to the next stepsize
            Ex.Call_Expansion(Method, 'expand', Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
                              Parameter_file=keyword_parameters['Parameter_file'], dcrystal_matrix=dcrystal_matrix,
                              volume_fraction_change=volume_fraction_change, Output='RK4')
        # Multiplying the found gradient by the fraction it will contribute to the overall gradient
        RK_grad[i] = RK_grad[i]*RK_multiply[i]
    # Summing all RK gradients for the overall numerical gradient
    numerical_gradient = np.sum(RK_grad, axis=0)

    # Removign excess files
    os.system('rm RK4'+file_ending)
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
    from scipy.interpolate import spline
    print "Using cubic spline to determine intermediate temperature steps."
    # Setting file endings
    if Program == 'Tinker':
        file_ending = '.xyz'
    elif Program == 'Test':
        file_ending = '.npy'
        keyword_parameters['Parameter_file'] = ''

    Temperature = np.sort(np.unique(np.append(Temperature, properties[:,0])))

    # Setting step points and tangents/gradients at those points
    if (Method == 'GiQ') or (Method == 'GiQg'):
        spline_points = np.zeros(len(Temperature))
        tangent = np.load(Output + '_dV_' + Method + '.npy')
    elif (Method == 'GaQ') or (Method == 'GaQg'):
        spline_points = np.zeros((len(Temperature), 3, 3))
        tangent = np.load(Output + '_dh_' + Method + '.npy')

    for i in range(len(Temperature)):
        if any(Temperature[i] == properties[:, 0]) != True:
            for j in range(len(properties[:, 0])):
                if properties[j, 0] < Temperature[i]:
                    lower_bound = j
                if properties[j, 0] > Temperature[i]:
                    upper_bound = j
                    break
            h = properties[upper_bound, 0] - properties[lower_bound, 0]
            theta = (Temperature[i] - properties[lower_bound, 0]) / h
            if (Method == 'GiQ') or (Method == 'GiQg'):
                spline_points[i] = RK_Dense_Output(theta, properties[lower_bound, 6], properties[upper_bound, 6],
                                                   tangent[lower_bound, 2], tangent[upper_bound, 2], h)
            if (Method == 'GaQ') or (Method == 'GaQg'):
                matrix_order = np.matrix([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]])
                lower_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(properties[lower_bound, 7:13])
                upper_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(properties[upper_bound, 7:13])
                for j in range(6):
                    spline_points[i, matrix_order[j, 0], matrix_order[j, 1]] = \
                        RK_Dense_Output(theta, lower_crystal_matrix[matrix_order[j, 0], matrix_order[j, 1]],
                                        upper_crystal_matrix[matrix_order[j, 0], matrix_order[j, 1]],
                                        tangent[lower_bound, matrix_order[j, 0], matrix_order[j, 1] + 4],
                                        tangent[upper_bound, matrix_order[j, 0], matrix_order[j, 1] + 4], h)

    for i in range(len(Temperature)):
        if any(Temperature[i] == properties[:, 0]) != True:
            print "   Adding in temperature point: " + str(Temperature[i]) + " K"
            for j in range(len(properties[:, 0])):
                if properties[j, 0] < Temperature[i]:
                    lower_bound = j
                if properties[j, 0] > Temperature[i]:
                    upper_bound = j
                    break

            if (Method == 'GiQ') or (Method == 'GiQg'):
                volume_fraction_change = spline_points[i]/properties[i - 1, 6]
                dcrystal_matrix = 0.
                keyword_parameters['Crystal_matrix_Reference'] = 0.
                if Method == 'GiQ':
                    keyword_parameters['Gruneisen'] = 0.
                    keyword_parameters['Volume_Reference'] = 0.
                    keyword_parameters['Wavenumber_Reference'] = 0.
            elif (Method == 'GaQ') or (Method == 'GaQg'):
                volume_fraction_change = 0.
                dcrystal_matrix = spline_points[i] - \
                                  Ex.Lattice_parameters_to_Crystal_matrix(properties[lower_bound, 7:13])
                keyword_parameters['Volume_Reference'] = 0.
                if Method == 'GaQ':
                    keyword_parameters['Gruneisen'] = 0.
                    keyword_parameters['Wavenumber_Reference'] = 0.
                    keyword_parameters['Crystal_matrix_Reference'] = 0.

#            Ex.Call_Expansion(Method, 'expand', Program, 'Cords/' + Output + '_' + Method + 'T' +
#                              str(properties[lower_bound, 0]) + file_ending, molecules_in_coord, min_RMS_gradient,
            Ex.Call_Expansion(Method, 'expand', Program, 'temp' + file_ending, molecules_in_coord, min_RMS_gradient,
                              Parameter_file=keyword_parameters['Parameter_file'],
                              volume_fraction_change=volume_fraction_change, dcrystal_matrix=dcrystal_matrix,
                              Output=Output + '_' + Method + 'T' + str(Temperature[i]))
            os.system('cp ' + Output + '_' + Method + 'T' + str(Temperature[i]) + file_ending + ' temp' + file_ending)
            wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program,
                                               Coordinate_file=Output + '_' + Method + 'T' + str(Temperature[i])
                                                               + file_ending,
                                               Parameter_file=keyword_parameters['Parameter_file'],
                                               Gruneisen=keyword_parameters['Gruneisen'],
                                               Wavenumber_Reference=keyword_parameters['Wavenumber_Reference'],
                                               Volume_Reference=keyword_parameters['Volume_Reference'],
                                               Crystal_matrix_Reference=keyword_parameters['Crystal_matrix_Reference'],
                                               New_Crystal_matrix=spline_points[i],
                                               New_Volume=spline_points[i])
            properties = np.insert(properties, upper_bound, Pr.Properties(Output + '_' + Method + 'T' +
                                                                                str(Temperature[i]) + file_ending,
                                                                                wavenumbers, Temperature[i], Pressure,
                                                                                Program, Statistical_mechanics,
                                                                                molecules_in_coord, Parameter_file=
                                                                                keyword_parameters['Parameter_file']),
                                   axis=0)
            os.system('mv ' + Output + '_' + Method + 'T' + str(Temperature[i]) + file_ending + ' Cords/')
        else:
            os.system('cp Cords/' + Output + '_' + Method + 'T' + str(Temperature[i]) + file_ending + ' ./temp' + file_ending)
    os.system('rm temp' + file_ending)
    return properties


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
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file))
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
    existing_wavenumbers = False
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
            existing_wavenumbers = True

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
        if existing_wavenumbers == True:
            for j in range(len(old_wavenumbers[:, 0])):
                if old_wavenumbers[j, 0] == volume_fraction[i]:
                    print "   ... Using wavenumbers already computed in: " + Output + "_WVN_" + Method + ".npy"
                    wavenumbers[i, 1:] = old_wavenumbers[j, 1:]
                    find_wavenumbers = False

        if find_wavenumbers == True:
            wavenumbers[i, 1:] = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program, Gruneisen=Gruneisen,
                                                      Wavenumber_Reference=Wavenumber_Reference,
                                                      Volume_Reference=Volume_Reference,
                                                      New_Volume=volume_fraction[i]*lattice_volume,
                                                      Coordinate_file=(Output + '_' + Method + str(volume_fraction[i])
                                                                       + file_ending),
                                                      Parameter_file=keyword_parameters['Parameter_file'], cp2kroot=keyword_parameters(['cp2kroot']))

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
                                                                 Parameter_file=keyword_parameters['Parameter_file'], cp2kroot = keyword_parameters(['cp2kroot']))
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
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file))
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
                                         molecules_in_coord, Parameter_file=keyword_parameters['Parameter_file'], cp2kroot=keywor_parameters['cp2kroot'])

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
                                               Statistical_mechanics, molecules_in_coord,
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
        
##########################################
#     Gradient Anisotropic Expansion     #
##########################################
def Ansotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method, Gradient_MaxTemp,
                                  Pressure, LocGrd_LatParam_FracStep, Statistical_mechanics,
                                  NumAnalysis_step, NumAnalysis_method, Aniso_LocGrad_Type, Temperature, min_RMS_gradient,
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
    :param LocGrd_LatParam_FracStep: anisotropic crystal matrix fractional stepsize for local gradient
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
    if Program == 'Tinker':
        file_ending = '.xyz'
        number_of_wavenumbers = Pr.Tinker_atoms_per_molecule(Coordinate_file, 1)*3
    elif Program == 'Test':
        file_ending = '.npy'
        number_of_wavenumbers = len(Wvn.Test_Wavenumber(Coordinate_file))
        keyword_parameters['Parameter_file'] = ''

    # Setting the temperature array
    temperature = np.arange(0, Gradient_MaxTemp + 1, NumAnalysis_step)

    # Setting the volume gradient array to be filled
    lattice_gradient = np.zeros((len(temperature), 3, 7))
    lattice_gradient[:, 0, 0] = temperature[:len(temperature)]
    if os.path.isfile(Output + '_dh_' + Method + '.npy'):
        lattice_gradient_hold = np.load(Output + '_dh_' + Method + '.npy')
        # If the temperatures line up, then the previous local gradients will be used
        if len(lattice_gradient_hold[:, 0, 0]) <= len(lattice_gradient[:, 0, 0]):
            if all(lattice_gradient_hold[:, 0, 0] == lattice_gradient[:len(lattice_gradient_hold[:, 0, 0]), 0, 0]):
                print "   Using lattice gradients in: " + Output + "_dh_" + Method + ".npy"
                lattice_gradient[:len(lattice_gradient_hold[:, 0, 0]), :, :] = lattice_gradient_hold

    # Setting up a matrix to store the wavenumbers in
    wavenumbers = np.zeros((len(temperature), number_of_wavenumbers+1))
    wavenumbers[:, 0] = temperature

    # Setting parameters for the Gruneisen parameter and loading in previously found wavenumbers for SiQ
    if Method == 'GaQg':
        Gruneisen, Wavenumber_Reference, Crystal_matrix_Reference = \
            Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Output=Output, Coordinate_file=Coordinate_file,
                                 Parameter_file=keyword_parameters['Parameter_file'], Program=Program,
                                 molecules_in_coord=molecules_in_coord, Gruneisen_Lat_FracStep=keyword_parameters['Gruneisen_Lat_FracStep'])
    elif Method == 'GaQ':
        Gruneisen = 0.
        Wavenumber_Reference = 0.
        Crystal_matrix_Reference = 0.
        if os.path.isfile(Output + '_WVN_' + Method + '.npy'):
            wavenumbers_hold = np.load(Output + '_WVN_' + Method + '.npy')
            # If the temperatures line up in the previous wavenumber matrix, it will be used in the current run
            if len(wavenumbers_hold[:, 0]) <= len(wavenumbers[:, 0]):
                if all(wavenumbers_hold[:, 0] == wavenumbers[:len(wavenumbers_hold[:, 0]), 0]):
                    print "   Using wavenumbers already computed in: " + Output + "_WVN_" + Method + ".npy"
                    wavenumbers[:len(wavenumbers_hold[:, 0]), :] = wavenumbers_hold

    # Setting up an array to store the properties
    properties = np.zeros((len(temperature), 14))

    # Holding lattice structures as the structure at 0K
    os.system('cp ' + Coordinate_file + ' ' + Output + '_' + Method + 'T' + str(temperature[0]) + file_ending)

    # Finding structures at higher temperatures
    for i in range(len(temperature) - 1):
        print "   Determining local gradient and thermal properties at: " + str(temperature[i]) + " K"
        if any(wavenumbers[i, 4:] != 0.) or Method == 'GaQg' and np.any(lattice_gradient[i, :, 1:4] != 0.) and \
                (os.path.isfile('Cords/' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending)):
            print "   ... Using expansion gradient and wavenumbers previously found"
            os.system('mv ' 'Cords/' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending + ' ./')
            if Method == 'GaQg':
                if Program == 'Tinker':
                    new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(
                        Pr.Tinker_Lattice_Parameters(Output + '_' + Method + 'T' + str(temperature[i]) + file_ending))
                elif Program == 'Test':
                    new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(
                        Pr.Test_Lattice_Parameters(Output + '_' + Method + 'T' + str(temperature[i]) + file_ending))
                wavenumbers[i, 4:] == Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen,
                                                           Wavenumber_Reference=Wavenumber_Reference,
                                                           Crystal_matrix_Reference=Crystal_matrix_Reference,
                                                           New_Crystal_matrix=new_crystal_matrix)
        else:
            if NumAnalysis_method == 'RK4':
                lattice_gradient[i, :, 1:4], wavenumbers[i, 1:], ignore, lattice_gradient[i, :, 4:] = \
                    Runge_Kutta_Fourth_Order(Method, Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                             Program, temperature[i], Pressure,
                                             molecules_in_coord, Statistical_mechanics, NumAnalysis_step, min_RMS_gradient,
                                             Parameter_file=keyword_parameters['Parameter_file'],
                                             LocGrd_LatParam_FracStep=LocGrd_LatParam_FracStep,
                                             Aniso_LocGrad_Type=Aniso_LocGrad_Type, Gruneisen=Gruneisen,
                                             Wavenumber_Reference=Wavenumber_Reference,
                                             Crystal_matrix_Reference=Crystal_matrix_Reference)
            elif NumAnalysis_method == 'Euler':
                lattice_gradient[i, :, 1:4], wavenumbers[i, 1:] = \
                    Ex.Call_Expansion(Method, 'local_gradient', Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
                                      Temperature=temperature[i], Pressure=Pressure,
                                      matrix_parameters_fraction_change=LocGrd_LatParam_FracStep,
                                      Statistical_mechanics=Statistical_mechanics,
                                      Parameter_file=keyword_parameters['Parameter_file'], Gruneisen=Gruneisen,
                                      Wavenumber_Reference=Wavenumber_Reference,
                                      Crystal_matrix_Reference=Crystal_matrix_Reference,
                                      Aniso_LocGrad_Type=Aniso_LocGrad_Type)
                lattice_gradient[i, :, 4:] = lattice_gradient[i, :, 1:4]

        # Saving wavenumbers and local gradient information
        if Method == 'GaQ':
            np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])
        np.save(Output + '_dh_' + Method, lattice_gradient)

        # Populating the properties for the current temperature
        properties[i, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                                         wavenumbers[i, 1:], temperature[i], Pressure, Program, Statistical_mechanics,
                                         molecules_in_coord, Parameter_file=keyword_parameters['Parameter_file'])

        # Expanding to the next strucutre
        Ex.Call_Expansion(Method, 'expand', Program, Output + '_' + Method + 'T' + str(temperature[i]) + file_ending,
                          molecules_in_coord, min_RMS_gradient, Parameter_file=keyword_parameters['Parameter_file'],
                          dcrystal_matrix=lattice_gradient[i, :, 1:4]*NumAnalysis_step,
                          Output=Output + '_' + Method + 'T' + str(temperature[i + 1]))

        os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i]) + file_ending + ' Cords/')
        if temperature[i + 1] == temperature[-1]:
            print "   Determining local gradient and thermal properties at: " + str(temperature[i+1]) + " K"
            lattice_gradient[i + 1, :, 4:], wavenumbers[i + 1, 1:] = \
                Ex.Call_Expansion(Method, 'local_gradient', Program, Output + '_' + Method + 'T' +
                                  str(temperature[i + 1]) + file_ending, molecules_in_coord, min_RMS_gradient,
                                  Temperature=temperature[i + 1], Pressure=Pressure,
                                  matrix_parameters_fraction_change=LocGrd_LatParam_FracStep,
                                  Statistical_mechanics=Statistical_mechanics,
                                  Parameter_file=keyword_parameters['Parameter_file'],
                                  Aniso_LocGrad_Type=Aniso_LocGrad_Type,
                                  Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                  Crystal_matrix_Reference=Crystal_matrix_Reference)

            properties[i + 1, :] = Pr.Properties(Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending,
                                                 wavenumbers[i + 1, 1:], temperature[i + 1], Pressure, Program,
                                                 Statistical_mechanics, molecules_in_coord,
                                                 Parameter_file=keyword_parameters['Parameter_file'])
            os.system('mv ' + Output + '_' + Method + 'T' + str(temperature[i + 1]) + file_ending + ' Cords/')
            if Method == 'GaQ':
                np.save(Output + '_WVN_' + Method, wavenumbers[~np.all(wavenumbers == 0, axis=1)])
            np.save(Output + '_dh_' + Method, lattice_gradient)

    # Saving the raw data before minimizing
    properties = Spline_Intermediate_Points(Output, Method, Program, properties, Temperature, molecules_in_coord,
                                      Pressure, Statistical_mechanics, min_RMS_gradient,
                                      Parameter_file=keyword_parameters['Parameter_file'],
                                      Gruneisen=Gruneisen, Wavenumber_Reference=Wavenumber_Reference,
                                      Crystal_matrix_Reference=Crystal_matrix_Reference)
    print "   All properties have been saved in " + Output + "_raw.npy"
    np.save(Output+'_raw', properties)
    return properties

