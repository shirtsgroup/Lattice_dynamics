#!/usr/bin/env python

import subprocess
import numpy as np
import itertools as it
import Expand as Ex

##########################################
#           Export PROPERTIES            #
##########################################
def Properties(Coordinate_file, wavenumbers, Temperature, Pressure, Program, Statistical_mechanics, molecules_in_coord,
               **keyword_parameters):
    """
    Function to calculate all properties for a single temperature and pressure

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    Pressure = single Pressure in atm
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    Coordinate_file = File containing lattice parameters and atom coordinates
    Statistical_mechanics = 'Classical' Classical mechanics
                            'Quantum' Quantum mechanics
    molecules_in_coord = number of molecules in coordinate file

    **Optional Inputs
    Parameter_file = Optional input for program
    """

    properties = np.zeros(14)
    properties[0] = Temperature  # Temperature
    properties[1] = Pressure  # Pressure
    if Program == 'Tinker':
        properties[3] = Tinker_U(Coordinate_file, keyword_parameters['Parameter_file']) / molecules_in_coord
        # Potential energy
        properties[7:13] = Tinker_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    elif Program == 'Test':
        properties[3] = Test_U(Coordinate_file) / molecules_in_coord  # Potential energy
        properties[7:13] = Test_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    properties[6] = Volume(lattice_parameters=properties[7:13])  # Volume
    if Temperature != 0.:
        if Statistical_mechanics == 'Classical':
            properties[4] = Classical_Vibrational_A(Temperature,
                                                    wavenumbers) / molecules_in_coord  # Classical vibrational Helmholtz
            properties[13] = Classical_Vibrational_S(Temperature,
                                                     wavenumbers) / molecules_in_coord  # Classical vibrational Entropy
        if Statistical_mechanics == 'Quantum':
            properties[4] = Quantum_Vibrational_A(Temperature,
                                                  wavenumbers) / molecules_in_coord  # Quantum vibrational Helmholtz
            properties[13] = Quantum_Vibrational_S(Temperature,
                                                   wavenumbers) / molecules_in_coord  # Quantum vibrational Entropy
    properties[5] = Pressure * properties[6] * (6.022 * 10 ** 23) * (2.390 * 10 ** (-29)) / molecules_in_coord  # PV
    properties[2] = sum(properties[3:6])  # Gibbs free energy
    return properties


def Properties_with_Temperature(Coordinate_file, wavenumbers, Temperature, Pressure, Program, Statistical_mechanics,
                                molecules_in_coord, **keyword_parameters):
    """
    This function collects the properties for a specific coordinate file over a temperature range

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    Pressure = single Pressure in atm
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    Coordinate_file = File containing lattice parameters and atom coordinates
    Statistical_mechanics = 'Classical' Classical mechanics
                            'Quantum' Quantum mechanics
    molecules_in_coord = number of molecules in coordinate file

    **Optional Inputs
    Parameter_file = Optional input for program
    """
    properties = np.zeros((len(Temperature), 14))
    for i in range(len(Temperature)):
        if 'Parameter_file' in keyword_parameters:
            properties[i, :] = Properties(Coordinate_file, wavenumbers, Temperature[i], Pressure, Program,
                                          Statistical_mechanics, molecules_in_coord,
                                          Parameter_file=keyword_parameters['Parameter_file'])
        else:
            properties[i, :] = Properties(Coordinate_file, wavenumbers, Temperature[i], Pressure, Program,
                                          Statistical_mechanics, molecules_in_coord)
    return properties


def Save_Properties(properties, Properties_to_save, Output, Method, Statistical_mechanics):
    """
    Function for saving user specified properties

    **Required Inputs
    properties = matrix of properties to be input over a temperature range [Temperature_i,property]
    Properties_to_save = Array of properties to save ex. 'T,U,G'
                         'T' Temperature
                         'P' Pressure
                         'G' Gibbs free energy
                         'U' Potential energy
                         'Av' Helmholtz vibrational energy
                         'V' Volume
                         'h' Lattice parameters
                         'S' Entropy
    Output = string to start the output of each file
    Method = Harmonic approximation ('HA');
             Stepwise Isotropic QHA ('SiQ');
             Stepwise Isotropic QHA w/ Gruneisen Parameter ('SiQg');
             Gradient Isotropic QHA ('GiQ');
             Gradient Isotropic QHA w/ Gruneisen Parameter ('GiQg');
             Gradient Anisotropic QHA ('GaQ');
             Gradient Anistoropic QHA w/ Gruneisen Parameter ('GaQg');
    Statistical_mechanics = 'Classical' Classical mechanics
                            'Quantum' Quantum mechanics
    """
    for i in Properties_to_save:
        if i == 'T':  # Temperature
            print "   ... Saving temperature in: " + Output + "_T_" + Method + ".npy"
            np.save(Output + '_T_' + Method, properties[:, 0])
        if i == 'P':  # Pressure
            print "   ... Saving Pressure in: " + Output + "_P_" + Method + ".npy"
            np.save(Output + '_P_' + Method, properties[:, 1])
        if i == 'G':  # Gibbs free energy
            print "   ... Saving Gibbs free energy in: " + Output + "_G" + Statistical_mechanics + "_" + Method +\
                  ".npy"
            np.save(Output + '_G' + Statistical_mechanics + '_' + Method, properties[:, 2])
        if i == 'U':  # Potential energy
            print "   ... Saving potential energy in: " + Output + "_U" + Statistical_mechanics + "_" + Method + ".npy"
            np.save(Output + '_U' + Statistical_mechanics + '_' + Method, properties[:, 3])
        if i == 'Av':  # Helmholtz vibrational energy
            print "   ... Saving vibrational Helmholtz free energy in: " + Output + "_Av" + Statistical_mechanics + "_"\
                  + Method + ".npy"
            np.save(Output + '_Av' + Statistical_mechanics + '_' + Method, properties[:, 4])
        if i == 'V':  # Volume
            print "   ... Saving volume in: " + Output + "_V" + Statistical_mechanics + "_" + Method + ".npy"
            np.save(Output + '_V' + Statistical_mechanics + '_' + Method, properties[:, 6])
        if i == 'h':  # Lattice parameters
            print "   ... Saving lattice parameters in: " + Output + "_h" + Statistical_mechanics + "_" + Method +\
                  ".npy"
            np.save(Output + '_h' + Statistical_mechanics + '_' + Method, properties[:, 7:13])
        if i == 'S':  # Entropy
            print "   ... Saving entropy in: " + Output + "_S" + Statistical_mechanics + "_" + Method + ".npy"
            np.save(Output + '_S' + Statistical_mechanics + '_' + Method, properties[:, 14])


##########################################
#       TINKER MOLECULAR MODELING        #
##########################################
def Tinker_U(Coordinate_file, Parameter_file):
    """
    Calls the Tinker analyze executable and extracts the total potential energy
    ******Eventually! I want to also be able to extract individual potential energy terms
 
    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    Parameter_file = Tinker .key file specifying the force field parameter
    """
    U = float(subprocess.check_output(
        "analyze %s -k %s E | grep 'Total'| grep -oP '[-+]*[0-9]*\.[0-9]*'" % (Coordinate_file, Parameter_file),
        shell=True))
    return U


def Tinker_atoms_per_molecule(Coordinate_file, molecules_in_coord):
    """
    This function determines the number of atoms per molecule

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    molecules_in_coord = number of molecules in Coordinate_file
    """
    with open('%s' % Coordinate_file, 'r') as l:
        coordinates = [lines.split() for lines in l]
        coordinates = np.array(list(it.izip_longest(*coordinates, fillvalue=' '))).T
    atoms_per_molecule = int(coordinates[0, 0]) / molecules_in_coord
    return atoms_per_molecule


def Tinker_Lattice_Parameters(Coordinate_file):
    """
    This function extracts the lattice parameters from within the Tinker coordinate file 

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    """
    with open('%s' % Coordinate_file, 'r') as l:
        lattice_parameters = [lines.split() for lines in l]
        lattice_parameters = np.array(list(it.izip_longest(*lattice_parameters, fillvalue=' '))).T
        lattice_parameters = lattice_parameters[1, :6].astype(float)
    return lattice_parameters


##########################################
#                 TEST                   #
##########################################
def Test_U(Coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns the potential energy
    Random funcitons can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters
    """
    original_lp = np.array([6.7,12.7,8.1,90.,110.,90.])
    new_lp = np.load(Coordinate_file)

    original_matrix = Ex.Lattice_parameters_to_Crystal_matrix(original_lp)
    new_matrix = Ex.Lattice_parameters_to_Crystal_matrix(new_lp)

    strain = (new_matrix - original_matrix)
    strain = np.array([strain[0,0]/original_matrix[0,0], strain[1,1]/original_matrix[1,1], strain[2,2]/original_matrix[2,2], strain[0,1]/original_matrix[0,1], strain [0,2]/original_matrix[0,2], strain[1,2]/original_matrix[1,2]])

    polynomial_normal = np.array([[  1.18742203e+01,  -3.48680028e+02,   3.84692732e+03,  -1.88823969e+04,    3.45027603e+04],
                                  [  5.54323244e-01,  -3.08548327e+01,   6.47357935e+02,  -6.05503179e+03,    2.10055547e+04],
                                  [  1.95539366e-01,  -1.57382623e+01,   3.29163360e+02,  -2.65196646e+03,    7.14369363e+03],
                                  [ -1.06949138e-01,   3.85031065e+01,  -5.19373271e+03,   3.11110084e+05,   -6.98276745e+06],
                                  [ -8.31008319e-05,   3.53173349e-02,  -5.48454476e+00,   3.69411233e+02,   -9.36572066e+03],
                                  [ -8.55769073e+04,  -3.85214691e+06,   4.93079577e+05,   3.11812591e+10,    5.61260873e+12]])

    U = 0.
    for i in xrange(3):
        p = np.poly1d(polynomial_normal[i])
        U = U + p(new_lp[i])
#    for j in xrange(3):
#        p = np.poly1d(polynomial_shear[i])
#        U = U + p(np.cos(strain[i]))

    return U


def Test_Lattice_Parameters(Coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns them

    **Required Inputs
    Coordinate_file = File containing lattice parameters and atom coordinates
    """
    lattice_parameters = np.load(Coordinate_file)
    return lattice_parameters


##########################################
#           THERMO-PROPERTIES            #
##########################################
def Volume(**keyword_parameters):
    """
    This function either takes a coordinate file and determines the structures volume or takes the lattice parameters to
     calculate the volume

    **Optional Inputs
    lattice_parameters = crystal lattice parameters as an array [a,b,c,alpha,beta,gamma]
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run 
    Coordinate_file = File containing lattice parameters and atom coordinates
    """
    if 'lattice_parameters' in keyword_parameters:
        # Assigning the lattice parameters
        lattice_parameters = keyword_parameters['lattice_parameters']
    else:
        # Assigning the type of program and coordinate file
        program = keyword_parameters['Program']
        coordinate_file = keyword_parameters['Coordinate_file']
        if program == 'Tinker':
            # Retrieving lattice parameters of a tinker coordinate file
            lattice_parameters = Tinker_Lattice_Parameters(coordinate_file)
        elif program == 'Test':
            # Retrieving lattice parameters of a test coordinate file
            lattice_parameters = Test_Lattice_Parameters(coordinate_file)

    V = lattice_parameters[0] * lattice_parameters[1] * lattice_parameters[2] * np.sqrt(
        1 - np.cos(np.radians(lattice_parameters[3])) ** 2 - np.cos(np.radians(lattice_parameters[4])) ** 2 - np.cos(
            np.radians(lattice_parameters[5])) ** 2 + 2 * np.cos(np.radians(lattice_parameters[3])) * np.cos(
            np.radians(lattice_parameters[4])) * np.cos(np.radians(lattice_parameters[5])))
    return V


def Classical_Vibrational_A(Temperature, wavenumbers):
    """
    Function to calculate the Classical Helmholtz vibrational energy at a given temperature
    
    **Required Inputs
    Temperature = single temperature in Kelvin to determine the Helmholtz vibrational at (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    """
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-35)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-24)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    A = []
    for i in wavenumbers[3:]:  # Skipping the translational modes
        if i > 0:  # Skipping negative wavenumbers
            a = (1 / beta) * np.log(beta * h * i * c * 2 * np.pi) * Na / 1000
            A.append(a)
        else:
            pass
    A = sum(A)
    return A


def Quantum_Vibrational_A(Temperature, wavenumbers):
    """
    Function to calculate the Quantum Helmholtz vibrational energy at a given temperature

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the Helmholtz vibrational at (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    """
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-35)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-24)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    A = []
    for i in wavenumbers[3:]:  # Skipping translational modes
        if i > 0:  # Skipping negative wavenumbers
            a = ((h * i * c * np.pi) + (1 / beta) * np.log(1 - np.exp(-beta * h * i * c * 2 * np.pi))) * Na / 1000
            A.append(a)
        else:
            pass
    A = sum(A)
    return A

def Vibrational_Entropy(Temperature, wavenumbers, Statistical_mechanics):
    """
    Function to call the vibraitonal entropy based off of a specific statistical mechanics

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    Statistical_mechanics = 'Classical'
                            'Quantum'
    """
    if Statistical_mechanics == 'Classical':
        S = Classical_Vibrational_S(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        S = Quantum_Vibrational_S(Temperature, wavenumbers)
    return S


def Classical_Vibrational_S(Temperature, wavenumbers):
    """
    Funciton to calculate the classical vibrational entropy at a given temperature

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    """
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-35)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-24)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    S = []
    for i in wavenumbers[3:]:  # Skipping translational modes
        if i > 0:  # Skipping negative wavenumbers
            s = k * (1 - np.log(beta * h * i * c * 2 * np.pi)) * Na / 1000
            S.append(s)
        else:
            pass
    S = sum(S)
    return S


def Quantum_Vibrational_S(Temperature, wavenumbers):
    """
    Funciton to calculate the quantum vibrational entropy at a given temperature

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    """
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-35)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-24)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    S = []
    for i in wavenumbers[3:]:
        if i > 0:
            s = (h * i * c * 2 * np.pi / (Temperature * (np.exp(beta * h * i * c * 2 * np.pi) - 1)) - k * np.log(
                1 - np.exp(-beta * h * i * c * 2 * np.pi))) * Na / 1000
            S.append(s)
        else:
            pass
    S = sum(S)
    return S


def Gibbs_Free_Energy(Temperature, Pressure, Program, wavenumbers, Coordinate_file, Statistical_mechanics,
                      molecules_in_coord, **keyword_parameters):
    """
    Function to calculate the Gibbs free energy from the potential energy and vibrational Helmholtz free energy

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    Pressure = single Pressure in atm
    Program = 'Tinker' for Tinker Molecular Modeling
              'Test' for a test run
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    Coordinate_file = File containing lattice parameters and atom coordinates
    Statistical_mechanics = 'Classical' Classical mechanics
                            'Quantum' Quantum mechanics
    molecules_in_coord = number of molecules in coordinate file

    **Optional Inputs
    Parameter_file = Optional input for program
    """
    # Potential Energy
    if Program == 'Tinker':
        U = Tinker_U(Coordinate_file, keyword_parameters['Parameter_file']) / molecules_in_coord  # Potential Energy
    elif Program == 'Test':
        U = Test_U(Coordinate_file) / molecules_in_coord

    # Volume
    volume = Volume(Program=Program, Coordinate_file=Coordinate_file)

    # Helmholtz free energy
    if Statistical_mechanics == 'Classical':
        if Temperature != 0.:
            A = Classical_Vibrational_A(Temperature, wavenumbers) / molecules_in_coord
        else:
            A = 0.
    elif Statistical_mechanics == 'Quantum':
        A = Quantum_Vibrational_A(Temperature, wavenumbers) / molecules_in_coord 

    # Gibbs Free energy
# kcal/mol----    atm    * Ang^3  *   Avagadro's #     *kcal/(L*atm)*  (L/Ang^3)
    G = U + A + Pressure * volume * (6.022 * 10 ** 23) * (0.024201) * (10**(-27)) /molecules_in_coord
    return G


