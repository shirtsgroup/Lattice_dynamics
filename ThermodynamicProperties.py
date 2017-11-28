#!/usr/bin/env python

import subprocess
import numpy as np
import itertools as it


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
    lattice_parameters = Test_Lattice_Parameters(Coordinate_file)
    U = (lattice_parameters[0] - 10) ** 2 + (lattice_parameters[1] - 7) ** 2 + \
        (lattice_parameters[2] - 12) ** 2 + 200.0 * (lattice_parameters[3] - 90) ** 2 + 100. * \
        (lattice_parameters[4] - 90) ** 2 + 150. * (lattice_parameters[5] - 90) ** 2 - 43.
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
    G = U + A + Pressure * volume * (6.022 * 10 ** 23) * (0.024201) * (10**(-27)) /molecules_in_coord
    return G


