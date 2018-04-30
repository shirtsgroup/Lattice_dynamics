#!/usr/bin/env python

from __future__ import print_function
import subprocess
import numpy as np
import itertools as it

##########################################
#           Export PROPERTIES            #
##########################################
def Properties(Coordinate_file, wavenumbers, Temperature, Pressure, Program, Statistical_mechanics, molecules_in_coord, cp2kroot,
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
    elif Program == 'CP2K':
        properties[3] = CP2K_U() / molecules_in_coord  # Potential energy
        properties[7:13] = CP2K_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    elif Program == 'Test':
        properties[3] = Test_U(Coordinate_file) / molecules_in_coord  # Potential energy
        properties[7:13] = Test_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    properties[6] = Volume(lattice_parameters=properties[7:13])  # Volume
    properties[4] = Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics) / molecules_in_coord
    properties[13] = Vibrational_Entropy(Temperature, wavenumbers, Statistical_mechanics) / molecules_in_coord
    properties[5] = PV_energy(Pressure, properties[6]) / molecules_in_coord  # PV
    properties[2] = sum(properties[3:6])  # Gibbs free energy
    return properties


def Properties_with_Temperature(Coordinate_file, wavenumbers, Temperature, Pressure, Program, Statistical_mechanics,
                                molecules_in_coord, cp2kroot, **keyword_parameters):
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
                                          Statistical_mechanics, molecules_in_coord, cp2kroot,
                                          Parameter_file=keyword_parameters['Parameter_file'])
        else:
            properties[i, :] = Properties(Coordinate_file, wavenumbers, Temperature[i], Pressure, Program,
                                          Statistical_mechanics, molecules_in_coord, cp2kroot)
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
            print("   ... Saving temperature in: " + Output + "_T_" + Method + ".npy")
            np.save(Output + '_T_' + Method, properties[:, 0])
        if i == 'P':  # Pressure
            print("   ... Saving Pressure in: " + Output + "_P_" + Method + ".npy")
            np.save(Output + '_P_' + Method, properties[:, 1])
        if i == 'G':  # Gibbs free energy
            print("   ... Saving Gibbs free energy in: " + Output + "_G" + Statistical_mechanics + "_" + Method +\
                  ".npy")
            np.save(Output + '_G' + Statistical_mechanics + '_' + Method, properties[:, 2])
        if i == 'U':  # Potential energy
            print("   ... Saving potential energy in: " + Output + "_U" + Statistical_mechanics + "_" + Method + ".npy")
            np.save(Output + '_U' + Statistical_mechanics + '_' + Method, properties[:, 3])
        if i == 'Av':  # Helmholtz vibrational energy
            print("   ... Saving vibrational Helmholtz free energy in: " + Output + "_Av" + Statistical_mechanics + "_"\
                  + Method + ".npy")
            np.save(Output + '_Av' + Statistical_mechanics + '_' + Method, properties[:, 4])
        if i == 'V':  # Volume
            print("   ... Saving volume in: " + Output + "_V" + Statistical_mechanics + "_" + Method + ".npy")
            np.save(Output + '_V' + Statistical_mechanics + '_' + Method, properties[:, 6])
        if i == 'h':  # Lattice parameters
            print("   ... Saving lattice parameters in: " + Output + "_h" + Statistical_mechanics + "_" + Method +\
                  ".npy")
            np.save(Output + '_h' + Statistical_mechanics + '_' + Method, properties[:, 7:13])
        if i == 'S':  # Entropy
            print("   ... Saving entropy in: " + Output + "_S" + Statistical_mechanics + "_" + Method + ".npy")
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
        coordinates = np.array(list(it.zip_longest(*coordinates, fillvalue=' '))).T
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
        lattice_parameters = np.array(list(it.zip_longest(*lattice_parameters, fillvalue=' '))).T
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
    new_lp = np.load(Coordinate_file)
    U = Test_U_poly(new_lp)
    return U

def Test_U_poly(array):
#    # 4th order polynomial to describe how the potential energy changes as a function of the lattice parameters
#    # Given as [A, B, C, D, E]: A*x**4 + B*x**3 + C*x**2 + D*x + E
#                                  # Lattice vectors [Ang.]
#    polynomial_normal = np.array([[  1.32967100e+01,  -3.86230306e+02,   4.21859038e+03,  -2.05171077e+04,    3.71986258e+04],
#                                  [  5.96012607e-00,  -3.29643734e+01,   6.87371706e+02,  -6.39222328e+03,    2.20706869e+04],
#                                  [  6.54309591e-01,  -2.95928106e+01,   4.82192443e+02,  -3.37907613e+03,    8.38105955e+03],
#                                  # Lattice Angles [Degrees]
#                                  [  1.73101682e-04,  -6.23169023e-02,   8.53269922e+00,  -5.26349633e+02,    1.20653441e+04],
#                                  [  1.87342875e-04,  -6.45384256e-02,   8.09885843e+00,  -4.58100859e+02,    1.27492294e+04],
#                                  [ -7.85675953e-05,   2.82845448e-02,  -3.66590698e+00,   2.01651921e+02,   -4.18251942e+03]])
#    U = 0.
#    for i in range(6):
#        p = np.poly1d(polynomial_normal[i])
#        U = U + p(array[i])
    # Energy returned in kcal/mol



    p = np.array([  4.23707763e+03,  -4.84369902e+02,  -4.14736558e+02,  -3.52679881e+02,
                   -3.56924620e+00,   1.84204932e+01,   2.91373650e+00,   3.39706395e+01,
                    1.43083158e+01,   1.29649063e+01,   2.44591565e-02,   1.48827075e-01,
                    2.43687850e-02,   1.79715135e+01,   2.30779870e+01,   8.31777610e-02,
                   -3.37390151e+00,  -3.77035091e-01,   1.41456418e+01,   6.39700536e-02,
                   -8.04076344e-01,  -7.14652248e-02,  -1.39556833e-01,  -1.75045683e+00,
                   -3.25448447e-02,   7.63592279e-03,  -1.32279455e-02,  -2.32066032e-02])

    U = p[0] + array[0]*p[1] + array[1]*p[2] + array[2]*p[3] + array[3]*p[4] + array[4]*p[5] + array[5]*p[6] \
        + array[0]**2*p[7] + array[1]**2*p[8] + array[2]**2*p[9] + array[3]**2*p[10] + array[4]**2*p[11] + array[5]**2*p[12] \
        + array[0]*array[1]*p[13] + array[0]*array[2]*p[14] + array[0]*array[3]*p[15] + array[0]*array[4]*p[16] + array[0]*array[5]*p[17] \
        + array[1]*array[2]*p[18] + array[1]*array[3]*p[19] + array[1]*array[4]*p[20] + array[1]*array[5]*p[21] \
        + array[2]*array[3]*p[22] + array[2]*array[4]*p[23] + array[2]*array[5]*p[24] \
        + array[3]*array[4]*p[25] + array[3]*array[5]*p[26] \
        + array[4]*array[5]*p[27]
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
#                 CP2K                   #
##########################################

def CP2K_U():
    """
    This function takes a set of lattice parameters in a .npy file and returns the potential energy
    Random funcitons can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters
    """
    l = open('NMA-r-0.out')
    lines = l.readlines()
    for x in range(0,len(lines)):
        if 'ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): ' in lines[x]:
            U = float(lines[x].split()[-1])*627.5	   
    return U


def CP2K_Lattice_Parameters(Coordinate_file):
    """
    This function extracts the lattice parameters from within the Tinker coordinate file 

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    """
    with open('%s' % Coordinate_file, 'r') as l:
        lines = l.readlines()
        lattice_parameterstemp = (lines[1].split()[1:7])
    lattice_parameters = np.zeros((6,))
    for x in range(0,6):
        lattice_parameters[x] = float(lattice_parameterstemp[x])
    return lattice_parameters

def CP2K_atoms_per_molecule(Coordinate_file, molecules_in_coord):
    """
    This function determines the number of atoms per molecule

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    molecules_in_coord = number of molecules in Coordinate_file
    """
    numatoms = 0
    with open('%s' % Coordinate_file, 'r') as l:
        lines = l.readlines()
        for line in lines:
            if line.split()[0] == 'ATOM':
                numatoms+=1
    atoms_per_molecule = numatoms / molecules_in_coord
    return atoms_per_molecule


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
        elif program == 'CP2K':
            # Retrieving lattice parameters of a test coordinate file
            lattice_parameters = CP2K_Lattice_Parameters(coordinate_file)

    V = lattice_parameters[0] * lattice_parameters[1] * lattice_parameters[2] * np.sqrt(
        1 - np.cos(np.radians(lattice_parameters[3])) ** 2 - np.cos(np.radians(lattice_parameters[4])) ** 2 - np.cos(
            np.radians(lattice_parameters[5])) ** 2 + 2 * np.cos(np.radians(lattice_parameters[3])) * np.cos(
            np.radians(lattice_parameters[4])) * np.cos(np.radians(lattice_parameters[5])))
    return V


def Potential_energy(Program, **keyword_parameters):
    if Program == 'Tinker':
        U = Tinker_U(keyword_parameters['Coordinate_file'], keyword_parameters['Parameter_file'])
    elif Program == 'Test':
        U = Test_U(keyword_parameters['Coordinate_file'])
    elif Program == 'CP2K':
        U = CP2K_U()
    return U

def Lattice_parameters(Program, Coordinate_file):
    if Program == 'Tinker':
        lattice_parameters = Tinker_Lattice_Parameters(Coordinate_file)
    elif Program == 'Test':
        lattice_parameters = Test_Lattice_Parameters(Coordinate_file)
    elif Program == 'CP2K':
        lattice_parameters = CP2K_Lattice_Parameters(Coordinate_file)
    return lattice_parameters

def RotationFree_StrainArray_from_CrystalMatrix(ref_crystal_matrix, new_crystal_matrix):
    # The deformation matrix is C = F*C0 --> F = C*C0^-1
    F = np.dot(new_crystal_matrix, np.linalg.inv(ref_crystal_matrix))

    # Computing necessary variables
    C = np.dot(F.T, F)
    F_eig = np.linalg.eig(F)[0]
    I_U = np.sum(F_eig)
    II_U = F_eig[0]*F_eig[1] + F_eig[1]*F_eig[2] + F_eig[2]*F_eig[0]
    III_U = np.product(F_eig)

    # F = R*U, where R is the rotational piece
    # For us, R is simply an artifact of computing between Cartesian coordinates
    U = np.dot(np.linalg.inv(C + II_U*np.identity(3)), I_U*C +III_U*np.identity(3))

    # Computing the strain on the crystal due to the non-rotational deformation matrix
    # We assume the relationship can be described with small strains
    eta = 0.5*(U + U.T) - np.identity(3)
    eta = np.array([eta[0, 0], eta[1, 1], eta[2, 2],
                    eta[0, 1], eta[0, 2], eta[1, 2]])
    for i in range(6):
        if np.absolute(eta[i]) < 1e-10:
            eta[i] = 0.
    return eta


def PV_energy(Pressure, volume):
    return Pressure * volume * (6.022 * 10 ** 23) * (0.024201) * (10**(-27)) 

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
            if Temperature == 0:
                a = ((h * i * c * np.pi) * Na / 1000 )
            else:
                a = ((h * i * c * np.pi) + (k * Temperature) * np.log(1 - np.exp(-beta * h * i * c * 2 * np.pi))) * Na / 1000
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
    if Statistical_mechanics == 'Classical' and T != 0.:
        S = Classical_Vibrational_S(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        S = Quantum_Vibrational_S(Temperature, wavenumbers)
    return S

def Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics):
    if Statistical_mechanics == 'Classical' and T != 0.:
        Av = Classical_Vibrational_A(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        Av = Quantum_Vibrational_A(Temperature, wavenumbers)
    return Av


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
    elif Program == 'CP2K':
        U = CP2K_U()
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
    G = U + A + PV_energy(Pressure, volume) / molecules_in_coord
    return G, U, A


