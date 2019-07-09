#!/usr/bin/env python

from __future__ import print_function
import subprocess
import numpy as np
import itertools as it
import Expand as Ex
import Wavenumbers as Wvn
import equations_of_state as eos
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##########################################
#           Export PROPERTIES            #
##########################################
def Properties(inputs, Coordinate_file, wavenumbers, Temperature):
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
    properties[1] = inputs.pressure  # Pressure
    if inputs.program == 'Tinker':
        properties[3] = Tinker_U(Coordinate_file, inputs.tinker_parameter_file) / inputs.number_of_molecules
        # Potential energy
        properties[7:13] = Tinker_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    elif inputs.program == 'CP2K':
        properties[3] = CP2K_U(Coordinate_file) / inputs.number_of_molecules  # Potential energy
        properties[7:13] = CP2K_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    elif inputs.program == 'QE':
        properties[3] = QE_U(Coordinate_file) / inputs.number_of_molecules
        properties[7:13], matrix = QE_Lattice_Parameters(Coordinate_file)
    elif inputs.program == 'Test':
        properties[3] = Test_U(Coordinate_file) / inputs.number_of_molecules  # Potential energy
        properties[7:13] = Test_Lattice_Parameters(Coordinate_file)  # Lattice parameters
    properties[6] = Volume(lattice_parameters=properties[7:13])  # Volume
    properties[4] = Vibrational_Helmholtz(Temperature, wavenumbers, inputs.statistical_mechanics) / \
                    inputs.number_of_molecules
    properties[13] = Vibrational_Entropy(Temperature, wavenumbers, inputs.statistical_mechanics) / \
                     inputs.number_of_molecules
    properties[5] = PV_energy(inputs.pressure, properties[6]) / inputs.number_of_molecules  # PV
    properties[2] = sum(properties[3:6])  # Gibbs free energy
    return properties


def Properties_with_Temperature(inputs, Coordinate_file, wavenumbers):
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
    properties = np.zeros((len(inputs.temperature), 14))
    properties[0, :] = Properties(inputs, Coordinate_file, wavenumbers, inputs.temperature[0])
    properties[:, 0] = inputs.temperature
    properties[:, 1] = properties[0, 1]
    properties[:, 3] = properties[0, 3]
    properties[:, 7:13] = properties[0, 7:13]
    properties[:, 6] = properties[0, 6]
    properties[:, 5] = properties[0, 5]
    for i in range(1, len(inputs.temperature)):
        properties[i, 4] = Vibrational_Helmholtz(inputs.temperature[i], wavenumbers, inputs.statistical_mechanics) / \
                           inputs.number_of_molecules
        properties[i, 13] = Vibrational_Entropy(inputs.temperature[i], wavenumbers, inputs.statistical_mechanics) / \
                            inputs.number_of_molecules
        properties[i, 2] = sum(properties[i, 3:6])
    return properties


def Save_Properties(inputs, properties):#properties, Properties_to_save, Output, Method, Statistical_mechanics):
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
    if 'T' in inputs.properties_to_save:  # Temperature
        print("   ... Saving temperature in: " + inputs.output + "_T_" + inputs.method + ".npy")
        np.save(inputs.output + '_T_' + inputs.method, properties[:, 0])
    if 'P' in inputs.properties_to_save:  # Pressure
        print("   ... Saving Pressure in: " + inputs.output + "_P_" + inputs.method + ".npy")
        np.save(inputs.output + '_P_' + inputs.method, properties[:, 1])
    if 'G' in inputs.properties_to_save:  # Gibbs free energy
        print("   ... Saving Gibbs free energy in: " + inputs.output + "_G" + inputs.statistical_mechanics + "_" +
              inputs.method + ".npy")
        np.save(inputs.output + '_G' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 2])
    if 'U' in inputs.properties_to_save:  # Potential energy
        print("   ... Saving potential energy in: " + inputs.output + "_U" + inputs.statistical_mechanics + "_" +
              inputs.method + ".npy")
        np.save(inputs.output + '_U' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 3])
    if 'Av' in inputs.properties_to_save:  # Helmholtz vibrational energy
        print("   ... Saving vibrational Helmholtz free energy in: " + inputs.output + "_Av" +
              inputs.statistical_mechanics + "_" + inputs.method + ".npy")
        np.save(inputs.output + '_Av' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 4])
    if 'V' in inputs.properties_to_save:  # Volume
        print("   ... Saving volume in: " + inputs.output + "_V" + inputs.statistical_mechanics + "_" +
              inputs.method + ".npy")
        np.save(inputs.output + '_V' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 6])
    if 'h' in inputs.properties_to_save:  # Lattice parameters
        print("   ... Saving lattice parameters in: " + inputs.output + "_h" + inputs.statistical_mechanics +
              "_" + inputs.method + ".npy")
        np.save(inputs.output + '_h' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 7:13])
    if 'S' in inputs.properties_to_save:  # Entropy
        print("   ... Saving entropy in: " + inputs.output + "_S" + inputs.statistical_mechanics + "_" +
              inputs.method + ".npy")
        np.save(inputs.output + '_S' + inputs.statistical_mechanics + '_' + inputs.method, properties[:, 13])


def polynomial_properties_optimize(volumes, V0, wavenumbers, eigenvectors, molecules_in_coord, Statistical_mechanics,
                                   Temperature, Pressure, eq_of_state, poly_order, prop_0K, Output, Program):
    # Organize vibrational modes
    wavenumbers_organized = np.zeros((len(volumes), len(wavenumbers[0, 1:])))
    basis_placement = np.where(np.around(V0, 3) == np.around(volumes, 3))[0][0]
    number_of_modes = len(wavenumbers[0, 1:])
    for i in range(len(volumes)):
        if Program == 'Test':
            wavenumbers_organized[i] = wavenumbers[i, 1:]
        else:
            z, _ = Wvn.matching_eigenvectors_of_modes(number_of_modes, eigenvectors[basis_placement], eigenvectors[i])
            wavenumbers_organized[i] = Wvn.reorder_modes(z, wavenumbers[i, 1:])

    # Create polynomials for vibrational modes as a function of volume
    wavenumber_poly = np.zeros((number_of_modes, poly_order + 1))
    for i in range(number_of_modes):
        wavenumber_poly[i] = np.polyfit(volumes, wavenumbers_organized[:, i], poly_order)

    # Create polynomials for lattice parameters as a function of volume
    lattice_parameter_poly = np.zeros((6, poly_order + 1))
    for i in range(6):
        lattice_parameter_poly[i] = np.polyfit(volumes, prop_0K[:, 7 + i], poly_order)

    # Equation of state fit
    E0 = prop_0K[basis_placement, 3] * molecules_in_coord
    [B, dB], _ = scipy.optimize.curve_fit(lambda volumes, B, dB: eos.EV_EOS(volumes, V0, B, dB, E0, eq_of_state),
                                          volumes, molecules_in_coord * prop_0K[:, 3], p0=[2., 2.])
    np.save(Output + '_EOS', [V0, B, dB, E0])
    np.save(Output + '_EOS_V', volumes)
    np.save(Output + '_EOS_U', molecules_in_coord * prop_0K[:, 3])

    plt.plot(volumes, eos.EV_EOS(volumes, V0, B, dB, E0, eq_of_state) / molecules_in_coord - prop_0K[:, 3])
    plt.xlabel('Volume [Ang.$^{3}$]', fontsize=18)
    plt.ylabel('$\delta(U_{EOS})$ [kcal/mol]', fontsize=18)
    plt.tight_layout()
    plt.savefig(Output + '_EOS_dU.pdf')
    plt.close()


    # Minimize Gibbs free energy at a given temperature using all polynomial functions
    minimum_gibbs_properties = np.zeros((len(Temperature), 14))
    for i in range(len(Temperature)):
        V = scipy.optimize.minimize(Poly_Gibbs_energy, V0, args=(Pressure, Temperature[i], wavenumber_poly, V0, B, dB, E0,
                                                                 eq_of_state, Statistical_mechanics),
                                    method='L-BFGS-B', tol=1.e-10).x
        wvn = wavenumbers_from_poly(wavenumber_poly, V)
        minimum_gibbs_properties[i, 0] = Temperature[i]  # Temperature
        minimum_gibbs_properties[i, 1] = Pressure  # Pressure
        minimum_gibbs_properties[i, 3] = eos.EV_EOS(V, V0, B, dB, E0, eq_of_state) / molecules_in_coord
        for j in range(6):
            lp_poly = np.poly1d(lattice_parameter_poly[j])
            minimum_gibbs_properties[i, 7 + j] = lp_poly(V)  # Lattice parameters
        minimum_gibbs_properties[i, 6] = V  # Volume
        minimum_gibbs_properties[i, 4] = Vibrational_Helmholtz(Temperature[i], wvn, Statistical_mechanics) / molecules_in_coord
        minimum_gibbs_properties[i, 13] = Vibrational_Entropy(Temperature[i], wvn, Statistical_mechanics) / molecules_in_coord
        minimum_gibbs_properties[i, 5] = PV_energy(Pressure, V) / molecules_in_coord  # PV
        minimum_gibbs_properties[i, 2] = sum(minimum_gibbs_properties[i, 3:6])  # Gibbs free energy
    return minimum_gibbs_properties

def Poly_Gibbs_energy(V, P, T, wavenumber_poly, V0, B, dB, E0, eq_of_state, Statistical_mechanics):
    wavenumbers = wavenumbers_from_poly(wavenumber_poly, V)
    PV = PV_energy(P, V)
    U = eos.EV_EOS(V, V0, B, dB, E0, eq_of_state)
    Av = Vibrational_Helmholtz(T, wavenumbers, Statistical_mechanics)
    return PV + U + Av

def wavenumbers_from_poly(ply, V):
    wavenumbers_out = np.zeros(len(ply[:, 0]))
    for i in range(len(wavenumbers_out)):
        wvn_ply = np.poly1d(ply[i])
        wavenumbers_out[i] = wvn_ply(V)
    return wavenumbers_out


def atoms_count(Program, Coordinate_file, molecules_in_coord=1):
    """
    This program will determine the number of atoms in the coordinate file if molecules_in_coord is not specified.
    It will return the number of atoms per molecule if molecule_in_coord is specified.
    :param Program: Tinker, CP2K, or test
    :param Coordinate_file: input coordinate file for supported program
    :param molecules_in_coord: the number of molecules within the coordinate file. This is an optional parameter
    :return: 
    """
    if Program == 'Tinker':
        atoms_in_coord = len(Ex.Return_Tinker_Coordinates(Coordinate_file)[:, 0])
    elif Program == 'Test':
        atoms_in_coord = len(Wvn.Test_Wavenumber(Coordinate_file, True)) / 3
    elif Program == 'CP2K':
        atoms_in_coord = len(Ex.Return_CP2K_Coordinates(Coordinate_file)[:, 0])
    elif Program == 'QE':
        atoms_in_coord = len(QE_atoms_per_molecule(Coordinate_file, 1))
    return int(atoms_in_coord / molecules_in_coord)


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

def CP2K_U(coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns the potential energy
    Random funcitons can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters
    """
    l = open(coordinate_file)
    lines = l.readlines()
    U = float(lines[0].split()[-1])*627.5	   
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

def QE_U(Coordinate_file):
    print('getting energy from'+Coordinate_file)
    with open(Coordinate_file,'r') as lines:
        rlines = lines.readlines()
        energy = float(rlines[2].split()[5]) *313.754
    return energy

def QE_Lattice_Parameters(Coordinate_file):
    lfile = open(Coordinate_file+'bv')
    lines = lfile.readlines()
    matrix = np.zeros((3,3))
    for x in range(0,3):
        vect = lines[x+1]
        matrix[x,0] = float(vect.split()[0])
        matrix[x,1] = float(vect.split()[1])
        matrix[x,2] = float(vect.split()[2])
    lattice_parameters = Ex.crystal_matrix_to_lattice_parameters(np.transpose(matrix))
    return lattice_parameters, matrix




def QE_atoms_per_molecule(Coordinate_file, molecules_in_coord):
    lfile = open(Coordinate_file)
    filelines = lfile.readlines()
    numatom = 0
    for x in range(0,len(filelines)):
        if filelines[x].split()[0] in ['C','H','O','S','I','Cl','N']:
            numatom+=1
    atoms_per_molecule = numatom / molecules_in_coord
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
        elif program == 'QE':
            print('gettinc lattice from'+coordinate_file)
            lattice_parameters, matrix = QE_Lattice_Parameters(coordinate_file)
    V = lattice_parameters[0] * lattice_parameters[1] * lattice_parameters[2] * np.sqrt(
        1 - np.cos(np.radians(lattice_parameters[3])) ** 2 - np.cos(np.radians(lattice_parameters[4])) ** 2 - np.cos(
            np.radians(lattice_parameters[5])) ** 2 + 2 * np.cos(np.radians(lattice_parameters[3])) * np.cos(
            np.radians(lattice_parameters[4])) * np.cos(np.radians(lattice_parameters[5])))
    return V


def Potential_energy(Coordinate_file, Program, Parameter_file=''):
    if Program == 'Tinker':
        U = Tinker_U(Coordinate_file, Parameter_file)
    elif Program == 'Test':
        U = Test_U(Coordinate_file)
    elif Program == 'CP2K':
        U = CP2K_U(Coordinate_file)
    elif Program == 'QE':
        U = QE_U(Coordinate_file)
    return U

def Lattice_parameters(Program, Coordinate_file):
    if Program == 'Tinker':
        lattice_parameters = Tinker_Lattice_Parameters(Coordinate_file)
    elif Program == 'Test':
        lattice_parameters = Test_Lattice_Parameters(Coordinate_file)
    elif Program == 'CP2K':
        lattice_parameters = CP2K_Lattice_Parameters(Coordinate_file)
    elif Program == 'QE':
        lattice_parameters, matrix = QE_Lattice_Parameters(Coordinate_file)
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
    h = 2.520 * 10 ** (-38)  # Reduced Plank's constant in kcal*s
    k = 3.2998 * 10 ** (-27)  # Boltzmann constant in kcal/K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    A = []
    for i in wavenumbers[3:]:  # Skipping the translational modes
        if i > 0:  # Skipping negative wavenumbers
            a = (1 / beta) * np.log(beta * h * i * c) * Na
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
    h = 2.520 * 10 ** (-38)  # Reduced Plank's constant in kcal/s
    k = 3.2998 * 10 ** (-27)  # Boltzmann constant in kcal/K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    A = []
    for i in wavenumbers[3:]:  # Skipping translational modes
        if i > 0:  # Skipping negative wavenumbers
            if Temperature == 0:
                a = ((h * i * c / (4 * np.pi)) )
            else:
                a = ((h * i * c / (4 *np.pi)) + (k * Temperature) * np.log(1 - np.exp(-beta * h * i * c / (2 * np.pi))))
            A.append(a)
        else:
            pass
    A = sum(A)*Na
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
    if Statistical_mechanics == 'Classical' and Temperature != 0.:
        S = Classical_Vibrational_S(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        S = Quantum_Vibrational_S(Temperature, wavenumbers)
    else:
        S = 0.
    return S

def Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics):
    if Statistical_mechanics == 'Classical' and Temperature != 0.:
        Av = Classical_Vibrational_A(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        Av = Quantum_Vibrational_A(Temperature, wavenumbers)
    else:
        Av = 0.
    return Av


def Classical_Vibrational_S(Temperature, wavenumbers):
    """
    Funciton to calculate the classical vibrational entropy at a given temperature

    **Required Inputs
    Temperature = single temperature in Kelvin to determine the vibrational entropy (does not work at 0 K)
    wavenumbers = array of wavenumber (in order with the first three being 0 cm**-1 for the translational modes)
    """
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-38)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-27)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    S = []
    for i in wavenumbers[3:]:  # Skipping translational modes
        if i > 0:  # Skipping negative wavenumbers
            s = k * (1 - np.log(beta * h * i * c)) * Na 
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
    h = 2.520 * 10 ** (-38)  # Reduced Plank's constant in cal*s
    k = 3.2998 * 10 ** (-27)  # Boltzmann constant in cal*K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    S = []
    for i in wavenumbers[3:]:
        if i > 0:
            s = ((h * i * c / (2 * np.pi))/(Temperature*(np.exp(h * i * c * beta / (2* np.pi))-1))) - (k*Temperature*np.log(1-np.exp(-h * i * c * beta / (2 *np.pi))))
            S.append(s)
        else:
            pass
    S = sum(S)*Na
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
        U = CP2K_U(Coordinate_file)
    elif Program == 'QE':
        U = QE_U(Coordinate_file)
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


