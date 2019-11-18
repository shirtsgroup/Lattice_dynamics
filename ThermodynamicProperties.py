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
import program_specific_functions as psf
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##########################################
#           Export PROPERTIES            #
##########################################
def Properties(inputs: list, Coordinate_file: int, wavenumbers: list, Temperature: [float, int]) -> list:
    """
    Computes all harmonic properties of a crystal structure at the specified temperature.

    Paramaeters
    -----------
    inputs : list with all inputs from the yaml file
    Coordinate_file : coordinate file corresponding with program selected
    wavenumbers : wavenumbers [cm^-1]
    Temperature : system temperature [K]

    Returns
    -------
    :return: length 14 array of thermodynamic properties in the following order
      0 - Temperature[K]
      1 - Pressure [atm]
      2 - Gibbs free energy [kcal/mol]
      3 - Potential energy [kcal/mol]
      4 - Harmonic Helmholtz free energy [kcal/mol]
      5 - Pressure-volume energy contribution [kcal/mol]
      6 - Volume [Ang.^3]
      7,8,9 - a, b, and c lattice vectors [Ang.]
      10,11,12 - alpha, beta, and gamma lattice angles [Deg.]
      13 - Harmonic entropy [kcal/(mol*K)]
    """

    properties = np.zeros(14)  # list to save properties to be returned

    properties[0] = Temperature  # Temperature
    properties[1] = inputs.pressure  # Pressure
    properties[3] = psf.Potential_energy(Coordinate_file, inputs.program, inputs.tinker_parameter_file)  # Potential Energy
    properties[7:13] = psf.Lattice_parameters(inputs.program, Coordinate_file)  # Lattice parameters
    properties[6] = Volume(lattice_parameters=properties[7:13])  # Volume
    properties[4] = Vibrational_Helmholtz(Temperature, wavenumbers, inputs.statistical_mechanics) / \
                    inputs.number_of_molecules  # Computing the Helmholtz vibrational free energy
    properties[13] = Vibrational_Entropy(Temperature, wavenumbers, inputs.statistical_mechanics) / \
                     inputs.number_of_molecules  # Computing the vibrational entropy
    properties[5] = PV_energy(inputs.pressure, properties[6]) / inputs.number_of_molecules  # Pressure-Volume energy contribution
    properties[2] = sum(properties[3:6])  # Gibbs free energy
    return properties


def Properties_with_Temperature(inputs: list, Coordinate_file: int, wavenumbers: list) -> list:
    """
    Computes all harmonic properties of a crystal structure at all temperatures.

    Paramaeters
    -----------
    inputs : list with all inputs from the yaml file
    Coordinate_file : coordinate file corresponding with program selected
    wavenumbers : wavenumbers [cm^-1]
    Temperature : system temperatures [K]

    Returns
    -------
    :return: length 14 array of thermodynamic properties in the following order
      0 - Temperature[K]
      1 - Pressure [atm]
      2 - Gibbs free energy [kcal/mol]
      3 - Potential energy [kcal/mol]
      4 - Harmonic Helmholtz free energy [kcal/mol]
      5 - Pressure-volume energy contribution [kcal/mol]
      6 - Volume [Ang.^3]
      7,8,9 - a, b, and c lattice vectors [Ang.]
      10,11,12 - alpha, beta, and gamma lattice angles [Deg.]
      13 - Harmonic entropy [kcal/(mol*K)]
    """
    properties = np.zeros((len(inputs.temperature), 14))  # list to save properties to be returned
    properties[0, :] = Properties(inputs, Coordinate_file, wavenumbers, inputs.temperature[0]) # Computing all properties for the first temperature
    properties[:, 0] = inputs.temperature  # Temperature
    properties[:, 1] = properties[0, 1]  # Pressure
    properties[:, 3] = properties[0, 3]  # Potential energy
    properties[:, 7:13] = properties[0, 7:13]  # Lattice parameters
    properties[:, 6] = properties[0, 6]  # Volume
    properties[:, 5] = properties[0, 5]  # Pressure-volume energy
    for i in range(1, len(inputs.temperature)):
        properties[i, 4] = Vibrational_Helmholtz(inputs.temperature[i], wavenumbers, inputs.statistical_mechanics) / \
                           inputs.number_of_molecules  # Harmonic Helmholtz free energy
        properties[i, 13] = Vibrational_Entropy(inputs.temperature[i], wavenumbers, inputs.statistical_mechanics) / \
                            inputs.number_of_molecules  # Harmonic entropy
        properties[i, 2] = sum(properties[i, 3:6])  # Gibbs free energy
    return properties


def Save_Properties(inputs: list, properties: list):
    """
    Goes through the user specified properties to save to individual numpy files.

    Paramaeters
    -----------
    inputs : list with all inputs from the yaml file
    properties : computed properties at all temperature of interest
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
        lattice_parameters = psf.Lattice_parameters(program, coordinate_file)

    V = lattice_parameters[0] * lattice_parameters[1] * lattice_parameters[2] * np.sqrt(
        1 - np.cos(np.radians(lattice_parameters[3])) ** 2 - np.cos(np.radians(lattice_parameters[4])) ** 2 - np.cos(
            np.radians(lattice_parameters[5])) ** 2 + 2 * np.cos(np.radians(lattice_parameters[3])) * np.cos(
            np.radians(lattice_parameters[4])) * np.cos(np.radians(lattice_parameters[5])))
    return V

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


# Hemholtz free energy of a harmonic oscillator
def Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics):
    if Statistical_mechanics == 'Classical' and Temperature != 0.:
        Av = Classical_Vibrational_A(Temperature, wavenumbers)
    elif Statistical_mechanics == 'Quantum':
        Av = Quantum_Vibrational_A(Temperature, wavenumbers)
    else:
        Av = 0.
    return Av

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
    wavenumbers = np.sort(wavenumbers)
    A = []
    if Temperature == 0.:
        A = 0.
    else:
        beta = 1 / (k * Temperature)
        for i in wavenumbers[3:]:  # Skipping the translational modes
            if i > 0:  # Skipping negative wavenumbers
                a = (1 / beta) * np.log(beta * h * i * c) * Na
                A.append(a)
            else:
                pass
        A = sum(A)
    return A

def Quantum_Vibrational_A(Temperature, wavenumbers):
    c = 2.998 * 10 ** 10  # Speed of light in cm/s
    h = 2.520 * 10 ** (-38)  # Reduced Plank's constant in kcal/s
    k = 3.2998 * 10 ** (-27)  # Boltzmann constant in kcal/K
    Na = 6.022 * 10 ** 23  # Avogadro's number
    if Temperature != 0.:
        beta = 1 / (k * Temperature)
    wavenumbers = np.sort(wavenumbers)
    A = []
    for i in wavenumbers[3:]:  # Skipping translational modes
        if i > 0:  # Skipping negative wavenumbers
            if Temperature == 0.:
                a = (h * i * c / 2) * Na
            else:
                a = (h * i * c / 2 + (1 / beta) * np.log(1 - np.exp(- beta * h * i * c ))) * Na
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
    if Temperature == 0.:
        if Statistical_mechanics == 'Classical':
            S = Classical_Vibrational_S(Temperature, wavenumbers)
        elif Statistical_mechanics == 'Quantum':
            S = Quantum_Vibrational_S(Temperature, wavenumbers)
    else:
        S = 0.
    return S

def Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics):
    if Statistical_mechanics == 'Classical':
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
            s = (k * h * i * c / (Temperature * (np.exp(h * i * c * beta) - 1)) - k * np.log(1 - np.exp(-h * i * c * beta))) * Na
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
        U = CP2K_U(Coordinate_file)
    elif Program == 'QE':
        U = QE_U(Coordinate_file)
    # Volume
    volume = Volume(Program=Program, Coordinate_file=Coordinate_file)

    # Helmholtz free energy
    if Statistical_mechanics == 'Classical':
        if Temperature < 0.1:
            A = Classical_Vibrational_A(Temperature, wavenumbers) / molecules_in_coord
        else:
            A = 0.
    elif Statistical_mechanics == 'Quantum':
        A = Quantum_Vibrational_A(Temperature, wavenumbers) / molecules_in_coord 

    # Gibbs Free energy
    G = U + A + PV_energy(Pressure, volume) / molecules_in_coord
    return G, U, A

