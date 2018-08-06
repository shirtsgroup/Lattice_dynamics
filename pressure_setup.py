#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import subprocess
import scipy.optimize
import numpy as np
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Thermal_NumericalAnalysis as TNA
import Expand as Ex
import System_sensitivity as Ss
import matplotlib.pyplot as plt
import equations_of_state as eos

def pressure_setup(Temperature=[0.0, 25.0, 50.0, 75.0, 100.0], Pressure=1., Method='HA', Program='Test',
                   Output='out', Coordinate_file='molecule.xyz', Parameter_file='keyfile.key',
                   molecules_in_coord=1, properties_to_save=['G', 'T'], NumAnalysis_method='RK4',
                   NumAnalysis_step=25.0,
                   LocGrd_Vol_FracStep='',
                   LocGrd_CMatrix_FracStep='',
                   StepWise_Vol_StepFrac=1.5e-3,
                   StepWise_Vol_LowerFrac=0.97, StepWise_Vol_UpperFrac=1.16,
                   Statistical_mechanics='Classical', Gruneisen_Vol_FracStep=1.5e-3,
                   Gruneisen_Lat_FracStep=1.0e-3, Wavenum_Tol=-1., Gradient_MaxTemp=300.0,
                   Aniso_LocGrad_Type='6D', min_RMS_gradient=0.01, eq_of_state='Murnaghan', 
                   gru_from_0T_0P=True, cp2kroot='BNZ_NMA_p2'):

    if Method == 'HA':
        print("Pressure vs. Temperature methods have not been implimented for the Harmonic Approximation")
        sys.exit()

    elif (Method == 'GaQ') or (Method == 'GaQg'):
        print("Pressure vs. Temperature methods have not been implimented for anisotropic expansion")
        sys.exit()

    file_ending = Ex.assign_file_ending(Program)

    # Making an array of volume fractions
    V_frac = np.arange(StepWise_Vol_LowerFrac, 1.0, StepWise_Vol_StepFrac)
    V_frac = np.arange(StepWise_Vol_LowerFrac, StepWise_Vol_UpperFrac, StepWise_Vol_StepFrac)

    # Volume of lattice minimum strucutre
    V0 = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)

    # Making an array to store the potential energy and volume of each structure
    U = np.zeros(len(V_frac))
    V = np.zeros(len(V_frac))

    for i in range(len(V_frac)):
        # Expanding the structures and saving the required data
        Ex.Call_Expansion(Method, 'expand', Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
                          volume_fraction_change=V_frac[i], Output='temporary', Parameter_file=Parameter_file)
        U[i] = Pr.Potential_energy('temporary' + file_ending, Program, Parameter_file=Parameter_file)
        V[i] = Pr.Volume(Program=Program, Coordinate_file='temporary' + file_ending)
        subprocess.call(['rm', 'temporary' + file_ending])

    V0 = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)
    E0 = Pr.Potential_energy(Coordinate_file, Program, Parameter_file=Parameter_file)

    if eq_of_state != 'None':
        [B, dB],_ = scipy.optimize.curve_fit(lambda V, B, dB: eos.EV_EOS(V, V0, B, dB, E0, eq_of_state), V, U, p0=[2.,2.])
        np.save(Output + '_EOS', [V0, B, dB, E0])

        plt.plot(V, (eos.EV_EOS(V, V0, B, dB, E0, eq_of_state) - U) / molecules_in_coord)
        plt.xlabel('Volume [Ang.$^{3}$]', fontsize=18)
        plt.ylabel('$\delta(U_{EOS})$ [kcal/mol]', fontsize=18)
        plt.tight_layout()
        plt.savefig(Output + '_EOS_dU.pdf')
        plt.close()

        plt.plot(V, eos.EV_EOS(V, V0, B, dB, E0, eq_of_state) / molecules_in_coord)
        plt.scatter(V, U / molecules_in_coord)
        plt.xlabel('Volume [Ang.$^{3}$]', fontsize=18)
        plt.ylabel('$U$ [kcal/mol]', fontsize=18)
        plt.tight_layout()
        plt.savefig(Output + '_EOS_U.pdf')
        plt.close()

        if gru_from_0T_0P == True:
            EOS_TvP_Gru_0T_0P(Method, Temperature, Pressure, Program, Output, Coordinate_file, Parameter_file, 
                              molecules_in_coord, properties_to_save, Statistical_mechanics, Gruneisen_Vol_FracStep, 
                              Wavenum_Tol, min_RMS_gradient, eq_of_state, cp2kroot, V0, B, dB, E0)
            sys.exit()

        else:
            lattice_parameter_name = Coordinate_file.split('.')[0]
            for i in range(len(Pressure)):
                v_p = scipy.optimize.minimize(pressure_minimization_at_0T, V0, args=(Pressure[i], V0, B, dB, eq_of_state), method='Nelder-Mead', tol=1.e-15).x
     
                print(Pressure[i], v_p)
                V = np.arange(100, 5000, 10)
                plt.plot(V, eos.EV_EOS(V, V0, B, dB, E0, eq_of_state) + Pressure[i]*V)
                plt.scatter(v_p, eos.EV_EOS(v_p, V0, B, dB, E0, eq_of_state) + Pressure[i]*v_p)
                plt.show()
    else:
        print("WARNING: Option of None for EOS is not set up yet, exiting.")
        sys.exit()

def pressure_minimization_at_0T(V, P, V0, B, dB, eq_of_state):
    return np.absolute(eos.PV_EOS(V, V0, B, dB, eq_of_state) - P)



#    U_out = np.zeros(len(Pressure))
#    V_out = np.zeros(len(Pressure))
#
#    for i in range(len(Pressure)):
#        # Fitting the U + PV energy to a 4th order polynomial
#        U_fit = np.polyfit(V, U + Pr.PV_energy(Pressure[i], V), 4)
#        U_fit = np.poly1d(U_fit)
#
#        # Using a fine volume spacing to determine the minimum energy volume at each pressure
#        V_fine = np.arange(min(V), max(V), 1.)
#        U_fine = U_fit(V_fine)
#        V_min = V_fine[np.where(U_fine == min(U_fine))]
#
#        # Expanding the lattice minimum structure to the minimum energy structure at Pressure i
#        Ex.Call_Expansion(Method, 'expand', Program, Coordinate_file, molecules_in_coord, min_RMS_gradient,
#                          volume_fraction_change=V_min/V0, Output='temporary', Parameter_file=Parameter_file)
#
#        # Running a tighter minimization
#        print('Pressure: ', Pressure[i])
#        scipy.optimize.minimize(U_PV, V_min, args=(Pressure[i], file_ending),method='Nelder-Mead', tol=1.e-15)
#
#        # Making a new directory
#        subprocess.call(['mkdir', Output + '_' + str(Pressure[i]) + 'atm'])
#
#        # Copying necessary files over
#        subprocess.call(['cp', 'temporary' + file_ending, Output + '_' + str(Pressure[i]) + 'atm/' + Coordinate_file])
#        try:
#            subprocess.call(['cp', Parameter_file, Output + '_' + str(Pressure[i]) + 'atm'])
#        except ValueError:
#            pass
#        write_input_file(Temperature, Pressure[i], Method, Program, Output, Coordinate_file, Parameter_file,
#                         molecules_in_coord, properties_to_save, NumAnalysis_method, NumAnalysis_step, LocGrd_Vol_FracStep,
#                         LocGrd_CMatrix_FracStep, StepWise_Vol_StepFrac, StepWise_Vol_LowerFrac,
#                         StepWise_Vol_UpperFrac, Statistical_mechanics, Gruneisen_Vol_FracStep, Gruneisen_Lat_FracStep,
#                         Wavenum_Tol, Gradient_MaxTemp, Aniso_LocGrad_Type, min_RMS_gradient,
#                         Output + '_' + str(Pressure[i]) + 'atm/input.inp')
#        U_out[i] = (Pr.Potential_energy('temporary' + file_ending, Program, Parameter_file=Parameter_file) + Pr.PV_energy(Pressure[i], Pr.Volume(Program=Program, Coordinate_file='temporary' + file_ending))) / molecules_in_coord
#        V_out[i] = Pr.Volume(Program=Program, Coordinate_file='temporary' + file_ending)
#        subprocess.call(['rm', 'temporary' + file_ending])
#
#    np.save('U', U_out)
#    np.save('V', V_out)
#    np.save('P', Pressure)
#
#def U_PV(V, Pressure, file_ending):
#    # WARNING: This function should only be used with Pressure_setup !!!
#    V_hold = Pr.Volume(Program=Program, Coordinate_file='temporary' + file_ending)
#    V_frac = V / V_hold
#    Ex.Call_Expansion(Method, 'expand', Program, 'temporary' + file_ending, molecules_in_coord, min_RMS_gradient,
#                      volume_fraction_change=V_frac, Output='temporary', Parameter_file=Parameter_file)
#    U = (Pr.Potential_energy('temporary' + file_ending, Program, Parameter_file=Parameter_file) \
#           + Pr.PV_energy(Pressure, Pr.Volume(Program=Program, Coordinate_file='temporary' + file_ending))) / molecules_in_coord
#    return U


#### Running EOS P vs. T using the Gruneisen parameters computed at 0K and 0atm ####
def EOS_TvP_Gru_0T_0P(Method, Temperature, Pressure, Program, Output, Coordinate_file, Parameter_file, molecules_in_coord, 
                      properties_to_save, Statistical_mechanics, Gruneisen_Vol_FracStep, Wavenum_Tol, min_RMS_gradient, 
                      eq_of_state, cp2kroot, V0, B, dB, E0):

    # Calculating the Gruneisen parameter with the lattice minimum structure
    print("   Calculating the isotropic Gruneisen parameter")
    Gruneisen, Wavenumber_Reference, Volume_Reference = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Output=Output, 
                                                                             Coordinate_file=Coordinate_file, 
                                                                             Program=Program, Gruneisen_Vol_FracStep=
                                                                             Gruneisen_Vol_FracStep, molecules_in_coord=
                                                                             molecules_in_coord, Parameter_file=
                                                                             Parameter_file, cp2kroot=cp2kroot)

    # Setting up an array to store the properties
    properties = np.zeros((len(Pressure), len(Temperature), 14))
    if Program == 'Tinker':
        properties[:, :, 7:13] = Pr.Tinker_Lattice_Parameters(Coordinate_file)

    for i in range(len(Pressure)):
        for j in range(len(Temperature)):
            V = scipy.optimize.minimize(Gibbs_EOS_temperature_pressure, V0, args=(Method, min_RMS_gradient, Gruneisen, 
                                        Wavenumber_Reference, V0, B, dB, E0, eq_of_state, Temperature[j], 
                                        Statistical_mechanics, Pressure[i], molecules_in_coord), method='Nelder-Mead', 
                                        tol=1.e-15).x

            wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen, Wavenumber_Reference=
                                               Wavenumber_Reference, Volume_Reference=V0, New_Volume=V)

            properties[i, j, 0] = Temperature[j]
            properties[i, j, 1] = Pressure[i]
            properties[i, j, 3] = eos.EV_EOS(V, V0, B, dB, E0, eq_of_state) / molecules_in_coord
            properties[i, j, 4] = Pr.Vibrational_Helmholtz(Temperature[j], wavenumbers, Statistical_mechanics) / molecules_in_coord
            properties[i, j, 5] = Pr.PV_energy(Pressure[i], V) / molecules_in_coord
            properties[i, j, 2] = sum(properties[i, j, 3:6])
            properties[i, j, 6] = V
            properties[i, j, 7:10] = (V / V0) ** (1 / 3) * properties[i, j, 7:10]
            properties[i, j, 13] = Pr.Vibrational_Entropy(Temperature[j], wavenumbers, Statistical_mechanics) / molecules_in_coord

    Save_Properties_PvsT(properties, properties_to_save, Output, Method, Statistical_mechanics)
    sys.exit()
            


def Gibbs_EOS_temperature_pressure(V, Method, min_RMS_gradient, Gruneisen, Wavenumber_Reference, V0, B, dB, E0,
                                   eq_of_state, Temperature, Statistical_mechanics, Pressure, molecules_in_coord):
    # Calculating the wavenumbers of the expanded structure
    wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Gruneisen=Gruneisen, Wavenumber_Reference=
                                       Wavenumber_Reference, Volume_Reference=V0, New_Volume=V)

    # Calculating the potential energy
    U = eos.EV_EOS(V, V0, B, dB, E0, eq_of_state)

    # Calculating the vibrational energy
    Av = Pr.Vibrational_Helmholtz(Temperature, wavenumbers, Statistical_mechanics)

    # Calculting PV energy
    PV = Pr.PV_energy(Pressure, V)

    return (U + Av + PV) / molecules_in_coord



def Save_Properties_PvsT(properties, properties_to_save, Output, Method, Statistical_mechanics):
    for i in properties_to_save:
        if i == 'T':  # Temperature
            print("   ... Saving temperature in: " + Output + "_T_" + Method + ".npy")
            np.save(Output + '_T_' + Method, properties[:, :, 0])
        if i == 'P':  # Pressure
            print("   ... Saving Pressure in: " + Output + "_P_" + Method + ".npy")
            np.save(Output + '_P_' + Method, properties[:, :, 1])
        if i == 'G':  # Gibbs free energy
            print("   ... Saving Gibbs free energy in: " + Output + "_G" + Statistical_mechanics + "_" + Method +\
                  ".npy")
            np.save(Output + '_G' + Statistical_mechanics + '_' + Method, properties[:, :, 2])
        if i == 'U':  # Potential energy
            print("   ... Saving potential energy in: " + Output + "_U" + Statistical_mechanics + "_" + Method + ".npy")
            np.save(Output + '_U' + Statistical_mechanics + '_' + Method, properties[:, :, 3])
        if i == 'Av':  # Helmholtz vibrational energy
            print("   ... Saving vibrational Helmholtz free energy in: " + Output + "_Av" + Statistical_mechanics + "_"\
                  + Method + ".npy")
            np.save(Output + '_Av' + Statistical_mechanics + '_' + Method, properties[:, :, 4])
        if i == 'V':  # Volume
            print("   ... Saving volume in: " + Output + "_V" + Statistical_mechanics + "_" + Method + ".npy")
            np.save(Output + '_V' + Statistical_mechanics + '_' + Method, properties[:, :, 6])
        if i == 'h':  # Lattice parameters
            print("   ... Saving lattice parameters in: " + Output + "_h" + Statistical_mechanics + "_" + Method +\
                  ".npy")
            np.save(Output + '_h' + Statistical_mechanics + '_' + Method, properties[:, :, 7:13])

