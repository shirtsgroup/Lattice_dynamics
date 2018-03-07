#!/usr/bin/env python
import os
import subprocess
import sys
import numpy as np
import Expand as Ex
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn

#######################################################################################################################
##############                                    General Funcitons                                       #############
#######################################################################################################################

def write_out(string):
    with open("numerical_checks.out", "a") as myfile:
        myfile.write(string)


def numpy_write_out(array):
    with open('numerical_checks.out', 'ab') as myfile:
        np.savetxt(myfile, array, '%16.10f')
        myfile.write(b'\n')

def numpy_write_out_precision(array):
    with open('numerical_checks.out', 'ab') as myfile:
        np.savetxt(myfile, array, '%16.10f')
        myfile.write(b'\n')



def tolerance_of_wavenumbers(wavenumbers, Wavenum_Tol):
    if any(wavenumbers < Wavenum_Tol):
        with open('numerical_checks.out', 'a') as myfile:
            myfile.write('Wavenumbers [cm^-1]: ' + str(wavenumbers[0]) + ' ' + str(wavenumbers[1]) + ' ' + 
                         str(wavenumbers[2]))

def raw_energies(U_0, Av_0, U, Av):
    write_out('U (Potential Energy) [kcal/mol] = \n')
    write_out('    U_0 = \n')
    numpy_write_out(U_0)
    write_out('     -dC    +dC   \n')
    numpy_write_out(U)

    write_out('Av (Vibational Energy) [kcal/mol] = \n')
    write_out('    Av_0 = \n')
    numpy_write_out(Av_0)
    write_out('     -dC    +dC   \n')
    numpy_write_out(Av)


#######################################################################################################################
##############                                    4th Order Runge-Kutta                                   #############
#######################################################################################################################

def start_RK(T_0, dT):
    write_out('//////////////// Runge-Kutta from ' + str(T_0) + ' to ' + str(T_0 + dT) + 'K ////////////////\n')


def step_RK(index, T, Program, Coordinate_file):
    write_out('///   RUNGE-KUTTA Step ' + str(index + 1) + ' at T = ' + str(T)+ 'K   ///   \n')
    lattice_parameters = Pr.Lattice_parameters(Program, Coordinate_file)
    write_out('Lattice vectors [Ang.^3] = \n')
    numpy_write_out(lattice_parameters[:3])
    write_out('Lattice angles [Degrees] = \n')
    numpy_write_out(lattice_parameters[3:])

def end_RK(k_values):
    write_out('///   RUNGE-KUTTA summary   ///   \n')
    numpy_write_out(k_values)



#######################################################################################################################
##############                                    General Gradient                                        #############
#######################################################################################################################

def gradient_output(T, Program, Coordinate_file):
    write_out('//////////////// Gradient at ' + str(T) + 'K ////////////////\n')
    lattice_parameters = Pr.Lattice_parameters(Program, Coordinate_file)
    write_out('Lattice vectors [Ang.^3] = \n')
    numpy_write_out(lattice_parameters[:3])
    write_out('Lattice angles [Degrees] = \n')
    numpy_write_out(lattice_parameters[3:])


#######################################################################################################################
##############                                     Isotropic Gradient                                     #############
#######################################################################################################################

def iso_gradient(dG, ddG, dS, dV):
    # Setting output for dG/deta: the current structure should be at a minima. Therefore, dG/deta should be zero
    warning = False
    if (dG[0] < dG[1] < dG[2]) and (dG[0] < 0.) and (0. < dG[2]):
        # There is no issue if the following dG/deta are true for the finite difference rankings:
        #     backwards < 0 < forwards
        # Central should be inbetween them
        pass
    else:
        # Raising a flag if a strain is negative
        warning = True

    write_out('dG/dV [kcal/(mol*Ang.^3)] =  \n')
    write_out('  Backwards    Central    Forward \n')
    numpy_write_out(np.matrix(dG))
    if warning == True:
        write_out('WARNING: current structure is not the minimum free energy structure at this temperature!\n')

    # Outputting d^2G/deta^2
    write_out('d^(2)G/dV^(2) [kcal/(mol*Ang.^3)] = \n')
    numpy_write_out(np.matrix(ddG))

    # Outputting dS/deta
    write_out('dS/dV [kcal/(mol*K*Ang.^3)] = \n')
    numpy_write_out(np.matrix(dS))

    # Outputting deta/dT
    write_out('dV/dT [Ang.^3/K] = \n')
    numpy_write_out_precision(np.matrix(dV))
    write_out('\n')


#######################################################################################################################
##############                                    Anisotropic Gradient                                    #############
#######################################################################################################################

def aniso_gradient(dG, ddG, dS, dC):
    # Setting output for dG/deta: the current structure should be at a minima. Therefore, dG/deta should be zero
    warning = False
    for i in range(6):
        if (dG[i, 0] < dG[i,1] < dG[i,2]) and (dG[i,0] < 0.) and (0. < dG[i,2]):
            # There is no issue if the following dG/deta are true for the finite difference rankings:
            #     backwards < 0 < forwards
            # Central should be inbetween them
            pass
        elif dG[i,0] == dG[i,1] == dG[i,2] == 0.:
            pass
        else:
            # Raising a flag if a strain is negative
            warning = True

    write_out('dG/dC [kcal/(mol*Ang.)] = \n')
    write_out('   Backwards     Central     Forward \n')
    numpy_write_out(dG)
    if warning == True:
        write_out('WARNING: current structure is not the minimum free energy structure at this temperature!\n')

    # Outputting d^2G/deta^2
    write_out('d^(2)G/dC^(2) [kcal/(mol*Ang.^2)] = \n')
    numpy_write_out(ddG)

    # Outputting dS/deta
    write_out('dS/dC [kcal/(mol*K*Ang.)] = \n')
    numpy_write_out(np.matrix(dS))

    # Outputting deta/dT
    write_out('dC/dT [Ang./K] = \n')
    numpy_write_out_precision(dC)
    write_out('\n')


def iso_pressure_gradient(dG, ddG, dV):
    # Setting output for dG/deta: the current structure should be at a minima. Therefore, dG/deta should be zero
    warning = False
    if (dG[0] < dG[1] < dG[2]) and (dG[0] < 0.) and (0. < dG[2]):
        # There is no issue if the following dG/deta are true for the finite difference rankings:
        #     backwards < 0 < forwards
        # Central should be inbetween them
        pass
    else:
        # Raising a flag if a strain is negative
        warning = True

    write_out('dG/dV [kcal/(mol*Ang.^3)] =  \n')
    write_out('  Backwards    Central    Forward \n')
    numpy_write_out(np.matrix(dG))
    if warning == True:
        write_out('WARNING: current structure is not the minimum free energy structure at this temperature!\n')

    # Outputting d^2G/deta^2
    write_out('d^(2)G/dV^(2) [kcal/(mol*Ang.^3)] = \n')
    numpy_write_out(np.matrix(ddG))

    # Outputting deta/dT
    write_out('dV/dP [Ang.^3/atm] = \n')
    numpy_write_out_precision(np.matrix(dV))
    write_out('\n')



