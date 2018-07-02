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
import pressure_setup as ps

def Temperature_Lattice_Dynamics(Temperature=[0.,300.], Pressure=1., Method='HA', Program='Test',
                                 Output='out', Coordinate_file='test.npy', Parameter_file='keyfile.key',
                                 molecules_in_coord=1, properties_to_save=['G', 'T', 'V'], NumAnalysis_method='RK4',
                                 NumAnalysis_step=300.0, LocGrd_Vol_FracStep=0., LocGrd_CMatrix_FracStep=0.,
                                 StepWise_Vol_StepFrac=1e-3, StepWise_Vol_LowerFrac=0.97, StepWise_Vol_UpperFrac=1.16,
                                 Statistical_mechanics='Classical', Gruneisen_Vol_FracStep=1.5e-3, 
                                 Gruneisen_Lat_FracStep=1.e-3, Wavenum_Tol=-1., Gradient_MaxTemp=300.0, 
                                 Aniso_LocGrad_Type='6D', min_RMS_gradient=0.0001, cp2kroot='BNZ_NMA_p3'):
    Temperature = np.array(Temperature).astype(float)
    if Method == 'HA':
        print("Performing Harmonic Approximation")
        # Running the Harmonic Approximation
        if os.path.isfile(Output + '_' + Method + '_WVN.npy'):
            wavenumbers = np.load(Output + '_' + Method + '_WVN.npy')
            print("   Importing wavenumbers from:" + Output + '_' + Method + '_WVN.npy')
        else:
            print("   Computing wavenumbers of coordinate file")
            wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program, Coordinate_file=Coordinate_file,
                                               Parameter_file=Parameter_file, cp2kroot=cp2kroot)
            np.save(Output + '_' + Method + '_WVN', wavenumbers)

        if all(wavenumbers > Wavenum_Tol):
            print("   All wavenumbers are greater than tolerance of: " + str(Wavenum_Tol) + " cm^-1")
            properties = Pr.Properties_with_Temperature(Coordinate_file, wavenumbers, Temperature, Pressure, Program,
                                                        Statistical_mechanics, molecules_in_coord, cp2kroot=cp2kroot,
                                                        Parameter_file=Parameter_file)
            print("   All properties have been saved in " + Output + "_raw.npy")
            np.save(Output + '_raw', properties)
            print("   Saving user specified properties in indipendent files:")
            Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
            print("Harmonic Approximation is complete!")
    else:
        if os.path.isdir('Cords') != True:
            print("Creating directory 'Cords/' to store structures along Gibbs free energy path")
            subprocess.call(['mkdir', 'Cords'])

    if (Method == 'SiQ') or (Method == 'SiQg'):
        # Stepwise Isotropic QHA
        print("Performing Stepwise Isotropic Quasi-Harmonic Approximation")
        properties = TNA.Isotropic_Stepwise_Expansion(StepWise_Vol_StepFrac, StepWise_Vol_LowerFrac,
                                                      StepWise_Vol_UpperFrac, Coordinate_file, Program, Temperature,
                                                      Pressure, Output, Method, molecules_in_coord, Wavenum_Tol,
                                                      Statistical_mechanics, min_RMS_gradient,
                                                      Parameter_file=Parameter_file,
                                                      Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep,
                                                      cp2kroot=cp2kroot)
        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print("Stepwise Isotropic Quasi-Harmonic Approximation is complete!")

    elif (Method == 'GiQ') or (Method == 'GiQg'):
        if LocGrd_Vol_FracStep == 0.:
            LocGrd_dV = Ss.isotropic_gradient_settings(Coordinate_file, Program, Parameter_file, molecules_in_coord,
                                                       min_RMS_gradient, Output, Pressure)
        else:
            V_0 = Pr.Volume(Program=Program, Coordinate_file=Coordinate_file)
            LocGrd_dV = LocGrd_Vol_FracStep * V_0
        # Gradient Isotropic QHA
        print("Performing Gradient Isotropic Quasi-Harmonic Approximation")
        properties = TNA.Isotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method,
                                                      Gradient_MaxTemp, Pressure, LocGrd_dV,
                                                      Statistical_mechanics, NumAnalysis_step, NumAnalysis_method,
                                                      Temperature, min_RMS_gradient,
                                                      Parameter_file=Parameter_file,
                                                      Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep,
                                                      cp2kroot=cp2kroot)
        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print("Gradient Isotropic Quasi-Harmonic Approximation is complete!")

    elif ((Method == 'GaQ') or (Method == 'GaQg')) and (Aniso_LocGrad_Type != '1D'):
        if any(LocGrd_CMatrix_FracStep != 0.):
            crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file)))
            LocGrd_dC = np.absolute(LocGrd_CMatrix_FracStep * crystal_matrix_array)
        else:
            LocGrd_dC = Ss.anisotropic_gradient_settings(Coordinate_file, Program, Parameter_file, molecules_in_coord,
                                                         min_RMS_gradient, Output)
        print("Performing Gradient Anisotropic Quasi-Harmonic Approximation")
        properties = TNA.Anisotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method,
                                                        Gradient_MaxTemp, Pressure, LocGrd_dC,
                                                        Statistical_mechanics, NumAnalysis_step,
                                                        NumAnalysis_method, Aniso_LocGrad_Type, Temperature,
                                                        min_RMS_gradient, Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep, 
                                                        Parameter_file=Parameter_file, cp2kroot=cp2kroot)
        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print("Gradient Anisotropic Quasi-Harmonic Approximation is complete!")

    elif (Method == 'GaQ') or (Method == 'GaQg') and (Aniso_LocGrad_Type == '1D'):
        if any(LocGrd_CMatrix_FracStep != 0.):
            crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file)))
            LocGrd_dC = np.absolute(LocGrd_CMatrix_FracStep * crystal_matrix_array)
        else:
            LocGrd_dC = Ss.anisotropic_gradient_settings(Coordinate_file, Program, Parameter_file, molecules_in_coord,
                                                         min_RMS_gradient, Output)
        print("Performing 1D-Gradient Anisotropic Quasi-Harmonic Approximation")
        properties = TNA.Anisotropic_Gradient_Expansion_1D(Coordinate_file, Program, molecules_in_coord, Output, Method,
                                                           Gradient_MaxTemp, Pressure, LocGrd_dC,
                                                           Statistical_mechanics, NumAnalysis_step,
                                                           NumAnalysis_method, Aniso_LocGrad_Type, Temperature,
                                                           min_RMS_gradient, Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep,
                                                           Parameter_file=Parameter_file, cp2kroot=cp2kroot)
        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print("Gradient Anisotropic Quasi-Harmonic Approximation is complete!")


def write_input_file(Temperature, Pressure, Method, Program, Output, Coordinate_file, Parameter_file, 
                     molecules_in_coord, properties_to_save, NumAnalysis_method, NumAnalysis_step, LocGrd_Vol_FracStep,
                     LocGrd_CMatrix_FracStep, StepWise_Vol_StepFrac, StepWise_Vol_LowerFrac, 
                     StepWise_Vol_UpperFrac, Statistical_mechanics, Gruneisen_Vol_FracStep, Gruneisen_Lat_FracStep,
                     Wavenum_Tol, Gradient_MaxTemp, Aniso_LocGrad_Type, min_RMS_gradient, input_file_location_and_name):
    properties_out = ''
    for i in range(len(properties_to_save) - 1):
        properties_out = properties_out + properties_to_save[i] + ','
    properties_out = properties_out + properties_to_save[-1]

    with open(input_file_location_and_name, 'a') as myfile:
        myfile.write('Temperature = ' + ','.join('{:,}'.format(x) for x in Temperature) + '\n')
        myfile.write('Pressure = ' + str(Pressure) + '\n')
        myfile.write('Method = ' + Method + '\n')
        myfile.write('Program = ' + Program + '\n')
        myfile.write('Output = ' + Output + '\n')
        myfile.write('Coordinate_file = ' + Coordinate_file + '\n')
        myfile.write('Parameter_file = ' + Parameter_file + '\n')
        myfile.write('molecules_in_coord = ' + str(molecules_in_coord) + '\n')
        myfile.write('properties_to_save = ' + properties_out + '\n')
        myfile.write('NumAnalysis_method = ' + NumAnalysis_method + '\n')
        myfile.write('NumAnalysis_step = ' + str(NumAnalysis_step) + '\n')
        myfile.write('LocGrd_Vol_FracStep = ' + str(LocGrd_Vol_FracStep) + '\n')
        myfile.write('LocGrd_CMatrix_FracStep = ' + np.array2string(LocGrd_CMatrix_FracStep,separator=',').strip('[').strip(']').replace(' ','') + '\n')
        myfile.write('StepWise_Vol_StepFrac = ' + str(StepWise_Vol_StepFrac) + '\n')
        myfile.write('StepWise_Vol_LowerFrac = ' + str(StepWise_Vol_LowerFrac) + '\n')
        myfile.write('StepWise_Vol_UpperFrac = ' + str(StepWise_Vol_UpperFrac) + '\n')
        myfile.write('Statistical_mechanics = ' + Statistical_mechanics + '\n')
        myfile.write('Gruneisen_Vol_FracStep = ' + str(Gruneisen_Vol_FracStep) + '\n')
        myfile.write('Gruneisen_Lat_FracStep = ' + str(Gruneisen_Lat_FracStep) + '\n')
        myfile.write('Wavenum_Tol = ' + str(Wavenum_Tol) + '\n')
        myfile.write('Gradient_MaxTemp = ' + str(Gradient_MaxTemp) + '\n')
        myfile.write('Aniso_LocGrad_Type = ' + str(Aniso_LocGrad_Type) + '\n')
        myfile.write('min_RMS_gradient = ' + str(min_RMS_gradient) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate free energies as a function of T using lattice dynamics')
    parser.add_argument('-i', '--input_file', dest='Input_file', default='input_test.py',
                        help='Input file containing all parameters for the run')
    parser.add_argument('-D', '--Start_Fresh', action='store_true',
                        help='Removes any files from previous runs')

    args = parser.parse_args()

    try:
        Method = subprocess.check_output("less " + str(args.Input_file) + " | grep Method | grep = ", shell=True).decode("utf-8")
        Method = str(Method).split('=')[1].strip()
        if Method not in ['HA', 'SiQ', 'SiQg', 'GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print("Input method is not supported. Please select from the following:")
            print("   HA, SiQ, SiQg, GiQ, GiQg, GaQ, GaQg")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("No method was selected")
        print("Exiting code")
        sys.exit()

    try:
        Program = subprocess.check_output("less " + str(args.Input_file) + " | grep Program | grep = ", shell=True).decode("utf-8")
        Program = Program.split('=')[1].strip()
        if Program not in ['Tinker', 'Test', 'CP2K']:
            print("Input program is not supported. Please select from the following:")
            print("   Tinker, Test")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("No program was selected")
        print("Exiting code")
        sys.exit()

    try:
        Statistical_mechanics = subprocess.check_output("less " + str(args.Input_file) + " | grep Statistical_mechanics"
                                                                                         " | grep = ", shell=True).decode("utf-8")
        Statistical_mechanics = Statistical_mechanics.split('=')[1].strip()
        if Statistical_mechanics not in ['Classical', 'Quantum']:
            print("Input statistical mechanics is not supported. Please select from the following:")
            print("   Classical, Quantum")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("Statistical mechnics was not specified")
        print("Exiting code")
        sys.exit()

    try:
        Temperature = subprocess.check_output("less " + str(args.Input_file) + " | grep Temperature"
                                                                               " | grep = ", shell=True).decode("utf-8")
        Temperature = np.array(Temperature.split('=')[1].strip().split(',')).astype(float)
    except subprocess.CalledProcessError as grepexc:
        if Method in ['HA', 'SiQ', 'SiQg']:
            Temperature = [0.0, 25.0, 50.0, 75.0, 100.0]
            print("No temperatures were selected, using default temperatures of:")
            print("   " + str(Temperature))
        else:
            Temperature = []

    try:
        Pressure = subprocess.check_output("less " + str(args.Input_file) + " | grep Pressure | grep = ", shell=True).decode("utf-8")
        Pressure = np.array(Pressure.split('=')[1].strip().split(',')).astype(float) 
        if len(Pressure) == 1:
            Pressure = Pressure[0]
            pressure_scan = False
        elif len(Pressure) > 1:
            pressure_scan = True
    except subprocess.CalledProcessError as grepexc:
        print("No pressure was selected, using default pressure")
        Pressure = 1.

    try:
        Output = subprocess.check_output("less " + str(args.Input_file) + " | grep Output | grep = ", shell=True).decode("utf-8")
        Output = Output.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        Output = 'out'


    # Removing all old files if flagged
    if args.Start_Fresh:
        subprocess.call(['rm -rf Cords numerical_checks.out minimization.out out_*'], shell=True)


    try:
        Coordinate_file = subprocess.check_output("less " + str(args.Input_file) + " | grep Coordinate_file"
                                                                                   " | grep = ", shell=True).decode("utf-8")
        Coordinate_file = Coordinate_file.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        print("Coordinate file was not provided")
        print("Exiting code")
        sys.exit()

    try:
        Parameter_file = subprocess.check_output("less " + str(args.Input_file) + " | grep Parameter_file"
                                                                                  " | grep = ", shell=True).decode("utf-8")
        Parameter_file = Parameter_file.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        Parameter_file = ''
        if Program == 'Tinker':
            print("Parameter file was not provided for Tinker")
            print("Exiting code")
            sys.exit()

    try:
        molecules_in_coord = subprocess.check_output("less " + str(args.Input_file) + " | grep molecules_in_coord"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        molecules_in_coord = int(molecules_in_coord.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        if Program == 'Test':
            molecules_in_coord = 1
        else:
            print("Number of molecules in system was not specified")
            print("Exiting code")
            sys.exit()

    try:
        properties_to_save_temp = subprocess.check_output("less " + str(args.Input_file) + " | grep properties_to_save"
                                                                                           " | grep = ", shell=True).decode("utf-8")
        properties_to_save_temp = properties_to_save_temp.split('=')[1].strip().split(',')
        properties_to_save = []
        for i in range(len(properties_to_save_temp)):
            if properties_to_save_temp[i] in ['G', 'S', 'T', 'P', 'Av', 'V', 'h', 'U']:
                properties_to_save.append(properties_to_save_temp[i])
            else:
                print("The following input is not a choice in properites: " + properties_to_save_temp[i])
    except subprocess.CalledProcessError as grepexc:
        properties_to_save = ['G', 'T']

    try:
        NumAnalysis_method = subprocess.check_output("less " + str(args.Input_file) + " | grep NumAnalysis_method"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        NumAnalysis_method = NumAnalysis_method.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        NumAnalysis_method = 'Euler'
        if Method in ['GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print("Numerical analysis method  was not specified")
            print("... Using default method: Euler")

    try:
        NumAnalysis_step = subprocess.check_output("less " + str(args.Input_file) + " | grep NumAnalysis_step"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        NumAnalysis_step = float(NumAnalysis_step.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        NumAnalysis_step = 150.
        if Method in ['GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print("Numerical analysis step size  was not specified")
            print("... Using default step size: " + str(NumAnalysis_step))

    try:
        LocGrd_Vol_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep LocGrd_Vol_FracStep"
                                                                                       " | grep = ", shell=True).decode("utf-8")
        LocGrd_Vol_FracStep = float(LocGrd_Vol_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        LocGrd_Vol_FracStep = 0.

    try:
        LocGrd_CMatrix_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep LocGrd_CMatrix_FracStep"
                                                                                     " | grep = ", shell=True).decode("utf-8")
        LocGrd_CMatrix_FracStep = np.array(LocGrd_CMatrix_FracStep.split('=')[1].strip().split(',')).astype(float)
    except subprocess.CalledProcessError as grepexc:
        LocGrd_CMatrix_FracStep = np.zeros(6)

    try:
        StepWise_Vol_StepFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep StepWise_Vol_StepFrac"
                                                                                         " | grep = ", shell=True).decode("utf-8")
        StepWise_Vol_StepFrac = float(StepWise_Vol_StepFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_StepFrac = 1.5e-03

    try:
        StepWise_Vol_LowerFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "StepWise_Vol_LowerFrac"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        StepWise_Vol_LowerFrac = float(StepWise_Vol_LowerFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_LowerFrac = 0.99

    try:
        StepWise_Vol_UpperFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "StepWise_Vol_UpperFrac"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        StepWise_Vol_UpperFrac = float(StepWise_Vol_UpperFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_UpperFrac = 1.02

    try:
        Gruneisen_Vol_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "Gruneisen_Vol_FracStep"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        Gruneisen_Vol_FracStep = float(Gruneisen_Vol_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gruneisen_Vol_FracStep = 1.5e-03

    try:
        Gruneisen_Lat_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "Gruneisen_Lat_FracStep"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        Gruneisen_Lat_FracStep = float(Gruneisen_Lat_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gruneisen_Lat_FracStep = 1.5e-03

    try:
        Wavenum_Tol = subprocess.check_output("less " + str(args.Input_file) + " | grep Wavenum_Tol"
                                                                               " | grep = ", shell=True).decode("utf-8")
        Wavenum_Tol = float(Wavenum_Tol.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Wavenum_Tol = -1.

    try:
        Gradient_MaxTemp = subprocess.check_output("less " + str(args.Input_file) + " | grep Gradient_MaxTemp"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        Gradient_MaxTemp = float(Gradient_MaxTemp.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gradient_MaxTemp = 300.

    try:
        Aniso_LocGrad_Type = subprocess.check_output("less " + str(args.Input_file) + " | grep Aniso_LocGrad_Type"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        Aniso_LocGrad_Type = Aniso_LocGrad_Type.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        Aniso_LocGrad_Type = '6D'

    try:
        min_RMS_gradient = subprocess.check_output("less " + str(args.Input_file) + " | grep min_RMS_gradient"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        min_RMS_gradient = float(min_RMS_gradient.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        min_RMS_gradient = 0.01

    try:
        cp2kroot =  subprocess.check_output("less " + str(args.Input_file) + " | grep cp2kroot"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        cp2kroot = (cp2kroot.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        cp2kroot = 'BNZ_NMA_p2'

    if pressure_scan == False: 
        Temperature_Lattice_Dynamics(Temperature=Temperature,
                                     Pressure=Pressure,
                                     Method=Method,
                                     Program=Program,
                                     Output=Output,
                                     Coordinate_file=Coordinate_file,
                                     Parameter_file=Parameter_file,
                                     molecules_in_coord=molecules_in_coord,
                                     properties_to_save=properties_to_save,
                                     NumAnalysis_method=NumAnalysis_method,
                                     NumAnalysis_step=NumAnalysis_step,
                                     LocGrd_Vol_FracStep=LocGrd_Vol_FracStep,
                                     LocGrd_CMatrix_FracStep=LocGrd_CMatrix_FracStep,
                                     StepWise_Vol_StepFrac=StepWise_Vol_StepFrac,
                                     StepWise_Vol_LowerFrac=StepWise_Vol_LowerFrac,
                                     StepWise_Vol_UpperFrac=StepWise_Vol_UpperFrac,
                                     Statistical_mechanics=Statistical_mechanics,
                                     Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep,
                                     Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep,
                                     Wavenum_Tol=Wavenum_Tol,
                                     Gradient_MaxTemp=Gradient_MaxTemp,
                                     Aniso_LocGrad_Type=Aniso_LocGrad_Type,
                                     min_RMS_gradient=min_RMS_gradient,
                                     cp2kroot=cp2kroot)
    
    else:
        try:
            eq_of_state = subprocess.check_output("less " + str(args.Input_file) + " | grep eq_of_state | grep = ", shell=True).decode("utf-8")
            eq_of_state = str(eq_of_state).split('=')[1].strip()
            if eq_of_state not in ['None', 'Murnaghan', 'Birch-Murnaghan', 'Rose-Vinet']:
                print("Input eq_of_state is not supported. Please select from the following:")
                print("   None, Murnaghan, Birch-Murnaghan, Rose-Vinet")
                print("Exiting code")
                sys.exit()
        except subprocess.CalledProcessError as grepexc:
            print("No method was selected, will continue using Murnaghan EOS")
            eq_of_state = 'Murnaghan'

        try:
            gru_from_0T_0P = subprocess.check_output("less " + str(args.Input_file) + " | grep gru_from_0T_0P"
                                                                                      " | grep = ", shell=True).decode("utf-8")
            gru_from_0T_0P = str(gru_from_0T_0P).split('=')[1].strip()
            if gru_from_0T_0P == 'True':
                gru_from_0T_0P = True
            elif gru_from_0T_0P == 'False':
                gru_from_0T_0P = False
            else:
                print("Input option of ", gru_from_0T_0P, " for gru_from_0T_0P is not supported. Will use True.")
                gru_from_0T_0P = True
        except subprocess.CalledProcessError as grepexc:
            gru_from_0T_0P = True


        if Statistical_mechanics == 'Quantum':
            print("Warning! The scanning of multiple pressures is not yet supported in this code for Qunantum Mechanics.")
            print("... Please contact Nate Abraham (nate.abraham@colorado.edu) with ways to perform this.")
        else:
            ps.pressure_setup(Temperature=Temperature,
                              Pressure=Pressure,
                              Method=Method,
                              Program=Program,
                              Output=Output,
                              Coordinate_file=Coordinate_file,
                              Parameter_file=Parameter_file,
                              molecules_in_coord=molecules_in_coord,
                              properties_to_save=properties_to_save,
                              NumAnalysis_method=NumAnalysis_method,
                              NumAnalysis_step=NumAnalysis_step,
                              LocGrd_Vol_FracStep=LocGrd_Vol_FracStep,
                              LocGrd_CMatrix_FracStep=LocGrd_CMatrix_FracStep,
                              StepWise_Vol_StepFrac=StepWise_Vol_StepFrac,
                              StepWise_Vol_LowerFrac=StepWise_Vol_LowerFrac,
                              StepWise_Vol_UpperFrac=StepWise_Vol_UpperFrac,
                              Statistical_mechanics=Statistical_mechanics,
                              Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep,
                              Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep,
                              Wavenum_Tol=Wavenum_Tol,
                              Gradient_MaxTemp=Gradient_MaxTemp,
                              Aniso_LocGrad_Type=Aniso_LocGrad_Type,
                              min_RMS_gradient=min_RMS_gradient,
                              eq_of_state=eq_of_state,
                              gru_from_0T_0P=gru_from_0T_0P,
                              cp2kroot=cp2kroot)


