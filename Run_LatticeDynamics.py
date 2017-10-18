#!/usr/bin/env python

import os
import sys
import numpy as np
import subprocess
import ThermodynamicProperties as Pr
import Wavenumbers as Wvn
import Thermal_NumericalAnalysis as TNA

def Lattice_Dynamics(Temperature=[0.0, 25.0, 50.0, 75.0, 100.0], Pressure=1., Method='HA', Program='Test',
                     Output='out', Coordinate_file='molecule.xyz', Parameter_file='keyfile.key',
                     molecules_in_coord=1, properties_to_save=['G', 'T'], NumAnalysis_method='RK4',
                     NumAnalysis_step=25.0,
                     LocGrd_Vol_FracStep=3e-02,
                     LocGrd_LatParam_FracStep=5e-05, StepWise_Vol_StepFrac=1.5e-3,
                     StepWise_Vol_LowerFrac=0.97, StepWise_Vol_UpperFrac=1.16,
                     Statistical_mechanics='Classical', Gruneisen_Vol_FracStep=1.5e-3, Gruneisen_Lat_FracStep=1.0e-3,
                     Wavenum_Tol=-1., Gradient_MaxTemp=300.0, Aniso_LocGrad_Type='73', min_RMS_gradient=0.01):

    Temperature = np.array(Temperature).astype(float)
    if Method == 'HA':
        print "Performing Harmonic Approximation"
        # Running the Harmonic Approximation
        if os.path.isfile(Output + '_' + Method + '_WVN.npy'):
            wavenumbers = np.load(Output + '_' + Method + '_WVN.npy')
            print "   Importing wavenumbers from:" + Output + '_' + Method + '_WVN.npy'
        else:
            print "   Computing wavenumbers of coordinate file"
            wavenumbers = Wvn.Call_Wavenumbers(Method, min_RMS_gradient, Program=Program, Coordinate_file=Coordinate_file,
                                               Parameter_file=Parameter_file)
            np.save(Output + '_' + Method + '_WVN', wavenumbers)

        if all(wavenumbers > Wavenum_Tol):
            print "   All wavenumbers are greater than tolerance of: " + str(Wavenum_Tol) + " cm^-1"
            properties = Pr.Properties_with_Temperature(Coordinate_file, wavenumbers, Temperature, Pressure, Program,
                                                        Statistical_mechanics, molecules_in_coord,
                                                        Parameter_file=Parameter_file)
            print "   All properties have been saved in " + Output + "_raw.npy"
            np.save(Output + '_raw', properties)
            print "   Saving user specified properties in indipendent files:"
            Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
            print "Harmonic Approximation is complete!"
    else:
        if os.path.isdir('Cords') != True:
            print "Creating directory 'Cords/' to store structures along Gibbs free energy path"
            os.system('mkdir Cords')

    if (Method == 'SiQ') or (Method == 'SiQg'):
        # Stepwise Isotropic QHA
        print "Performing Stepwise Isotropic Quasi-Harmonic Approximation"
        properties = TNA.Isotropic_Stepwise_Expansion(StepWise_Vol_StepFrac, StepWise_Vol_LowerFrac,
                                                      StepWise_Vol_UpperFrac, Coordinate_file, Program, Temperature,
                                                      Pressure, Output, Method, molecules_in_coord, Wavenum_Tol,
                                                      Statistical_mechanics, min_RMS_gradient,
                                                      Parameter_file=Parameter_file,
                                                      Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep)
        print "   Saving user specified properties in indipendent files:"
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print "Stepwise Isotropic Quasi-Harmonic Approximation is complete!"

    if (Method == 'GiQ') or (Method == 'GiQg'):
        # Gradient Isotropic QHA
        print "Performing Gradient Isotropic Quasi-Harmonic Approximation"
        properties = TNA.Isotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method,
                                                      Gradient_MaxTemp, Pressure, LocGrd_Vol_FracStep,
                                                      Statistical_mechanics, NumAnalysis_step, NumAnalysis_method,
                                                      Temperature, min_RMS_gradient,
                                                      Parameter_file=Parameter_file,
                                                      Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep)
        print "   Saving user specified properties in indipendent files:"
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print "Gradient Isotropic Quasi-Harmonic Approximation is complete!"

    if (Method == 'GaQ') or (Method == 'GaQg'):
        print "Performing Gradient Anisotropic Quasi-Harmonic Approximation"
        properties = TNA.Ansotropic_Gradient_Expansion(Coordinate_file, Program, molecules_in_coord, Output, Method,
                                                       Gradient_MaxTemp, Pressure, LocGrd_LatParam_FracStep,
                                                       Statistical_mechanics, NumAnalysis_step,
                                                       NumAnalysis_method, Aniso_LocGrad_Type, Temperature,
                                                       min_RMS_gradient, Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep, 
                                                       Parameter_file=Parameter_file)
        print "   Saving user specified properties in indipendent files:"
        Pr.Save_Properties(properties, properties_to_save, Output, Method, Statistical_mechanics)
        print "Gradient Anisotropic Quasi-Harmonic Approximation is complete!"




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate free energies as a function of T using lattice dynamics')
    parser.add_argument('-i', '--input_file', dest='Input_file', default='input_test.py',
                        help='Input file containing all parameters for the run')

    args = parser.parse_args()

    try:
        Method = subprocess.check_output("less " + str(args.Input_file) + " | grep Method | grep = ", shell=True)
        Method = Method.split('=')[1].strip()
        if Method not in ['HA', 'SiQ', 'SiQg', 'GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print "Input method is not supported. Please select from the following:"
            print "   HA, SiQ, SiQg, GiQ, GiQg, GaQ, GaQg"
            print "Exiting code"
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print "No method was selected"
        print "Exiting code"
        sys.exit()

    try:
        Program = subprocess.check_output("less " + str(args.Input_file) + " | grep Program | grep = ", shell=True)
        Program = Program.split('=')[1].strip()
        if Program not in ['Tinker', 'Test']:
            print "Input program is not supported. Please select from the following:"
            print "   Tinker, Test"
            print "Exiting code"
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print "No program was selected"
        print "Exiting code"
        sys.exit()

    try:
        Statistical_mechanics = subprocess.check_output("less " + str(args.Input_file) + " | grep Statistical_mechanics"
                                                                                         " | grep = ", shell=True)
        Statistical_mechanics = Statistical_mechanics.split('=')[1].strip()
        if Statistical_mechanics not in ['Classical', 'Quantum']:
            print "Input statistical mechanics is not supported. Please select from the following:"
            print "   Classical, Quantum"
            print "Exiting code"
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print "Statistical mechnics was not specified"
        print "Exiting code"
        sys.exit()

    try:
        Temperature = subprocess.check_output("less " + str(args.Input_file) + " | grep Temperature"
                                                                               " | grep = ", shell=True)
        Temperature = np.array(Temperature.split('=')[1].strip().split(',')).astype(float)
    except subprocess.CalledProcessError as grepexc:
        if Method in ['HA', 'SiQ', 'SiQg']:
            Temperature = [0.0, 25.0, 50.0, 75.0, 100.0]
            print "No temperatures were selected, using default temperatures of:"
            print "   " + str(Temperature)
        else:
            Temperature = []

    try:
        Pressure = subprocess.check_output("less " + str(args.Input_file) + " | grep Pressure | grep = ", shell=True)
        Pressure = float(Pressure.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        print "No pressure was selected, using default pressure"
        Pressure = 1.

    try:
        Output = subprocess.check_output("less " + str(args.Input_file) + " | grep Output | grep = ", shell=True)
        Output = Output.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        Output = 'out'

    try:
        Coordinate_file = subprocess.check_output("less " + str(args.Input_file) + " | grep Coordinate_file"
                                                                                   " | grep = ", shell=True)
        Coordinate_file = Coordinate_file.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        print "Coordinate file was not provided"
        print "Exiting code"
        sys.exit()

    try:
        Parameter_file = subprocess.check_output("less " + str(args.Input_file) + " | grep Parameter_file"
                                                                                  " | grep = ", shell=True)
        Parameter_file = Parameter_file.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        Parameter_file = ''
        if Program == 'Tinker':
            print "Parameter file was not provided for Tinker"
            print "Exiting code"
            sys.exit()

    try:
        molecules_in_coord = subprocess.check_output("less " + str(args.Input_file) + " | grep molecules_in_coord"
                                                                                      " | grep = ", shell=True)
        molecules_in_coord = float(molecules_in_coord.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        if Program == 'Test':
            molecules_in_coord = 1
        else:
            print "Number of molecules in system was not specified"
            print "Exiting code"
            sys.exit()

    try:
        properties_to_save_temp = subprocess.check_output("less " + str(args.Input_file) + " | grep properties_to_save"
                                                                                           " | grep = ", shell=True)
        properties_to_save_temp = properties_to_save_temp.split('=')[1].strip().split(',')
        properties_to_save = []
        for i in range(len(properties_to_save_temp)):
            if properties_to_save_temp[i] in ['G', 'S', 'T', 'P', 'Av', 'V', 'h', 'U']:
                properties_to_save.append(properties_to_save_temp[i])
            else:
                print "The following input is not a choice in properites: " + properties_to_save_temp[i]
    except subprocess.CalledProcessError as grepexc:
        properties_to_save = ['G', 'T']

    try:
        NumAnalysis_method = subprocess.check_output("less " + str(args.Input_file) + " | grep NumAnalysis_method"
                                                                                      " | grep = ", shell=True)
        NumAnalysis_method = NumAnalysis_method.split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        NumAnalysis_method = 'Euler'
        if Method in ['GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print "Numerical analysis method  was not specified"
            print "... Using default method: Euler"

    try:
        NumAnalysis_step = subprocess.check_output("less " + str(args.Input_file) + " | grep NumAnalysis_step"
                                                                                    " | grep = ", shell=True)
        NumAnalysis_step = float(NumAnalysis_step.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        NumAnalysis_step = 150.
        if Method in ['GiQ', 'GiQg', 'GaQ', 'GaQg']:
            print "Numerical analysis step size  was not specified"
            print "... Using default step size: " + str(NumAnalysis_step)

    try:
        LocGrd_Vol_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep LocGrd_Vol_FracStep"
                                                                                       " | grep = ", shell=True)
        LocGrd_Vol_FracStep = float(LocGrd_Vol_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        LocGrd_Vol_FracStep = 3e-02

    try:
        LocGrd_LatParam_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                            "LocGrd_LatParam_FracStep"
                                                                                            " | grep = ", shell=True)
        LocGrd_LatParam_FracStep = float(LocGrd_LatParam_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        LocGrd_LatParam_FracStep = 5e-02

    try:
        StepWise_Vol_StepFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep StepWise_Vol_StepFrac"
                                                                                         " | grep = ", shell=True)
        StepWise_Vol_StepFrac = float(StepWise_Vol_StepFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_StepFrac = 1.5e-03

    try:
        StepWise_Vol_LowerFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "StepWise_Vol_LowerFrac"
                                                                                          " | grep = ", shell=True)
        StepWise_Vol_LowerFrac = float(StepWise_Vol_LowerFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_LowerFrac = 0.99

    try:
        StepWise_Vol_UpperFrac = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "StepWise_Vol_UpperFrac"
                                                                                          " | grep = ", shell=True)
        StepWise_Vol_UpperFrac = float(StepWise_Vol_UpperFrac.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        StepWise_Vol_UpperFrac = 1.02

    try:
        Gruneisen_Vol_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "Gruneisen_Vol_FracStep"
                                                                                          " | grep = ", shell=True)
        Gruneisen_Vol_FracStep = float(Gruneisen_Vol_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gruneisen_Vol_FracStep = 1.5e-03

    try:
        Gruneisen_Lat_FracStep = subprocess.check_output("less " + str(args.Input_file) + " | grep "
                                                                                          "Gruneisen_Lat_FracStep"
                                                                                          " | grep = ", shell=True)
        Gruneisen_Lat_FracStep = float(Gruneisen_Lat_FracStep.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gruneisen_Lat_FracStep = 1.5e-03

    try:
        Wavenum_Tol = subprocess.check_output("less " + str(args.Input_file) + " | grep Wavenum_Tol"
                                                                               " | grep = ", shell=True)
        Wavenum_Tol = float(Wavenum_Tol.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Wavenum_Tol = -1.

    try:
        Gradient_MaxTemp = subprocess.check_output("less " + str(args.Input_file) + " | grep Gradient_MaxTemp"
                                                                                    " | grep = ", shell=True)
        Gradient_MaxTemp = float(Gradient_MaxTemp.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Gradient_MaxTemp = 300.

    try:
        Aniso_LocGrad_Type = subprocess.check_output("less " + str(args.Input_file) + " | grep Aniso_LocGrad_Type"
                                                                                      " | grep = ", shell=True)
        Aniso_LocGrad_Type = int(Aniso_LocGrad_Type.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        Aniso_LocGrad_Type = 73

    try:
        min_RMS_gradient =  subprocess.check_output("less " + str(args.Input_file) + " | grep min_RMS_gradient"
                                                                                      " | grep = ", shell=True)
        min_RMS_gradient = float(min_RMS_gradient.split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        min_RMS_gradient = 0.01

    Lattice_Dynamics(Temperature=Temperature,
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
                     LocGrd_LatParam_FracStep=LocGrd_LatParam_FracStep,
                     StepWise_Vol_StepFrac=StepWise_Vol_StepFrac,
                     StepWise_Vol_LowerFrac=StepWise_Vol_LowerFrac,
                     StepWise_Vol_UpperFrac=StepWise_Vol_UpperFrac,
                     Statistical_mechanics=Statistical_mechanics,
                     Gruneisen_Vol_FracStep=Gruneisen_Vol_FracStep,
                     Gruneisen_Lat_FracStep=Gruneisen_Lat_FracStep,
                     Wavenum_Tol=Wavenum_Tol,
                     Gradient_MaxTemp=Gradient_MaxTemp,
                     Aniso_LocGrad_Type=Aniso_LocGrad_Type,
                     min_RMS_gradient=min_RMS_gradient)
