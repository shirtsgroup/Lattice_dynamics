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
import yaml

path = os.path.realpath(__file__).strip('Run_LatticeDynamics.py')


def temperature_lattice_dynamics(inputs):
    if inputs.method == 'HA':
        print("Performing Harmonic Approximation")
        # Running the Harmonic Approximation
        if os.path.isfile(inputs.output + '_' + inputs.method + '_WVN.npy'):
            wavenumbers = np.load(inputs.output + '_' + inputs.method + '_WVN.npy')
            print("   Importing wavenumbers from:" + inputs.output + '_' + inputs.method + '_WVN.npy')
        else:
            print("   Computing wavenumbers of coordinate file")
            wavenumbers = Wvn.Call_Wavenumbers(inputs, Coordinate_file=inputs.coordinate_file,
                                               Parameter_file=inputs.tinker_parameter_file)
            np.save(inputs.output + '_' + inputs.method + '_WVN', wavenumbers)

        if all(wavenumbers > inputs.wavenumber_tolerance):
            print("   All wavenumbers are greater than tolerance of: " + str(inputs.wavenumber_tolerance) + " cm^-1")
            properties = Pr.Properties_with_Temperature(inputs, inputs.coordinate_file, wavenumbers)
            print("   All properties have been saved in " + inputs.output + "_raw.npy")
            np.save(inputs.output + '_raw', properties)
            print("   Saving user specified properties in indipendent files:")
            Pr.Save_Properties(inputs, properties)

    else:
        if os.path.isdir('Cords') != True:
            print("Creating directory 'Cords/' to store structures along Gibbs free energy path")
            subprocess.call(['mkdir', 'Cords'])

    if (inputs.method == 'SiQ') or (inputs.method == 'SiQg'):
        # Stepwise Isotropic QHA
        print("Performing Stepwise Isotropic Quasi-Harmonic Approximation")
        properties = TNA.Isotropic_Stepwise_Expansion(inputs)

        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(inputs, properties)

    elif (inputs.method == 'GiQ') or (inputs.method == 'GiQg'):
        if inputs.gradient_vol_fraction == (0. or None):
            LocGrd_dV = Ss.isotropic_gradient_settings(inputs)

        else:
            V_0 = Pr.Volume(Program=inputs.program, Coordinate_file=inputs.coordinate_file)
            LocGrd_dV = inputs.gradient_vol_fraction * V_0
        # Gradient Isotropic QHA
        print("Performing Gradient Isotropic Quasi-Harmonic Approximation")
        properties = TNA.Isotropic_Gradient_Expansion(inputs, LocGrd_dV)

        print("   Saving user specified properties in indipendent files:")
        Pr.Save_Properties(inputs, properties)

    elif ((inputs.method == 'GaQ') or (inputs.method == 'GaQg')) and (inputs.anisotropic_type != '1D'):
        if any(inputs.gradient_matrix_fractions != 0.):
            crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
                Pr.Lattice_parameters(inputs.program, inputs.coordinate_file)))
            LocGrd_dC = np.absolute(inputs.gradient_matrix_fractions * crystal_matrix_array)
            for i in range(len(LocGrd_dC)):
                if LocGrd_dC[i] == 0.:
                    LocGrd_dC[i] = inputs.gradient_matrix_fractions[i]
        else:
            LocGrd_dC = Ss.anisotropic_gradient_settings(inputs)
        print("Performing Gradient Anisotropic Quasi-Harmonic Approximation")
        properties = TNA.Anisotropic_Gradient_Expansion(inputs, LocGrd_dC)
        print("   Saving user specified properties in independent files:")
        Pr.Save_Properties(inputs, properties)

    elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg') and (inputs.anisotropic_type == '1D'):
        if any(inputs.gradient_matrix_fractions != 0.):
            crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
                Pr.Lattice_parameters(inputs.program, inputs.coordinate_file)))
            LocGrd_dC = np.absolute(inputs.gradient_matrix_fractions * crystal_matrix_array)
        else:
            LocGrd_dC = Ss.anisotropic_gradient_settings(inputs)
        print("Performing 1D-Gradient Anisotropic Quasi-Harmonic Approximation")
        properties = TNA.Anisotropic_Gradient_Expansion_1D(inputs, LocGrd_dC)
        print("   Saving user specified properties in independent files:")
        Pr.Save_Properties(inputs, properties)

    elif inputs.method == 'SaQply':
        print("Performing Quasi-Anisotropic Quasi-Harmonic Approximation")
        properties = TNA.stepwise_expansion(inputs)
        print("   Saving user specified properties in independent files:")
        Pr.Save_Properties(inputs, properties)
    print("Lattice dynamic calculation is complete!")


"""
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
"""
def old_input_file(Input_file, data):
    try:
        data['coordinate_file'] = subprocess.check_output("less " + str(args.Input_file) + " | grep Coordinate_file"
                                                                                   " | grep = ", shell=True).decode("utf-8")
        data['coordinate_file'] = data['coordinate_file'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        print("Coordinate file was not provided")
        print("Exiting code")
        sys.exit()

    try:
        data['method'] = subprocess.check_output("less " + str(Input_file) + " | grep Method | grep = ", shell=True).decode("utf-8")
        data['method'] = str(data['method']).split('=')[1].strip()
        if data['method'] not in ['HA', 'SiQ', 'SiQg', 'GiQ', 'GiQg', 'GaQ', 'GaQg', 'SaQply']:
            print("Input method is not supported. Please select from the following:")
            print("   HA, SiQ, SiQg, GiQ, GiQg, GaQ, GaQg")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("No method was selected")
        print("Exiting code")
        sys.exit()

    try:
        data['program'] = subprocess.check_output("less " + str(Input_file) + " | grep Program | grep = ", shell=True).decode("utf-8")
        data['program'] = data['program'].split('=')[1].strip()
        if data['program'] not in ['Tinker', 'Test', 'CP2K', 'QE']:
            print("Input program is not supported. Please select from the following:")
            print("   Tinker, Test")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("No program was selected")
        print("Exiting code")
        sys.exit()

    try:
        data['statistical_mechanics'] = subprocess.check_output("less " + str(Input_file) + " | grep Statistical_mechanics"
                                                                                         " | grep = ", shell=True).decode("utf-8")
        data['statistical_mechanics'] = data['statistical_mechanics'].split('=')[1].strip()
        if data['statistical_mechanics'] not in ['Classical', 'Quantum']:
            print("Input statistical mechanics is not supported. Please select from the following:")
            print("   Classical, Quantum")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        print("Statistical mechnics was not specified")
        print("Exiting code")
        sys.exit()

    try:
        data['temperature'] = subprocess.check_output("less " + str(Input_file) + " | grep Temperature"
                                                                               " | grep = ", shell=True).decode("utf-8")
        data['temperature'] = data['temperature'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        print('Using default temperatures')

    try:
        data['pressure'] = subprocess.check_output("less " + str(Input_file) + " | grep Pressure | grep = ", shell=True).decode("utf-8")
        data['pressure'] = np.array(data['pressure'].split('=')[1].strip().split(',')).astype(float)
    except subprocess.CalledProcessError as grepexc:
        print("No pressure was selected, using default pressure")

    try:
        data['output'] = subprocess.check_output("less " + str(Input_file) + " | grep Output | grep = ", shell=True).decode("utf-8")
        data['output'] = data['output'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['tinker']['parameter_file'] = subprocess.check_output("less " + str(Input_file) + " | grep Parameter_file"
                                                                                  " | grep = ", shell=True).decode("utf-8")
        data['tinker']['parameter_file'] = data['tinker']['parameter_file'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['number_of_molecules'] = subprocess.check_output("less " + str(Input_file) + " | grep molecules_in_coord"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        data['number_of_molecules'] = int(data['number_of_molecules'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['properties_to_save'] = subprocess.check_output("less " + str(Input_file) + " | grep properties_to_save"
                                                                                           " | grep = ", shell=True).decode("utf-8")
        data['properties_to_save'] = data['properties_to_save'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gradient']['numerical_method'] = subprocess.check_output("less " + str(Input_file) + " | grep NumAnalysis_method"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        data['gradient']['numerical_method'] = data['gradient']['numerical_method'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gradient']['numerical_step'] = subprocess.check_output("less " + str(Input_file) + " | grep NumAnalysis_step"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        data['gradient']['numerical_step'] = float(data['gradient']['numerical_step'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gradient']['vol_fraction'] = subprocess.check_output("less " + str(Input_file) + " | grep LocGrd_Vol_FracStep"
                                                                                       " | grep = ", shell=True).decode("utf-8")
        data['gradient']['vol_fraction'] = float(data['gradient']['vol_fraction'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gradient']['matrix_fractions'] = subprocess.check_output("less " + str(Input_file) + " | grep LocGrd_CMatrix_FracStep"
                                                                                     " | grep = ", shell=True).decode("utf-8")
        data['gradient']['matrix_fractions'] = np.array(data['gradient']['matrix_fractions'].split('=')[1].strip().split(',')).astype(float)
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['stepwise']['volume_fraction_stepsize'] = subprocess.check_output("less " + str(Input_file) + " | grep StepWise_Vol_StepFrac"
                                                                                         " | grep = ", shell=True).decode("utf-8")
        data['stepwise']['volume_fraction_stepsize'] = float(data['stepwise']['volume_fraction_stepsize'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['stepwise']['volume_fraction_lower'] = subprocess.check_output("less " + str(Input_file) + " | grep "
                                                                                          "StepWise_Vol_LowerFrac"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        data['stepwise']['volume_fraction_lower'] = float(data['stepwise']['volume_fraction_lower'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['stepwise']['volume_fraction_upper'] = subprocess.check_output("less " + str(Input_file) + " | grep "
                                                                                          "StepWise_Vol_UpperFrac"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        data['stepwise']['volume_fraction_upper'] = float(data['stepwise']['volume_fraction_upper'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gruneisen']['volume_fraction_stepsize'] = subprocess.check_output("less " + str(Input_file) + " | grep "
                                                                                          "Gruneisen_Vol_FracStep"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        data['gruneisen']['volume_fraction_stepsize'] = float(data['gruneisen']['volume_fraction_stepsize'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gruneisen']['matrix_strain_stepsize'] = subprocess.check_output("less " + str(Input_file) + " | grep "
                                                                                          "Gruneisen_Lat_FracStep"
                                                                                          " | grep = ", shell=True).decode("utf-8")
        data['gruneisen']['matrix_strain_stepsize'] = float(data['gruneisen']['matrix_strain_stepsize'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['wavenumber_tolerance'] = subprocess.check_output("less " + str(Input_file) + " | grep Wavenum_Tol"
                                                                               " | grep = ", shell=True).decode("utf-8")
        data['wavenumber_tolerance'] = float(data['wavenumber_tolerance'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        data['wavenumber_tolerance'] = -1.

    try:
        data['gradient']['max_temperature'] = subprocess.check_output("less " + str(Input_file) + " | grep Gradient_MaxTemp"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        data['gradient']['max_temperature'] = float(data['gradient']['max_temperature'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['gradient']['anisotropic_type'] = subprocess.check_output("less " + str(Input_file) + " | grep Aniso_LocGrad_Type"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        data['gradient']['anisotropic_type'] = data['gradient']['anisotropic_type'].split('=')[1].strip()
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['min_rms_gradient'] = subprocess.check_output("less " + str(Input_file) + " | grep min_RMS_gradient"
                                                                                    " | grep = ", shell=True).decode("utf-8")
        data['min_rms_gradient'] = float(data['min_rms_gradient'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['cp2k']['root'] = subprocess.check_output("less " + str(Input_file) + " | grep cp2kroot"
                                                                                      " | grep = ", shell=True).decode("utf-8")
        data['cp2k']['root'] = (data['cp2k']['root'].split('=')[1].strip())
    except subprocess.CalledProcessError as grepexc:
        pass

    try:
        data['eq_of_state'] = subprocess.check_output("less " + str(Input_file) + " | grep eq_of_state | grep = ", shell=True).decode("utf-8")
        data['eq_of_state'] = str(data['eq_of_state']).split('=')[1].strip()
        if data['eq_of_state'] not in ['None', 'Murnaghan', 'Birch-Murnaghan', 'Rose-Vinet']:
            print("Input eq_of_state is not supported. Please select from the following:")
            print("   None, Murnaghan, Birch-Murnaghan, Rose-Vinet")
            print("Exiting code")
            sys.exit()
    except subprocess.CalledProcessError as grepexc:
        pass
    return data

def setdefault(input_data, default_values):
    # Function to fill in the input_data if the default values are not set
    for k in default_values:
        if isinstance(default_values[k], dict):
            setdefault(input_data.setdefault(k, {}), default_values[k])
        else:
            input_data.setdefault(k, default_values[k])

def yaml_loader(file_path):
    # Load in the default values
    with open(path + 'default.yaml', "r") as default_file:
        default_input = yaml.load(default_file)

    if file_path.split('.')[1] == 'inp':
        data = old_input_file(file_path, default_input)
    else:
        # Loads in a ymal file
        with open(file_path, "r") as input_file:
            data = yaml.load(input_file)
        # Setting the default values if not specified
        setdefault(data, default_input)
    return data

class Inputs:
    def __init__(self, data):
        self.temperature = np.array(data['temperature'].split(',')).astype(float)
        if len(str(data['pressure']).split(',')) == 1:
            self.pressure = data['pressure']
            self.pressure_scan = False
        else:
            self.pressure = np.array(data['pressure'].split(',')).astype(float)
            self.pressure_scan = True
        self.method = data['method']
        self.output = data['output']
        self.statistical_mechanics = data['statistical_mechanics']
        self.coordinate_file = data['coordinate_file']
        self.number_of_molecules = data['number_of_molecules']
        self.program = data['program']
        self.tinker_parameter_file = data['tinker']['parameter_file']
        self.cp2k_root = data['cp2k']['root']
        self.properties_to_save = data['properties_to_save']
        self.gradient_numerical_method = data['gradient']['numerical_method']
        self.gradient_numerical_step = data['gradient']['numerical_step']
        self.gradient_max_temperature = data['gradient']['max_temperature']
        self.gradient_vol_fraction = data['gradient']['vol_fraction']
        self.gradient_matrix_fractions = data['gradient']['matrix_fractions']
        self.anisotropic_type = data['gradient']['anisotropic_type']
        self.stepwise_volume_fraction_stepsize = data['stepwise']['volume_fraction_stepsize']
        self.stepwise_volume_fraction_lower = data['stepwise']['volume_fraction_lower']
        self.stepwise_volume_fraction_upper = data['stepwise']['volume_fraction_upper']
        self.gruneisen_volume_fraction_stepsize = data['gruneisen']['volume_fraction_stepsize']
        self.gruneisen_matrix_strain_stepsize = data['gruneisen']['matrix_strain_stepsize']
        self.wavenumber_tolerance = data['wavenumber_tolerance']
        self.min_rms_gradient = data['min_rms_gradient']
        self.eq_of_state = data['eq_of_state']
        self.poly_order = data['poly_order']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate free energies as a function of T using lattice dynamics')
    parser.add_argument('-i', '--input_file', dest='Input_file', default='input_test.py',
                        help='Input file containing all parameters for the run')
    parser.add_argument('-D', '--Start_Fresh', action='store_true',
                        help='Removes any files from previous runs')
    args = parser.parse_args()

    # Removing all old files if flagged
    if args.Start_Fresh:
        subprocess.call(['rm -rf Cords numerical_checks.out minimization.out out_*'], shell=True)

    # Importing user specified inputs from yaml file
    data = yaml_loader(args.Input_file)
    inputs = Inputs(data)
    if not inputs.pressure_scan:
        temperature_lattice_dynamics(inputs)
    else:
        print("Nate needs to re-setup pressure capabilities")

