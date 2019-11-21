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
import program_specific_functions as psf 

path = os.path.realpath(__file__).strip('Run_LatticeDynamics.py')


def temperature_lattice_dynamics(inputs, data, input_file='input.yaml'):
    # Geometry and lattice optimizing the input structure
    if (inputs.tinker_xtalmin == True) and (inputs.program == 'Tinker'):
        Ex.tinker_xtalmin(inputs)
        inputs.tinker_xtalmin = False

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

        if all(wavenumbers[:3] < inputs.wavenumber_tolerance) and all(wavenumbers[:3] > -1. * inputs.wavenumber_tolerance):
            print("   All wavenumbers are greater than tolerance of: " + str(inputs.wavenumber_tolerance) + " cm^-1")
            properties = Pr.Properties_with_Temperature(inputs, inputs.coordinate_file, wavenumbers)
            print("   All properties have been saved in " + inputs.output + "_raw.npy")
            np.save(inputs.output + '_raw', properties)
            print("   Saving user specified properties in indipendent files:")
            Pr.Save_Properties(inputs, properties)
        exit()
    else:
        if os.path.isdir('Cords') != True:
            print("Creating directory 'Cords/' to store structures along Gibbs free energy path")
            subprocess.call(['mkdir', 'Cords'])

    # Expanding the crystal with the zero point energy
    if inputs.statistical_mechanics == 'Quantum':
        if any(inputs.gradient_matrix_fractions != 0.):
            crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
                psf.Lattice_parameters(inputs.program, inputs.coordinate_file)))
            LocGrd_dC = np.absolute(inputs.gradient_matrix_fractions * crystal_matrix_array)
            for i in range(len(LocGrd_dC)):
                if LocGrd_dC[i] == 0.:
                    LocGrd_dC[i] = inputs.gradient_matrix_fractions[i]
        else:
            LocGrd_dC = Ss.anisotropic_gradient_settings(inputs, data, input_file)

        TNA.anisotropic_gradient_expansion_ezp(inputs, LocGrd_dC)
        inputs.coordinate_file = 'ezp_minimum' + psf.assign_coordinate_file_ending(inputs.program)

    # Running through QHA
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
    elif (inputs.method == 'GaQ') or (inputs.method == 'GaQg'):
        if inputs.statistical_mechanics == 'Classical':
            if any(inputs.gradient_matrix_fractions != 0.):
                crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(
                    psf.Lattice_parameters(inputs.program, inputs.coordinate_file)))
                LocGrd_dC = np.absolute(inputs.gradient_matrix_fractions * crystal_matrix_array)
                for i in range(len(LocGrd_dC)):
                    if LocGrd_dC[i] == 0.:
                        LocGrd_dC[i] = inputs.gradient_matrix_fractions[i]
            else:
                LocGrd_dC = Ss.anisotropic_gradient_settings(inputs, data, input_file)

        if inputs.anisotropic_type != '1D':
            print("Performing Gradient Anisotropic Quasi-Harmonic Approximation")
            properties = TNA.Anisotropic_Gradient_Expansion(inputs, LocGrd_dC)
            print("   Saving user specified properties in independent files:")
            Pr.Save_Properties(inputs, properties)
        else:
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
        default_input = yaml.load(default_file, Loader=yaml.FullLoader)

    # Loads in a ymal file
    with open(file_path, "r") as input_file:
        data = yaml.load(input_file, Loader=yaml.FullLoader)
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
        self.tinker_xtalmin = bool(data['tinker']['xtalmin'])
        self.tinker_xtalmin_tol = data['tinker']['xtalmin_tol']
        self.cp2k_root = data['cp2k']['root']
        if type(data['properties_to_save']) != type(None):
            self.properties_to_save = data['properties_to_save']
        else:
            self.properties_to_save = []

        self.gradient_numerical_method = data['gradient']['numerical_method']

        self.gradient_max_temperature = data['gradient']['max_temperature']
        self.gradient_numerical_step = data['gradient']['numerical_step']
        self.zeta_numerical_step = data['gradient']['zeta_numerical_step']
        if self.method in ['GiQ', 'GiQg', 'GaQ', 'GaQg']:
            # Making sure all temperature outputs are less than the max temperature sampled during integration
            if any(self.temperature > self.gradient_max_temperature):
                self.temperature = self.temperature[np.where(self.temperature <= self.gradient_max_temperature)]
                print("WARNING: Any 'temperature' exceeding 'gradient,max_temperature' will be ignored")

            # Making sure the numerical step size is a factor of the max temperature for numerical integration
            if self.gradient_max_temperature % self.gradient_numerical_step != 0:
                print("ERROR: the 'gradient,numerical_step' must be a factor of 'gradient,max_temperature'")
                sys.exit()

        self.gradient_vol_fraction = data['gradient']['vol_fraction']
        if data['gradient']['matrix_fractions'] is None:
            self.gradient_matrix_fractions = np.zeros(6)
        elif len(data['gradient']['matrix_fractions']) == 6:
            self.gradient_matrix_fractions = np.array(data['gradient']['matrix_fractions'])
        else:
            self.gradient_matrix_fractions = np.zeros(6)
        self.anisotropic_type = data['gradient']['anisotropic_type']
        self.stepwise_volume_fraction_stepsize = data['stepwise']['volume_fraction_stepsize']
        self.stepwise_volume_fraction_lower = data['stepwise']['volume_fraction_lower']
        self.stepwise_volume_fraction_upper = data['stepwise']['volume_fraction_upper']
        self.gruneisen_volume_fraction_stepsize = data['gruneisen']['volume_fraction_stepsize']
        self.gruneisen_matrix_strain_stepsize = data['gruneisen']['matrix_strain_stepsize']
        self.wavenumber_tolerance = np.absolute(float(data['wavenumber_tolerance']))
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
        temperature_lattice_dynamics(inputs, data, input_file=args.Input_file)
    else:
        print("Nate needs to re-setup pressure capabilities")

