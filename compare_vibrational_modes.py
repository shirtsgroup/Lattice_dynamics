#!/usr/bin/env python
from __future__ import print_function
import subprocess
import sys
import os
import Run_LatticeDynamics
import Expand as Ex
import Wavenumbers as Wvn
import ThermodynamicProperties as Pr
import numpy as np
import scipy.optimize
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-p', dest='program', default='Tinker', help="Prgorams to use: Tinker (default), test, cp2k")
parser.add_argument('-f', dest='input_files', nargs='+', type=str, help="All coordinate files to compare vibrational spectra of")
parser.add_argument('-k', dest='parameter_file', default='keyfile.key', help="Parmameter file for specific program. Defaultf = 'keyfile.key'")
args = parser.parse_args()

number_of_modes = np.zeros(len(args.input_files))

# Determining the number of vibrational modes for each coordinate file
for i in range(len(args.input_files)):
    if args.program == 'Tinker':
        number_of_modes[i] = 3*Pr.Tinker_atoms_per_molecule(args.input_files[i], 1)

# Making sure all coordinate files have the same number of modes
if np.all(number_of_modes == number_of_modes[0]):
    number_of_modes = int(number_of_modes[0])
else:
    print("The coordinate files are different sizes, the modes cannot be compared.")
    sys.exit()

# Getting the eigenvectors and wavenumbers
eigenvectors = np.zeros((len(args.input_files), number_of_modes, number_of_modes))
wavenumbers = np.zeros((len(args.input_files), number_of_modes))

for i in range(len(args.input_files)):
    # If the wavenumbers/eigenvalues have already been computed, opening those 
    if args.program == 'Tinker':
        wavenumbers[i], eigenvectors[i] = Wvn.Tinker_Wavenumber_and_Vectors(args.input_files[i], args.parameter_file)

# Matching the modes with each other
wavenumbers_matched = np.zeros(np.shape(wavenumbers))
wavenumbers_matched[0] = wavenumbers[0]
matched_modes = np.zeros(np.shape(wavenumbers))
matched_modes[0] = np.arange(1, number_of_modes + 1)
Weight = np.zeros((len(args.input_files) - 1, number_of_modes))

for i in range(1, len(args.input_files)):
    z, weight = Wvn.match_modes(number_of_modes, eigenvectors[0], eigenvectors[i])
    matched_modes[i, 3:] = z[:, 1]
    for j in z:
        wavenumbers_matched[i, j[0]] = wavenumbers[i, j[1]]
        Weight[i - 1, j[0]] = weight[j[0] - 3, j[1] - 3]

print(Weight)
#print(wavenumbers_matched)
#print(matched_modes)




