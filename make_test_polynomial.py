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
import pylab as plt
import scipy.optimize


Program = 'Tinker'
Coordinate_file = 'molecule.xyz'

Parameter_file = 'keyfile.key'
molecules_in_coord = 4
min_RMS_gradient = 0.00001
Output = 'out'

n_samples = 200

file_ending = Ex.assign_file_ending(Program)


lattice_parameters = Pr.Lattice_parameters(Program, Coordinate_file)
fractional_change = np.random.normal(0., 0.02, (n_samples,6))

dlattice_parameters = (fractional_change * lattice_parameters)


U = np.zeros(n_samples + 1)
U[-1] = Pr.Potential_energy(Program, Coordinate_file=Coordinate_file, Parameter_file=Parameter_file)

for i in range(n_samples):
    Ex.Expand_Structure(Coordinate_file, Program, 'lattice_parameters', molecules_in_coord,
                        Output, min_RMS_gradient, Parameter_file=Parameter_file,
                        dlattice_parameters=dlattice_parameters[i])
    U[i] = Pr.Potential_energy(Program, Coordinate_file=Output + file_ending, Parameter_file=Parameter_file)

np.save('U', U)

U = np.load('U.npy')
new_lattice_parameters = dlattice_parameters + lattice_parameters
new_lattice_parameters = np.append(new_lattice_parameters, lattice_parameters).reshape(n_samples + 1, 6)
a = new_lattice_parameters[:, 0]
b = new_lattice_parameters[:, 1]
c = new_lattice_parameters[:, 2]
alpha = new_lattice_parameters[:, 3]
beta = new_lattice_parameters[:, 4]
gamma = new_lattice_parameters[:, 5]


A = np.array([a*0 + 1, 
              a, b, c, alpha, beta, gamma, 
              a**2, b**2, c**2, alpha**2, beta**2, gamma**2,
              a*b, a*c, a*alpha, a*beta, a*gamma,
              b*c, b*alpha, b*beta, b*gamma,
              c*alpha, c*beta, c*gamma,
              alpha*beta, alpha*gamma,
              beta*gamma])

coeff, r, rank, s = np.linalg.lstsq(A.T, U)
print(coeff)


