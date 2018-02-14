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

Program = 'Test'
Coordinate_file = 'test.npy'

#Program = 'Tinker'
#Coordinate_file = 'molecule.xyz'

Parameter_file = 'keyfile.key'
molecules_in_coord = 4
min_RMS_gradient = 0.00001
Output = 'out'

file_ending = Ex.assign_file_ending(Program)

steps_exponents = np.arange(3.,7.)
steps = []
for i in range(len(steps_exponents)):
    steps.append(10.**(-steps_exponents[i]))
    steps.append(5*10.**(-steps_exponents[i]))
steps = np.sort(np.array(steps))

n_steps = len(steps)

U_0 = Pr.Potential_energy(Program, Coordinate_file=Coordinate_file, Parameter_file=Parameter_file)

U = np.zeros((6, n_steps))

crystal_matrix_array = Ex.triangle_crystal_matrix_to_array(Ex.Lattice_parameters_to_Crystal_matrix(Pr.Lattice_parameters(Program, Coordinate_file)))

for i in range(6):
    for k in range(n_steps):
        dlattice_matrix_array = np.zeros(6)
        dlattice_matrix_array[i] = crystal_matrix_array[i] * steps[k]
        dlattice_matrix = Ex.array_to_triangle_crystal_matrix(dlattice_matrix_array)

        Ex.Expand_Structure(Coordinate_file, Program, 'crystal_matrix', molecules_in_coord,
                            Output, min_RMS_gradient, Parameter_file=Parameter_file,
                            dcrystal_matrix=dlattice_matrix)
        U[i, k] = Pr.Potential_energy(Program, Coordinate_file=Output + file_ending, Parameter_file=Parameter_file)

c = ['r','r','r','b','b','b']
for i in range(6):
    plt.plot(np.log10(steps), U[i] - U_0, c=c[i], linestyle='--')
plt.xlabel('$\log({f})$',fontsize=22)
plt.ylabel('U [kcal/mol]',fontsize=22)
plt.tight_layout()
plt.show()



