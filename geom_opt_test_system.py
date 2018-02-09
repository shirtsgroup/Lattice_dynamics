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

# Setting a general starting point to run minimization of Test potential energy function
              # Lattice Vectors [Ang.]
x0 = np.array([10.,10.,10.,
              # Lattice Angles [Degrees]
               90.,90.,90.])

P = 1.

# PV + U energy
def U_PV(lp):
    V = Pr.Volume(lattice_parameters=lp)
    return Pr.Test_U_poly(lp) + Pr.PV_energy(P, V)

# Running the minimization
def run_minimization(X0):
    return scipy.optimize.minimize(U_PV, X0, method='CG', tol=1.e-16)

for i in range(100):
    minimization_output = run_minimization(x0)
    x0 = minimization_output.x
    

# Saving the minimum energy lattice parameters
np.save('test.npy', minimization_output.x)

