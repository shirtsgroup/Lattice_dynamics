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

# Running the minimization
minimization_output = scipy.optimize.minimize(Pr.Test_U_poly, x0, method='Nelder-Mead', tol=1e-6)

# Saving the minimum energy lattice parameters
np.save('test.npy', minimization_output.x)

