#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import subprocess
import numpy as np
import pylab as plt
sys.path.insert(0,'/home/nabraham/bin/Lattice_dynamics')
from Run_LatticeDynamics import Temperature_Lattice_Dynamics

steps = np.array([50.,100.,125.,200.,250.,500.,1000.])
T_max = 1000.
G_max = np.zeros((len(steps), 2))

# Inputs for code
Method = 'GaQ'
Aniso_LocGrad_type=73

for i in range(len(steps)):
    Temperature_Lattice_Dynamics(Method=Method, NumAnalysis_method='RK4', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type, Gradient_MaxTemp=T_max,LocGrd_Diag_FracStep = 1.0e-05,LocGrd_OffDiag_FracStep = 1.0e-05)
    G_max[i,0] = np.load('out_GClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm -rf Cords/ numerical_checks.out out_*'], shell=True)

    Temperature_Lattice_Dynamics(Method=Method, NumAnalysis_method='Euler', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type, Gradient_MaxTemp=T_max, LocGrd_Diag_FracStep = 1.0e-05,LocGrd_OffDiag_FracStep = 1.0e-05)
    G_max[i,1] = np.load('out_GClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm -rf Cords/ numerical_checks.out out_*'], shell=True)

x = T_max/steps
print(G_max)
y_G = np.absolute((G_max - G_max[0])/(G_max[-1] - G_max[0]))

plt.scatter(x, y_G[:, 0], c='r', label='RK4')
plt.plot(x, (1/x)**4, c='r', label='$(1/n)^{4}$')
plt.scatter(x, y_G[:, 1], c='b', label='Euler')
plt.plot(x, (1/x)**1, c='b', label='$1/n$')
plt.legend()
plt.show()



