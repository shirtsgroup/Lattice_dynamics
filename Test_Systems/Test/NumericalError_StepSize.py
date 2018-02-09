#!/usr/bin/env python
import os
import sys
import subprocess
import numpy as np
import pylab as plt
sys.path.insert(0,'/Users/mrshirts/work/papers/CRYSTALMAPPING/Lattice_dynamics')
from Run_LatticeDynamics import Temperature_Lattice_Dynamics

steps = np.array([5.,30.,50.,60.,75.,100.,150.,300.])
G_300 = np.zeros((len(steps), 2))
#V_300 = np.zeros((len(steps), 2))

# Inputs for code
Method = 'GaQ'
Aniso_LocGrad_type=73

for i in range(len(steps)):
    Temperature_Lattice_Dynamics(Method=Method, NumAnalysis_method='RK4', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type)
    G_300[i,0] = np.load('out_GClassical_' + Method + '.npy')[-1]
#    V_300[i,0] = np.load('out_VClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm', '-rf', 'Cords/', 'numerical_checks.out', 'out_WVN_' + Method + '.npy'])

    Temperature_Lattice_Dynamics(Method=Method, NumAnalysis_method='Euler', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type)
    G_300[i,1] = np.load('out_GClassical_' + Method + '.npy')[-1]
#    V_300[i,1] = np.load('out_VClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm', '-rf', 'Cords/', 'numerical_checks.out', 'out_WVN_' + Method + '.npy'])

x = 300./steps
print(G_300)
y_G = np.absolute((G_300 - G_300[0])/(G_300[-1] - G_300[0]))
#y_V = (V_300 - V_300[0])/(V_300[-1] - V_300[0])

#plt.scatter(x, y_V[:, 0], c='r', label='RK4')
#plt.plot(x, (1/x)**4, c='r', label='$(1/n)^{4}$')
#plt.scatter(x, y_V[:, 1], c='b', label='Euler')
#plt.plot(x, (1/x)**1, c='b', label='$1/n$')
#plt.legend()
#plt.xlabel('n', fontsize=24)
#plt.ylabel('$k_{frac}$', fontsize=24)
#plt.tight_layout()
#plt.show()

plt.scatter(x, y_G[:, 0], c='r', label='RK4')
plt.plot(x, (1/x)**4, c='r', label='$(1/n)^{4}$')
plt.scatter(x, y_G[:, 1], c='b', label='Euler')
plt.plot(x, (1/x)**1, c='b', label='$1/n$')
plt.legend()
plt.show()



