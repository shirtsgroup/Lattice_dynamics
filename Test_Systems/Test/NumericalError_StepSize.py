#!/usr/bin/env python
import os
import sys
import subprocess
import numpy as np
import pylab as plt
import pdb
sys.path.insert(0,'/Users/mrshirts/work/papers/CRYSTALMAPPING/Lattice_dynamics')
from Run_LatticeDynamics import Temperature_Lattice_Dynamics

steps = np.array([100.,200.,250.,400.,500.,1000.,2000.])
G_max = np.zeros((len(steps), 2))
V_max = np.zeros((len(steps), 2))

# Inputs for code
Method = 'GaQ'
Aniso_LocGrad_type=73
maxT = steps[-1]
for i in range(len(steps-1)): # 2000 degrees in one step is a bit much.
    Temperature_Lattice_Dynamics(Method=Method, Temperature=[0,maxT],NumAnalysis_method='RK4', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type,Gradient_MaxTemp=maxT)
    G_max[i,0] = np.load('out_GClassical_' + Method + '.npy')[-1]
    V_max[i,0] = np.load('out_VClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm', '-rf', 'Cords/', 'numerical_checks.out', 'out_WVN_' + Method + '.npy'])

    Temperature_Lattice_Dynamics(Method=Method, Temperature=[0,maxT],NumAnalysis_method='Euler', NumAnalysis_step=steps[i],Aniso_LocGrad_Type=Aniso_LocGrad_type,Gradient_MaxTemp=maxT)
    G_max[i,1] = np.load('out_GClassical_' + Method + '.npy')[-1]
    V_max[i,1] = np.load('out_VClassical_' + Method + '.npy')[-1]
    subprocess.call(['rm', '-rf', 'Cords/', 'numerical_checks.out', 'out_WVN_' + Method + '.npy'])

x = maxT/steps
print(G_max)
print(V_max)
y_G = np.absolute((G_max - G_max[0])/(G_max[-1] - G_max[0]))
y_V = (V_max - V_max[0])/(V_max[-1] - V_max[0])

plt.figure(1)
plt.scatter(x, y_V[:, 0], c='r', label='RK4')
plt.plot(x, (1/x)**4, c='r', label='$(1/n)^{4}$')
plt.scatter(x, y_V[:, 1], c='b', label='Euler')
plt.plot(x, (1/x)**1, c='b', label='$1/n$')
plt.legend()
plt.xlabel('n', fontsize=24)
plt.ylabel('$k_{frac}$', fontsize=24)
plt.tight_layout()
plt.show()
plt.savefig('Vplot.pdf')

plt.figure(2)
plt.scatter(x, y_G[:, 0], c='r', label='RK4')
plt.plot(x, (1/x)**4, c='r', label='$(1/n)^{4}$')
plt.scatter(x, y_G[:, 1], c='b', label='Euler')
plt.plot(x, (1/x)**1, c='b', label='$1/n$')
plt.legend()
plt.show()
plt.savefig('Gplot.pdf')



