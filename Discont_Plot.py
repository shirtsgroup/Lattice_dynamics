#!/usr/bin/env python

import numpy as np
import pylab as plt
import os
import sys

from argparse import ArgumentParser

import Expand as Ex
import ThermodynamicProperties as Pr

parser = ArgumentParser()
molecule = "../molecule.xyz"
with open(molecule) as m:
	molData = m.readlines()

n_atm = int(str.split(molData[0])[0])

files = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.',i)) and 'out' in i])
#print files

coordinates = np.zeros((len(files),n_atm, 3))
volume = np.zeros(len(files))
velocity = np.zeros((len(files)-1, n_atm))

for i in range(len(files)):
   coordinates[i] = Ex.Return_Tinker_Coordinates(files[i]) 
   volume[i] = Pr.Volume(Program='Tinker', Coordinate_file=files[i])
   if i > 0:
       for j in range(n_atm):
           velocity[i - 1,j] = np.linalg.norm(coordinates[i,j] - coordinates[i-1,j])

plt.plot(volume[1:]/volume[10],velocity)
plt.xlabel('$\Delta V / V$', fontsize=18)
plt.ylabel('$\Delta x$ [Ang.]', fontsize=18)
plt.show()
