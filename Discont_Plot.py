#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from argparse import ArgumentParser


import Expand as Ex
import ThermodynamicProperties as Pr

parser = ArgumentParser()
parser.add_argument('-T', dest = 'temp')
args = parser.parse_args()
temp = args.temp
if temp!= None:
	plot_temp = float(temp)
else:
	plot_temp = float(250)


print 'Plotting at',plot_temp*1,' K.'

molecule = "molecule.xyz"
struct_vol = Pr.Volume(Program='Tinker', Coordinate_file=molecule)

with open(molecule) as m:
	molData = m.readlines()

n_atm = int(str.split(molData[0])[0])

raw = np.load('out_raw.npy')
my_volumes = raw[:,int(plot_temp/10),6]
my_Gs = raw[:, int(plot_temp)/10,2]

os.chdir("Cords/")
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

 #float(str.split(molData[1])[0])*float(str.split(molData[1])[1])*float(str.split(molData[1])[2])

#print(struct_vol)

plt.rc('text', usetex=True)
plt.rc('font', size=16)

plt.plot(volume[1:]/struct_vol,velocity)
plt.title(r'Atomic Displacement with Thermal Expansion', size = 24)
axes = plt.gca()
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel(r'\Delta x [Ang.]', size=20)
plt.show()

#plt.figure(figsize=(8,3))
plt.scatter(my_volumes/struct_vol,my_Gs)
plot_title = 'Gibbs Free Energy at '+str(int(plot_temp))+' K'
plt.title(plot_title, size=24)
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel('G (kcal/mol)', size = 20)
plt.show()
#plt.figure(figsize=(3,8))

