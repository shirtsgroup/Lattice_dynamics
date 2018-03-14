#!/usr/bin/env python
#Note: To specify a temperature use the -T flag followed by the tempearture to plot at
#Note: To plot a second coordinate file, use the -S and specify the path to the new coordinate folder.


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from argparse import ArgumentParser


import Expand as Ex
import ThermodynamicProperties as Pr

parser = ArgumentParser()
parser.add_argument('-T', dest = 'temp')
parser.add_argument('-S', dest = 'second')
parser.add_argument('-Th', dest = 'third')
args = parser.parse_args()
temp = args.temp
if temp!= None:
	plot_temp = float(temp)
else:
	plot_temp = float(250)



second = args.second
if second!= None:
	second_dir = second;
third = args.third
if third !=None:
	third_dir = third;


print('Plotting at',plot_temp*1,' K.')

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

coordinates = np.zeros((len(files),n_atm, 3))
volume = np.zeros(len(files))
velocity = np.zeros((len(files)-1, n_atm))


for i in range(len(files)):
   coordinates[i] = Ex.Return_Tinker_Coordinates(files[i])
   volume[i] = Pr.Volume(Program='Tinker', Coordinate_file=files[i])
   if i > 0:
       for j in range(n_atm):
           velocity[i - 1,j] = np.linalg.norm(coordinates[i,j] - coordinates[i-1,j])
os.chdir("../")

if second != None:
	os.chdir(second_dir)
	with open(molecule) as m:
		molData2 = m.readlines()
	raw2 = np.load('out_raw.npy')
	my_volumes2 = raw2[:,int(plot_temp/10),6]
	my_Gs2 = raw2[:, int(plot_temp)/10,2]
	files2 = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.',i)) and 'out' in i])
	coordinates2 = np.zeros((len(files2),n_atm, 3))
	volume2 = np.zeros(len(files2))
if third != None:
        os.chdir(third_dir)
        with open(molecule) as m:
                molData3 = m.readlines()
        raw3 = np.load('out_raw.npy')
        my_volumes3 = raw3[:,int(plot_temp/10),6]
        my_Gs3 = raw3[:, int(plot_temp)/10,2]
        files3 = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.',i)) and 'out' in i])
        coordinates3 = np.zeros((len(files3),n_atm, 3))
        volume3 = np.zeros(len(files3))

plt.rc('text', usetex=True)
plt.rc('font', size=16)

plt.plot(volume[1:]/struct_vol,velocity)
plt.title(r'Atomic Displacement with Thermal Expansion', size = 24)
axes = plt.gca()
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel(r'\Delta x [Ang.]', size=20)
plt.show()

#plt.figure(figsize=(8,3))
p1 = plt.scatter(my_volumes/struct_vol,my_Gs)
if second != None:
	p2 = plt.scatter(my_volumes2/struct_vol,my_Gs2,color='red')
	plt.legend([p1, p2],["Structure 1", "Structure 2"])
if third != None:
        p3 = plt.scatter(my_volumes3/struct_vol,my_Gs3,color='green')
        plt.legend([p1, p2,p3],["Structure 1", "Structure 2", "Structure 3"])

plot_title = 'Gibbs Free Energy at '+str(int(plot_temp))+' K'
plt.title(plot_title, size=24)
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel('G (kcal/mol)', size = 20)
plt.show()
#plt.figure(figsize=(3,8))
