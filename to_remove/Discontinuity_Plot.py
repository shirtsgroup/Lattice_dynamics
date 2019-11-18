#!/usr/bin/env python
# Note: To specify a temperature use the -T flag followed by the tempearture to plot at
# Note: To plot a second coordinate file, use the -S and specify the path to the new coordinate folder.
# Now can do 3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

from argparse import ArgumentParser

import Expand as Ex
import ThermodynamicProperties as Pr

parser = ArgumentParser()
parser.add_argument('-T', dest='temp',help = 'Specify the temperature at which to plot the energy vs change in volume')
parser.add_argument('-S', dest='second', help = 'Specify the path of a second molecule directory')
parser.add_argument('-Th', dest='third',help = 'Specify the path of a third molecule directory')
parser.add_argument('-U', dest='potential_e', action='store_true',help = 'Include this flag to plot the internal energy vs change in volume')
parser.add_argument('-Av', dest='vibrational_e', action='store_true',help = 'Include this flag to plot the Helmholtz vibrational energy vs change in volume')
parser.add_argument('-N', dest='n_molecules',help = 'Include this flag with the number of molecules to color code the atomic displacement.')

args = parser.parse_args()
temp = args.temp
if temp is not None:
    plot_temp = float(temp)
else:
    plot_temp = float(250)

second = args.second
if second != None:
    second_dir = second
third = args.third
if third != None:
    third_dir = third
pot_E = args.potential_e
vib_e = args.vibrational_e
num_mols = (args.n_molecules)

print('Plotting at', plot_temp * 1, ' K.')

molecule = "molecule.xyz"
struct_vol = Pr.Volume(Program='Tinker', Coordinate_file=molecule)

with open(molecule) as m:
    molData = m.readlines()

n_atm = int(str.split(molData[0])[0])
if num_mols != None:
    num_mols = int(num_mols)
    atms_per_mol = int(n_atm/(num_mols))

raw = np.load('out_raw.npy')
S1_volumes = raw[:, int(plot_temp / 10), 6]
S1_Gs = raw[:, int(plot_temp) / 10, 2]
S1_Us = raw[:, int(plot_temp) / 10, 3]
S1_Avs = raw[:, int(plot_temp) / 10, 4]

os.chdir("Cords/")
files = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.', i)) and 'out' in i])

coordinates = np.zeros((len(files), n_atm, 3))
volume = np.zeros(len(files))
velocity = np.zeros((len(files) - 1, n_atm))

for i in range(len(files)):
    coordinates[i] = Ex.Return_Tinker_Coordinates(files[i])
    volume[i] = Pr.Volume(Program='Tinker', Coordinate_file=files[i])
    if i > 0:
        for j in range(n_atm):
            velocity[i - 1, j] = np.linalg.norm(coordinates[i, j] - coordinates[i - 1, j])
os.chdir("../")

if second != None:
    os.chdir(second_dir)
    with open(molecule) as m:
        molData2 = m.readlines()
    raw2 = np.load('out_raw.npy')
    S2_volumes = raw2[:, int(plot_temp / 10), 6]
    S2_Gs = raw2[:, int(plot_temp / 10), 2]
    S2_Us = raw2[:, int(plot_temp / 10), 3]
    S2_Avs = raw2[:, int(plot_temp / 10), 4]
    files2 = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.', i)) and 'out' in i])
    coordinates2 = np.zeros((len(files2), n_atm, 3))
    volume2 = np.zeros(len(files2))
if third != None:
    os.chdir(third_dir)
    with open(molecule) as m:
        molData3 = m.readlines()
    raw3 = np.load('out_raw.npy')
    S3_volumes = raw3[:, int(plot_temp / 10), 6]
    S3_Gs = raw3[:, int(plot_temp / 10), 2]
    S3_Us = raw3[:, int(plot_temp / 10), 3]
    S3_Avs = raw3[:, int(plot_temp / 10), 4]
    files3 = np.sort([i for i in os.listdir('.') if os.path.isfile(os.path.join('.', i)) and 'out' in i])
    coordinates3 = np.zeros((len(files3), n_atm, 3))
    volume3 = np.zeros(len(files3))

plt.rc('text', usetex=True)
plt.rc('font', size=16)

if num_mols != None:
    colors = iter(cm.rainbow(np.linspace(0, 1, (num_mols))))
    plot_list = []*int(num_mols)
    for j in range(0,num_mols):
        new_color = next(colors)
        temp_p = plt.plot(volume[1:] / struct_vol, velocity[:,j*atms_per_mol],color = new_color,  label = str(j+1))
        plt.plot(volume[1:] / struct_vol, velocity[:,j*atms_per_mol:(j+1)*atms_per_mol], color = temp_p[0].get_color())
    plt.legend()
else:
    plt.plot(volume[1:] / struct_vol, velocity)
plt.title(r'Atomic Displacement with Thermal Expansion', size=24)
axes = plt.gca()
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel(r'\Delta x [Ang.]', size=20)
plt.show()

p1 = plt.scatter(S1_volumes / struct_vol, S1_Gs)
if second != None:
    p2 = plt.scatter(S2_volumes / struct_vol, S2_Gs, color='red')
    plt.legend([p1, p2], ["Structure 1", "Structure 2"])
if third != None:
    p3 = plt.scatter(S3_volumes / struct_vol, S3_Gs, color='green')
    plt.legend([p1, p2, p3], ["Structure 1", "Structure 2", "Structure 3"])

plot_title = 'Gibbs Free Energy at ' + str(int(plot_temp)) + ' K'
plt.title(plot_title, size=24)
plt.xlabel(r'\Delta V / V', size=20)
plt.ylabel('G (kcal/mol)', size=20)
plt.show()

if pot_E:
    u1 = plt.scatter(S1_volumes / struct_vol, S1_Us)
    if second != None:
        u2 = plt.scatter(S2_volumes / struct_vol, S2_Us, color='red')
        plt.legend([u1, u2], ["Structure 1", "Structure 2"])
    if third != None:
        u3 = plt.scatter(S3_volumes / struct_vol, S3_Us, color='green')
        plt.legend([u1, u2, u3], ["Structure 1", "Structure 2", "Structure 3"])

    plot_title = 'Internal Energy at ' + str(int(plot_temp)) + ' K'
    plt.title(plot_title, size=24)
    plt.xlabel(r'\Delta V / V', size=20)
    plt.ylabel('U (kcal/mol)', size=20)
    plt.show()

if vib_e:
    Av1 = plt.scatter(S1_volumes / struct_vol, S1_Avs)
    if second != None:
        Av2 = plt.scatter(S2_volumes / struct_vol, S2_Avs, color='red')
        plt.legend([Av1, Av2], ["Structure 1", "Structure 2"])
    if third != None:
        Av3 = plt.scatter(S3_volumes / struct_vol, S3_Avs, color='green')
        plt.legend([Av1, Av2, Av3], ["Structure 1", "Structure 2", "Structure 3"])

    plot_title = 'Helmholtz Free Energy at ' + str(int(plot_temp)) + ' K'
    plt.title(plot_title, size=24)
    plt.xlabel(r'\Delta V / V', size=20)
    plt.ylabel('Av (kcal/mol)', size=20)
    plt.show()
