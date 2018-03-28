#!/usr/bin/env python

import numpy as np
import pylab as plt

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-L', dest = 'lattice_parameters', help = '<output>_h<StatMech>_<Method>.npy files containing lattice parameters', default = '')
parser.add_option('-V', dest = 'volume', help = '<output>_V<StatMech>_<Method>.npy files containing volume', default = '')
parser.add_option('-G', dest = 'Gibbs', help = '<output>_G<StatMech>_<Method>.npy files containing gibbs free energy', default = '')
parser.add_option('-T', dest = 'temperature', help = '<output>_T_<Method>.npy file containing temperature array', default = '')
parser.add_option('-l', dest = 'Labels', help = 'list of label names for input files', default = '')
parser.add_option('-U', dest = 'potential_energy', help = '<output>_U<StatMech>_<Method>.npy file containing potential energy array', default = '')

(options, args) = parser.parse_args()
lattice_parameters = (options.lattice_parameters).split(',')
volume = (options.volume).split(',')
Gibbs = (options.Gibbs).split(',')
temperature = (options.temperature).split(',')
Labels = (options.Labels).split(',')
potential_energy = (options.potential_energy).split(',')


if (Labels == '') and (len(temperature) >= 2):
    Labels = []
    for i in range(len(temperature)):
        Labels.append('input ' + str(i+1))

color = ['r', 'b', 'g']
line_style = ['--',':']

# Plotting lattice parameters
if (lattice_parameters[0] != '') and (len(lattice_parameters) == len(temperature)):
    label = ['a','b','c',r'$\alpha$',r'$\beta$',r'$\gamma$']
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for i in np.arange(0,3):
        for j in range(len(lattice_parameters)):
            hold_lattice_parameters = np.load(lattice_parameters[j])[:,i]
            ax1.plot(np.load(temperature[j]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100,label=label[i] + ' ' + Labels[j], c=color[i], linestyle=line_style[j])
    ax1.set_xlabel('Temperature [K]', fontsize=18)
    ax1.set_ylabel('% Change from ' + str(int(np.load(temperature[0])[0])) + 'K', fontsize=18)
    ax1.legend(loc='upper left', fontsize=18)

    for i in np.arange(3,6):
        for j in range(len(lattice_parameters)):
            hold_lattice_parameters = np.load(lattice_parameters[j])[:,i]
            ax2.plot(np.load(temperature[j]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100, label=label[i] + ' ' + Labels[j], c=color[i-3], linestyle=line_style[j])
    ax2.set_xlabel('Temperature [K]', fontsize=18)
    ax2.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.show()

# Plotting Gibbs free energy difference
if (Gibbs[0] != '') and (len(Gibbs) == len(temperature)):
    Gibbs_reference = np.load(Gibbs[0])
    temperature_reference = np.load(temperature[0])
    for i in np.arange(1, len(Gibbs)):
        if len(np.load(temperature[i])) == len(temperature_reference):
            plt.plot(np.load(temperature[i]), np.load(Gibbs[i]) - Gibbs_reference)#, label = Labels[i])
        else:
            temperature_hold = []
            dGibbs = []
            for j in range(len(temperature_reference)):
                for k in range(len(np.load(temperature[i]))):
                    if temperature_reference[j] == np.load(temperature[i])[k]:
                        temperature_hold.append(temperature_reference[j])
                        dGibbs.append(np.load(Gibbs[i])[k] - Gibbs_reference[j])
            plt.plot(temperature_hold, dGibbs, label= Labels[i])
    plt.xlabel('Temperature [K]', fontsize=18)
    plt.ylabel('$\Delta$Gibbs from ' + Labels[0], fontsize = 18)
    plt.axhline(0,c='black')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plotting volume
if (volume[0] != '') and (len(volume) == len(temperature)):
    for i in range(len(volume)):
        plt.plot(np.load(temperature[i]), np.load(volume[i]), c=color[i], linestyle=line_style[i], label=Labels[i])
    plt.ylabel('Volume [Ang.^3]', fontsize=18)
    plt.xlabel('Temperature [K]', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.show()

# Plotting potential energy
if (potential_energy[0] != '') and (len(potential_energy) == len(temperature)):
    for i in range(len(potential_energy)):
        plt.plot(np.load(temperature[i]), np.load(potential_energy[i]), c=color[i], linestyle=line_style[i], label=Labels[i])
    plt.ylabel('Potential Energy [kcal/mol]', fontsize=18)
    plt.xlabel('Temperature [K]', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.show()




