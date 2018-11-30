#!/usr/bin/env python

import numpy as np
import pylab as plt
from optparse import OptionParser

# Plotting lattice parameters
def plot_lattice_parameters_multiple_methods(lattice_parameters, temperature, color, line_style, lattice_parameters_error, Tmax=False, basis=False, save_loc=False):
    fig, ax = plt.subplots(figsize=(12,6))
    v1 = plt.subplot(231)
    v2 = plt.subplot(232)
    v3 = plt.subplot(233)
    a1 = plt.subplot(234)
    a2 = plt.subplot(235)
    a3 = plt.subplot(236)
    v1.axhline(y=0, color='black')
    v2.axhline(y=0, color='black')
    v3.axhline(y=0, color='black')
    a1.axhline(y=0, color='black')
    a2.axhline(y=0, color='black')
    a3.axhline(y=0, color='black')
    for i in range(len(lattice_parameters)):
        hold_lattice_parameters = np.load(lattice_parameters[i])
        if np.all(basis == False):
            base = hold_lattice_parameters[0]
        else:
            base = np.load(lattice_parameters[basis[i]])[0]
        hold_lattice_parameters = (hold_lattice_parameters - hold_lattice_parameters[0]) / base * 100.
        T = np.load(temperature[i])
        if not np.all(Tmax == False):
            T_place = np.where(T == Tmax[i])[0][0] + 1
        else:
            T_place = -1
        if lattice_parameters_error[i] == '':
            v1.plot(T[:T_place], hold_lattice_parameters[:T_place, 0], c=color[i], linestyle=line_style[i])
            v2.plot(T[:T_place], hold_lattice_parameters[:T_place, 1], c=color[i], linestyle=line_style[i])
            v3.plot(T[:T_place], hold_lattice_parameters[:T_place, 2], c=color[i], linestyle=line_style[i])
            a1.plot(T[:T_place], hold_lattice_parameters[:T_place, 3], c=color[i], linestyle=line_style[i])
            a2.plot(T[:T_place], hold_lattice_parameters[:T_place, 4], c=color[i], linestyle=line_style[i])
            a3.plot(T[:T_place], hold_lattice_parameters[:T_place, 5], c=color[i], linestyle=line_style[i])
        else:
            error = np.load(lattice_parameters_error[i])
            v1.plot(T[:T_place], hold_lattice_parameters[:T_place, 0], c=color[i], linestyle=line_style[i], zorder=0)
            v2.plot(T[:T_place], hold_lattice_parameters[:T_place, 1], c=color[i], linestyle=line_style[i], zorder=0)
            v3.plot(T[:T_place], hold_lattice_parameters[:T_place, 2], c=color[i], linestyle=line_style[i], zorder=0)
            a1.plot(T[:T_place], hold_lattice_parameters[:T_place, 3], c=color[i], linestyle=line_style[i], zorder=0)
            a2.plot(T[:T_place], hold_lattice_parameters[:T_place, 4], c=color[i], linestyle=line_style[i], zorder=0)
            a3.plot(T[:T_place], hold_lattice_parameters[:T_place, 5], c=color[i], linestyle=line_style[i], zorder=0)
            v1.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 0] - np.sign(hold_lattice_parameters[:T_place, 0]) * error[:T_place, 0], hold_lattice_parameters[:T_place, 0] + np.sign(hold_lattice_parameters[:T_place, 0]) * error[:T_place, 0], color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
            v2.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 1] - np.sign(hold_lattice_parameters[:T_place, 1]) * error[:T_place, 1],hold_lattice_parameters[:T_place, 1] + np.sign(hold_lattice_parameters[:T_place, 1]) * error[:T_place, 1],color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
            v3.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 2] - np.sign(hold_lattice_parameters[:T_place, 2]) * error[:T_place, 2],hold_lattice_parameters[:T_place, 2] + np.sign(hold_lattice_parameters[:T_place, 2]) * error[:T_place, 2],color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
            a1.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 3] - np.sign(hold_lattice_parameters[:T_place, 3]) * error[:T_place, 3],hold_lattice_parameters[:T_place, 3] + np.sign(hold_lattice_parameters[:T_place, 3]) * error[:T_place, 3],color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
            a2.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 4] - np.sign(hold_lattice_parameters[:T_place, 4]) * error[:T_place, 4],hold_lattice_parameters[:T_place, 4] + np.sign(hold_lattice_parameters[:T_place, 4]) * error[:T_place, 4],color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
            a3.fill_between(T[:T_place], hold_lattice_parameters[:T_place, 5] - np.sign(hold_lattice_parameters[:T_place, 5]) * error[:T_place, 5],hold_lattice_parameters[:T_place, 5] + np.sign(hold_lattice_parameters[:T_place, 5]) * error[:T_place, 5],color=color[i], linestyle=line_style[i], zorder=0,alpha=0.3)
    x = a1.get_ylim()
    if x[0] > -1.:
        lower = -1.
    else:
        lower = x[0]
    if x[1] < 1.:
        upper = 1.
    else:
        upper = x[1]
    a1.set_ylim((lower, upper))

    x = a2.get_ylim()
    if x[0] > -1.:
        lower = -1.
    else:
        lower = x[0]
    if x[1] < 1.:
        upper = 1.
    else:
        upper = x[1]
    a2.set_ylim((lower, upper))

    x = a3.get_ylim()
    if x[0] > -1.:
        lower = -1.
    else:
        lower = x[0]
    if x[1] < 1.:
        upper = 1.
    else:
        upper = x[1]
    a3.set_ylim((lower, upper))

    v1.title.set_text('a-Vector')
    v2.title.set_text('b-Vector')
    v3.title.set_text('c-Vector')
    a1.title.set_text(r'$\alpha$-Angle')
    a2.title.set_text(r'$\beta$-Angle')
    a3.title.set_text(r'$\gamma$-Angle')
    fig.text(0.53,0.01,'Temperature [K]', fontsize=20, ha='center')
    fig.text(0.02, 0.8, '% Expansion from 0 K', fontsize=20, rotation='vertical')
    plt.tight_layout()
    plt.subplots_adjust(top=0.941, bottom=0.102, left=0.091, right=0.985)
    if not save_loc == False:
        plt.savefig(save_loc)
        plt.close()
    else:
        plt.show()
             

def plot_lattice_parameters_one_method():
    label = ['a','b','c',r'$\alpha$',r'$\beta$',r'$\gamma$']
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for i in np.arange(0,3):
        if lattice_parameters_error == False:
            hold_lattice_parameters = np.load(lattice_parameters[0])[:,i]
            ax1.plot(np.load(temperature[0]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100,label=label[i] + ' ' + Labels[0], c=color[i], linestyle=line_style[0])
        else:
            hold_lattice_parameters = np.load(lattice_parameters[0])[:,i]
            error = np.load(lattice_parameters_error[0])[:,i]
            ax1.errorbar(np.load(temperature[0]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100,label=label[i] + ' ' + Labels[0], c=color[i], linestyle=line_style[0], yerr=error) 
    ax1.set_xlabel('Temperature [K]', fontsize=18)
    ax1.set_ylabel('% Change from ' + str(int(np.load(temperature[0])[0])) + 'K', fontsize=18)
    ax1.legend(loc='upper left', fontsize=18)

    for i in np.arange(3,6):
        if lattice_parameters_error == False:
            hold_lattice_parameters = np.load(lattice_parameters[0])[:,i]
            ax2.plot(np.load(temperature[0]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100, label=label[i] + ' ' + Labels[0], c=color[i-3], linestyle=line_style[0])
        else:
            hold_lattice_parameters = np.load(lattice_parameters[0])[:,i]
            error = np.load(lattice_parameters_error[0])[:,i]
            ax2.errorbar(np.load(temperature[0]), (hold_lattice_parameters - hold_lattice_parameters[0])/hold_lattice_parameters[0]*100, label=label[i] + ' ' + Labels[0], c=color[i-3], linestyle=line_style[0], yerr=error)
    ax2.set_xlabel('Temperature [K]', fontsize=18)
    ax2.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-L', dest = 'lattice_parameters', help = '<output>_h<StatMech>_<Method>.npy files containing lattice parameters', default = '')
    parser.add_option('-l', dest = 'lattice_parameters_error', help = '<output>_h<StatMech>_<Method>.npy files containing lattice parameters', default = False)
    parser.add_option('-V', dest = 'volume', help = '<output>_V<StatMech>_<Method>.npy files containing volume', default = '')
    parser.add_option('-G', dest = 'Gibbs', help = '<output>_G<StatMech>_<Method>.npy files containing gibbs free energy', default = '')
    parser.add_option('-T', dest = 'temperature', help = '<output>_T_<Method>.npy file containing temperature array', default = '')
    parser.add_option('-t', dest = 'Labels', help = 'list of label names for input files', default = '')
    parser.add_option('-U', dest = 'potential_energy', help = '<output>_U<StatMech>_<Method>.npy file containing potential energy array', default = '')
    
    (options, args) = parser.parse_args()
    lattice_parameters = (options.lattice_parameters).split(',')
    
    if options.lattice_parameters_error == False:
        lattice_parameters_error = options.lattice_parameters_error
    else:
        lattice_parameters_error = (options.lattice_parameters_error).split(',')
    
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
    line_style = ['--',':','-.']

    if (lattice_parameters[0] != '') and (len(lattice_parameters) == len(temperature)):
        if len(lattice_parameters) == 1:
            plot_lattice_parameters_one_method()
        else:
            plot_lattice_parameters_multiple_methods(lattice_parameters, temperature,color, line_style)
    
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




