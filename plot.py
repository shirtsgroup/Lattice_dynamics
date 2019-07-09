#!/usr/bin/env python

import numpy as np
import pylab as plt
from optparse import OptionParser
import ThermodynamicProperties as Pr

def find_dVvT(hvT, dhvT):
    dVvT = np.zeros(len(hvT[:,0]))
    for i in range(len(dVvT)):
        V0 = Pr.Volume(lattice_parameters=hvT[i])
        change = [-1,0,1]
        min_volume = Pr.Volume(lattice_parameters=hvT[i])
        for alpha in change:
            for beta in change:
                for gamma in change:
                    if Pr.Volume(lattice_parameters=hvT[i] + dhvT[i] * [-1, -1, -1, alpha, beta, gamma]) < min_volume:
                        dVvT[i] = V0 - Pr.Volume(lattice_parameters=hvT[i] + dhvT[i] * [-1, -1, -1, alpha, beta, gamma])
    return dVvT


# Plotting lattice parameters
def plot_lattice_parameters_multiple_methods(lattice_parameters, temperature, color, line_style, 
                                             lattice_parameters_error, label, Tmax=False, percent=True, save_loc=False, 
                                             correction=False, unit_vectors_per_super=[],
                                             exp_added=False, exp_labels=[], exp_values=[]):
#    fig, ax = plt.subplots(1,figsize=(12,6.5))
    fig = plt.figure(figsize=(14,7))
    #fig_V = plt.figure(figsize=(6,5.5))
    v1 = fig.add_subplot(241)
    v2 = fig.add_subplot(242, sharex=v1)
    v3 = fig.add_subplot(243, sharex=v1)
    a1 = fig.add_subplot(245, sharex=v1)
    a2 = fig.add_subplot(246, sharex=v1)
    a3 = fig.add_subplot(247, sharex=v1)
    ax_V = fig.add_subplot(144)
    if percent == True:
        v1.axhline(y=0, color='black')
        v2.axhline(y=0, color='black')
        v3.axhline(y=0, color='black')
    for i in range(len(lattice_parameters)):
        lp = np.load(lattice_parameters[i])
#        V = Pr.Volume(lattice_parameters=lp)
        V = np.zeros(len(lp[:,0]))
        for j in range(len(V)):
            V[j] = Pr.Volume(lattice_parameters = lp[j]) / np.prod(unit_vectors_per_super[i])

        if percent == True:
            if lattice_parameters_error[i] == '':
                lp[:, :3] = (lp[:, :3] - lp[0, :3]) / lp[0, :3] * 100.
            else:
                base = np.load(lattice_parameters[0])
                lp[:, :3] = (lp[:, :3] - base[0, :3]) / base[0, :3] * 100.
        else:
            lp[:, :3] = lp[:, :3] / unit_vectors_per_super[i]
            pass

        T = np.load(temperature[i])
        if not np.all(Tmax == False):
            Tp = np.where(T <= Tmax[i])[0][-1] + 1
        else:
            Tp = -1
        if lattice_parameters_error[i] == '':
            ax_V.plot(T[:Tp], V[:Tp], c=color[i], linestyle=line_style[i], label=label[i])
            v1.plot(T[:Tp], lp[:Tp, 0], c=color[i], linestyle=line_style[i], label=label[i])
            v2.plot(T[:Tp], lp[:Tp, 1], c=color[i], linestyle=line_style[i])
            v3.plot(T[:Tp], lp[:Tp, 2], c=color[i], linestyle=line_style[i])
            a1.plot(T[:Tp], lp[:Tp, 3], c=color[i], linestyle=line_style[i])
            a2.plot(T[:Tp], lp[:Tp, 4], c=color[i], linestyle=line_style[i])
            a3.plot(T[:Tp], lp[:Tp, 5], c=color[i], linestyle=line_style[i])
        else:
            error = np.load(lattice_parameters_error[i])
            if percent == False:
                error[:, :3] = error[:, :3] / unit_vectors_per_super[i]
            error_V = find_dVvT(lp, error)
            ax_V.plot(T[:Tp], V[:Tp], c=color[i], linestyle=line_style[i], label=label[i])
            ax_V.fill_between(T[2:Tp], V[2:Tp] - error_V[2:Tp], V[2:Tp] + error_V[2:Tp], color=color[i], linestyle=line_style[i], zorder=0, alpha=0.3)

            v1.plot(T[2:Tp], lp[2:Tp, 0], c=color[i], linestyle=line_style[i], zorder=0, label=label[i])
            v2.plot(T[2:Tp], lp[2:Tp, 1], c=color[i], linestyle=line_style[i], zorder=0)
            v3.plot(T[2:Tp], lp[2:Tp, 2], c=color[i], linestyle=line_style[i], zorder=0)
            a1.plot(T[2:Tp], lp[2:Tp, 3], c=color[i], linestyle=line_style[i], zorder=0)
            a2.plot(T[2:Tp], lp[2:Tp, 4], c=color[i], linestyle=line_style[i], zorder=0)
            a3.plot(T[2:Tp], lp[2:Tp, 5], c=color[i], linestyle=line_style[i], zorder=0)

            v1.fill_between(T[2:Tp], lp[2:Tp, 0] - np.sign(lp[2:Tp, 0]) * error[2:Tp, 0], lp[2:Tp, 0] 
                            + np.sign(lp[2:Tp, 0]) * error[2:Tp, 0],color=color[i], linestyle=line_style[i], 
                            zorder=0,alpha=0.3)
            v2.fill_between(T[2:Tp], lp[2:Tp, 1] - np.sign(lp[2:Tp, 1]) * error[2:Tp, 1], lp[2:Tp, 1] 
                            + np.sign(lp[2:Tp, 1]) * error[2:Tp, 1],color=color[i], linestyle=line_style[i], 
                            zorder=0,alpha=0.3)
            v3.fill_between(T[2:Tp], lp[2:Tp, 2] - np.sign(lp[2:Tp, 2]) * error[2:Tp, 2], lp[2:Tp, 2] 
                            + np.sign(lp[2:Tp, 2]) * error[2:Tp, 2],color=color[i], linestyle=line_style[i], 
                            zorder=0,alpha=0.3)
            a1.fill_between(T[2:Tp], lp[2:Tp, 3] - error[2:Tp, 3], lp[2:Tp, 3] + error[2:Tp, 3], color=color[i], 
                            linestyle=line_style[i], zorder=0, alpha=0.3)
            a2.fill_between(T[2:Tp], lp[2:Tp, 4] - error[2:Tp, 4], lp[2:Tp, 4] + error[2:Tp, 4], color=color[i], 
                            linestyle=line_style[i], zorder=0, alpha=0.3)
            a3.fill_between(T[2:Tp], lp[2:Tp, 5] - error[2:Tp, 5], lp[2:Tp, 5] + error[2:Tp, 5], color=color[i], 
                            linestyle=line_style[i], zorder=0, alpha=0.3)

    # Adding in experimental data
    if exp_added == True:
        colors = ['r','g','b','c','m', 'y', 'k']
        shapes = ["," , "o" , "v" , "^" , "<", ">"]
        for i, l in enumerate(np.unique(exp_labels)):
            loc = np.where(exp_labels == l)[0]
            V = np.zeros(len(loc))
            for j, placement in enumerate(loc):
                V[j] = Pr.Volume(lattice_parameters=np.squeeze(exp_values[placement, 1:]))
            v1.scatter(exp_values[loc, 0], exp_values[loc, 1], marker=shapes[i],  edgecolors=colors[i], label=l, facecolor='none')  
            v2.scatter(exp_values[loc, 0], exp_values[loc, 2], marker=shapes[i],  edgecolors=colors[i], facecolor='none')  
            v3.scatter(exp_values[loc, 0], exp_values[loc, 3], marker=shapes[i],  edgecolors=colors[i], facecolor='none')  
            a1.scatter(exp_values[loc, 0], exp_values[loc, 4], marker=shapes[i],  edgecolors=colors[i], facecolor='none')  
            a2.scatter(exp_values[loc, 0], exp_values[loc, 5], marker=shapes[i],  edgecolors=colors[i], facecolor='none')  
            a3.scatter(exp_values[loc, 0], exp_values[loc, 6], marker=shapes[i],  edgecolors=colors[i], facecolor='none')
            ax_V.scatter(exp_values[loc, 0], V, marker=shapes[i],  edgecolors=colors[i], label=l, facecolor='none')

    v1.title.set_text('a-Vector')
    v2.title.set_text('b-Vector')
    v3.title.set_text('c-Vector')
    a1.title.set_text(r'$\alpha$-Angle')
    a2.title.set_text(r'$\beta$-Angle')
    a3.title.set_text(r'$\gamma$-Angle')
    fig.text(0.4,0.12,'Temperature [K]', fontsize=20, ha='center')
    if percent == True:
        v1.set_ylabel('% Expansion\n from 0 K', fontsize=20, rotation='vertical')
    else:
        v1.set_ylabel('Lattice Vector [$\AA$]', fontsize=20, rotation='vertical')
    a1.set_ylabel('Angle [Deg.]', fontsize=20, rotation='vertical')
    v1.legend(loc='upper center', bbox_to_anchor=(2.4, -1.52), ncol=5, fontsize=14)
    ax_V.set_xlabel('Temperature [K]', fontsize=20)
    ax_V.set_ylabel('Volume [$\AA^{3}$]', fontsize=20)
    #ax_V.legend(loc='upper left', fontsize=14)
 
    a1_lim = a1.get_ylim()
    a2_lim = a2.get_ylim()
    a3_lim = a3.get_ylim()
    if np.absolute(a1_lim[0] - a1_lim[1]) < 2:
        a1.set_ylim((np.ceil(a1_lim[0]) - 1, np.ceil(a1_lim[0]) + 1))
    if np.absolute(a2_lim[0] - a2_lim[1]) < 2:
        a2.set_ylim((np.ceil(a2_lim[0]) - 1, np.ceil(a2_lim[0]) + 1))
    if np.absolute(a3_lim[0] - a3_lim[1]) < 2:
        a3.set_ylim((np.ceil(a3_lim[0]) - 1, np.ceil(a3_lim[0]) + 1))

    plt.xlim(0., max(Tmax))
    plt.tight_layout()
    fig.subplots_adjust(top=0.956, bottom=0.212, left=0.066, right=0.985)
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
            ax1.errorbar(np.load(temperature[0]), hold_lattice_parameters,label=label[i] + ' ' + Labels[0], c=color[i], linestyle=line_style[0], yerr=error) 
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
            ax2.errorbar(np.load(temperature[0]), hold_lattice_parameters, label=label[i] + ' ' + Labels[0], c=color[i-3], linestyle=line_style[0], yerr=error)
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




