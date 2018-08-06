#!/usr/bin/env python
from __future__ import print_function
import numpy as np


#### Equations of State ####
def EV_EOS(V, V0, B, dB, E0, eq_of_state):
    if eq_of_state == 'Murnaghan':
        return EV_Murnaghan_EOS(V, V0, B, dB, E0)
    elif eq_of_state == 'Birch-Murnaghan':
        return EV_BirchMurnaghan_EOS(V, V0, B, dB, E0)
    elif eq_of_state == 'Rose-Vinet':
        return EV_RoseVinet_EOS(V, V0, B, dB, E0)

def PV_EOS(V, V0, B, dB, eq_of_state):
    if eq_of_state == 'Murnaghan':
        return PV_Murnaghan_EOS(V, V0, B, dB)
    elif eq_of_state == 'Birch-Murnaghan':
        return PV_BirchMurnaghan_EOS(V, V0, B, dB)
    elif eq_of_state == 'Rose-Vinet':
        return PV_RoseVinet_EOS(V, V0, B, dB)

def PV_Murnaghan_EOS(V, V0, B, dB):
    return (B / dB) * ((V / V0) ** (-dB) - 1)

def EV_Murnaghan_EOS(V, V0, B, dB, E0):
    return E0 + B * V0 * ((1. / (dB * (dB - 1.))) * (V / V0) ** (1. - dB) + (V / (dB * V0)) - (1. / (dB - 1.)))

def PV_BirchMurnaghan_EOS(V, V0, B, dB):
    return (3 * B / 2) * ((V0 / V) ** (7 / 3) - (V0 / V) ** (5 / 3)) * (1 + (3 / 4) * (dB - 4) * ((V0 / V) ** (2 / 3) -
                                                                                                  1))

def EV_BirchMurnaghan_EOS(V, V0, B, dB, E0):
    return E0 + (9. * V0 * B / 16.) * (dB * ((V0 / V) ** (2. / 3.) - 1.) ** 3. + ((V0 / V) ** (2. / 3.) - 1.) ** 2. *
                                       (6. - 4. * (V0 / V) ** (2. / 3.)))

def PV_RoseVinet_EOS(V, V0, B, dB):
    return 3 * B * (1 - (V / V0) ** (1/ 3)) * ((V / V0) ** (1 / 3)) ** -2 * np.exp((3 / 2) * (dB - 1) *
                                                                                   (1 - (V / V0) ** (1 / 3)))

def EV_RoseVinet_EOS(V, V0, B, dB, E0):
    return E0 + (4 * B * V0 / (dB - 1) ** 2) - 2 * V0 * B * (dB - 1) ** -2 * (5 + 3 * dB * ((V / V0) ** (1 / 3) - 1) - 3
                                                                              * (V / V0) ** (1 / 3)) * \
                                               np.exp((-3 / 2) * (dB - 1) * ((V / V0) ** (1 / 3) - 1))
