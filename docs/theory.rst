######
Theory
######

The Harmonic Approximation
--------------------------
In the simplest lattice dynamic model, the harmonic approximation (HA), we 
assume that the crystal vibrates as a set of harmonic oscillators, providing 
a harmonic approximation to the Gibbs free energy:

.. math::
   G(T) = U(V_0) + A_{v}(V_{0}, T) + PV

Given a lattice and geometry minimized strucutre at volume (:math:`V_{0}`) 
the Gibbs free energy (:math:`G`) is a summation of the crystals potential
energy (:math:`U`), Helmhotz vibrational free energy of a harmonic oscillator
(:math:`A_{v}`), and the energy contributed from the pressure of the system
(:math:`PV`). Generally, the pressure is small enough where the final term 
can be neglected, but our code includes it to be self consistent.

The Helmholtz vibrational free energy can be expressed for a quantum or
classical approach.

* Quantum

.. math::
   A_v(V,T) = \sum_{k} (\frac{\hbar \omega{k}}{2} + \frac{1}{\beta} \ln({1 - \exp({-\beta \hbar \omega_{k}})}))

* Classical 

.. math::
   A_v(V,T) = \sum_{k} \frac{1}{\beta} \ln({\beta \hbar \omega_k(V)})

For both cases, we are summing the energy contribution from all 3:math:`\times N`
vibrational frequencies (:math:`\omega_{k}`) where :math:`N` is the number of atoms.
In this case :math:`\beta = \frac{1}{k_{B} T}`, :math:`\hbar` is reduced Planck's
constant.

To determine the crystals vibrational frequencies (:math:`\omega_{k}`) we compute 
the :math:`3N \times 3N` mass-weighted Hessian. Where the eigenvectors of the Hessian
are the vibrational frequencies.

The Quasi-Harmonic Approximation
--------------------------------
To most accurately model crystals, one should include the thermal expansion of a 
crystal. Crystals expand with temperature, not only increasing the total 
potential energy of the crystal but also increasing the entropy. To improve upon 
the lattice dynamic model, we can use the quasi-harmonic approximation (QHA).
In QHA we determine the Gibbs free energy by solving the argument minimization
of the volume:

.. math::
   \begin{eqnarray}
      G(T,P) &=& \min_{V} f(V,T,P) \\
      f(V,T,P) &=& U(V) + A_{v} (V,T) + PV
   \end{eqnarray}

Where now we must determine the volume of the crystal at a given temperature
and pressure that minimizes Gibbs free energy. 

Thermal Expansion
+++++++++++++++++
There are several ways that we can model thermal expansion of the crystal 
lattice.

Isotropic Expansion
___________________
It is typical to allow the crystal to expand isotropically, for convience and 
computational speed. The crystal lattice is composed of three lattice vectors
(:math:`a,b,c`) and three angles (:math:`\alpha,\beta,\gamma`). In isotropic
expansion we assume that the lattice vectors are always in proportion of each
other (:math:`a:b:c`) and the lattice angles are constant relative to the 
lattice minimum.

For isotropic expansion we only have to minimize the free energy along a single
dimmension, the isotropic volume relative to the lattice minimum strucutre.

Anisotropic Expansion
_____________________
In anisotropic expansion, we allow the lattice vectors and angles to move
independently of one another. In the case of our program, the internal portion
of our code works with the lattice parameters in terms of the upper triangular
tensor form (:math:`\boldsymbol{C_{T}}`):

.. math::
   \boldsymbol{C_{T}} = 
    \begin{bmatrix}
    C_{1} & C_{4} & C_{5} \\
    0     & C_{2} & C_{6} \\
    0     & 0     & C_{3}
    \end{bmatrix}

For easy of use in later sections, we represent the tensor in terms of a length
six array (:math:`\boldsymbol{C}`):

.. math::
   \boldsymbol{C} = 
    \begin{bmatrix}
    C_{1} \\
    C_{2} \\
    C_{3} \\
    C_{4} \\
    C_{5} \\
    C_{6} 
    \end{bmatrix}

Full anisotropic minimization of the crystal lattice for the Gibbs free energy
becomes increasingly expensive since we have to perform the minimization in
six dimensions.

Gruneisen Parameter
+++++++++++++++++++
The most expensive part of performing QHA is computing the mass-weighted Hessian
to get vibrational frequencies. At every new volume we have to re-compute the 
mass-weighted Hessian of the crystal lattice. Alternatively, we can assume the
the deviations in the vibrational frequencies are constant with a specific strain
applied to the lattice.

Isotropic Gruneisen Parameter
_____________________________
For isotropic expansion we can assume that the Gruneisen parameter 
(:math:`\gamma_{k}`) for the :math:`k^{th}` vibrational frequency 
(:math:`\omega_{k}`) has the following proportion to the changes in the lattice
volume:

.. math::
   \gamma_{k} = -\frac{V}{\omega_{k}} \frac{\partial \omega_{k}}{\partial V}

We can solve for the Gruneisen parameter by computing the mass-weighted Hessian
at the lattice minimum volume and a the lattice minimum strucutre isotropically
expanded by a mall amount. By doing this we can compute all of the Gruneisen 
parameters with a forward finite difference appraoch.

Once we know the Gruneisen parameters we can use them and the wavenumbers of the
lattice minimum structure to determine the vibrational frequencies at subsequent 
volumes of interest.

.. math::
   \omega_{k} (V) = \omega_{k}^{ref} \left(\frac{V}{V^{ref}}\right)^{-\gamma_{k}}

Anisotropic Gruneisen Parameter
_______________________________
Similarly, we can compute the Gruneisen parameter for each of the six principal
strains (:math:`\eta_{i}` where :math:`i=1,2,3,4,5,6`):

.. math::
      \gamma_{k,i} = -\frac{1}{\omega_{k}} \left(\frac{\partial \omega_{k}}{\partial \eta_i}\right)_{\eta_{j} \ne \eta_{i}}

For anisotropic expansion we need to compute the mass-weighted Hessian for the
lattice minimum strucutre and the lattice minimum structure strained in the
direction of one of the six principal strains.

Once we know the Gruneisen parameters we can determine the lattice parameters of
the crystal lattice due to any set of lattice parameters with:

.. math::
   \omega_{k}(\eta_{1,2,...6}) = \omega_{k}^{ref} \exp{\left(\sum_{i=1}^{6} -\eta_{i} \gamma_{k,i}\right)}






