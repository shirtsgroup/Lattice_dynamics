##############
Job Submission
##############
To submit a lattice dynamic simulation we run:

.. code-block:: bash

   Run_LatticeDynamics.py -i input.yaml


If you want to re-run the simulation and delete all previous results we run:

.. code-block:: bash

   Run_LatticeDynamics.py -i input.yaml -D

An input.yaml file must be present in the current working directory, which
specifies all inputs for simulation.

Input File
----------
A YAML formatted input file must be provided with the specific inputs for 
the simulation. The following input parameters and the options are as followed:

temperature:
   List of temperatures that the final output properties are calculated for.
pressure:
   Single pressure to run simulation at. The current code is set up to start from a lattice minimum strucutre at a given pressure and should only be used for pressures of 1 atm.
method:
   *HA* -- the harmonic approximation will be run, neglecting the affects of thermal expansion.

   *SiQ* -- peroforms isotropic expansion for QHA by using a stepwise expansion of the crystal lattice. The phonons will be evaluated for the crystal structure at every new lattice geometry.

   *SiQg* -- peroforms isotropic expansion for QHA by using a stepwise expansion of the crystal lattice. Additionally, this uses the Gruneisen parameter to compute the wavenumbers at each volume.

   *GiQ* -- performs isotropic expansion for QHA by computing the rate of thermal expansion. The phonons will be evaluated for the crystal structure at every new lattice geometry.

   *GiQg* -- performs isotropic expansion for QHA by computing the rate of thermal expansion. Additionally, this uses the Gruneisen parameter to compute the wavenumbers at each volume.

   *GaQ* -- performs anisotropic expansion for QHA by computing the rate of thermal expansion. The phonons will be evaluated for the crystal structure at every new lattice geometry.

   *GaQg* -- performs anisotropic expansion for QHA by computing the rate of thermal expansion. Additionally, this uses the Gruneisen parameter to compute the wavenumbers at each volume.
output:
   All output file will start with the string specified by this.
statistical_mechanics:
   *Classical* -- computes the helmholtz vibrational free energy and entropy using the classical equation for harmonic osccillator.

   *Quantum* -- computes the helmholtz vibrational free energy and entropy using the quantum equation for harmonic osccillator.
coordinate_file:
   Lattice minimum structure to start the simulation from. This coordinate file should be present in the working directory.
number_of_molecules:
   Number of molecules in the crystal lattice.
program:
   Simulation package to call from when running geometry optimizations, phonon calculations, or energy analysis. Options are: *Tinker*, *Test*, *CP2K*, and *QE*.
tinker:
   parameter_file:
      Keyfile specifying the inputs specific for Tinker Molecular modeling.
cp2k:
   root:
      ???
properties_to_save:
   List of properties to output in the simulation. Options are:

   *T* -- temperature stored in <output>_T_<method>.npy

   *P* -- pressure stored in <output>_P_<method>.npy

   *G* -- Gibbs free energy stored in <output>_G<statistical_mechanics>_<method>.npy

   *U* -- lattice energy stored in <output>_U<statistical_mechanics>_<method>.npy

   *Av* -- Helmhotz vibrational free energy of a harmonic oscillator stored in <output>_Av<statistical_mechanics>_<method>.npy

   *V* -- volume stored in <output>_V<statistical_mechanics>_<method>.npy

   *h* -- lattice vectors and angles stored in <output>_h<statistical_mechanics>_<method>.npy

   *S* -- vibrational entropy of a harmonic oscillator stored in <output>_S<statistical_mechanics>_<method>.npy

   All properties will be saved in <output>_raw.npy for the simulation, but are parsed out here for ease of use and analysis.
gradient:
   Inputs specific for the gradient methods (*GiQ*, *GiQg*, *GaQ*, and *GaQg*).

   numerical_method:
      *Euler* - Euler method for numerical integration

      *RK4* - 4th order Runge-Kutta method for numerical integrations
   numerical_step:
      Numerical step size for the **numerical_method** in Kelvin.
   max_temperature:
      Maximum temperature to run the simulation up to. Should be a factor of **numerical_step**.
   vol_fraction:
      Finite difference stepsize to solve :math:`\frac{\partial V}{\partial T}`. If this is not specified and **method** is set to *GiQ* or *GiQg* the code will attempt to guess an appropriate value.
   matrix_fractions:
      Six finite difference stepsizes to solve :math:`\frac{\partial \boldsymbol{C}}{\partial T}`. If this is not specified and **method** is set to *GaQ* or *GaQg* the code will attempt to guess an appropriate value.
   anisotropic_type:
      *6D* -- performs full optimization of the entire cell geometry across all temperatures for anisotropic expansion.

      *3D* -- performs optimization of the diagonal elements of the crystal lattice tensor (:math:`C_{i}` where :math:`i=1,2,3`) for anisotropic expansion. The off diagonal elements remain fixed.

      *1D* -- determines the full anisotorpic thermal expansion at 0 K and assumes the rate of each parameter remain in proportion across all temperatures of interest.
stepwise:
   Inputs specific for the stepwise methods (*SiQ* and *SiQg*).

   volume_fraction_lower:
      Lower bound on the isotropic expansion of the crystal lattice. A value of 0.99 would compress the strucutre to 99% of the lattice minimum volume.

   volume_fraction_upper:
      Upper bound on the isotropic expansion of the crystal lattice. A value of 1.01 would expand the strucutre to 101% of the lattice minimum volume.

   volume_fraction_stepsize:
      Sets the intermediate tolerance of expanded structures between *volume_fraction_lower* and *volume_fraction_upper*. If set to 0.01 the first compressed and expanded structures would be 99% and 101% of the lattice minimum volume, respectively.
gruneisen:
   volume_fraction_stepsize:
      Finite difference stepsize to solve for the isotropic Gruneisen parameters.      
   matrix_strain_stepsize:
      Finite difference stepsize to solve for the anisotropic Gruneisen parameters.
wavenumber_tolerance:
   Tolerance of the first three wavenumbers from 0 :math:`cm^{-1}`.
min_rms_gradient:
   Tolerance of the minimization during geometry optimizations.
eq_of_state:
   Not yet supported in this code.
poly_order: 
   Not yet supported in this code.


Default Units
-------------

**Temperature** -- Kelvin

**Pressure** -- atm

**Energy** -- kcal/mol

**Vector Lengths** -- Angstrom

**Cell Angles** -- Degrees

**Wavenumbers** -- :math:`cm^{-1}`


