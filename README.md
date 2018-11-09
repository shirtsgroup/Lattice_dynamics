Codes created and maintained by: Nate Abraham and Michael Shirts; 
                                 Shirts Group; 
                                 Department of Chemical and Biological Engineering; 
                                 University of Colorado Boulder

Currently, all lattice dynamic codes are designed to run for a desired temperature range and a single pressure.

Supported Programs:
- Tinker Molecular Modeling: Our code assumes that the Tinker binaries are callable without a direct path given
- Test systems: We have provided two funcitons to compute the  lattice energy and wavenumbers. More information can
be found in Test_Systems/Test/0README.md.

Additional packages to install
- Munkres http://software.clapper.org/munkres/


Running the scripts:
1) Contained in this directory, is an input.py file containing all of the user specified options. Move a copy of this
input file to another directory containing the coordinate file (and parameter file).
2) Make necessary changes to the input file. Most of the settings have already been tuned for the best results, for a
large subset of molecules and shouldn't need to be changed.
3) To run the lattice dynamic calculation, in the command line of the working directory type:
        Run_LatticeDynamics.py -i input.inp
If you want to re-run the script and remove all files from a previous run add the flag '-D', as so:
        Run_LatticeDynamics.py -i input.inp -D


Scripts:
- ThermodynamicProperties.py: Contains functions to calculate individual properties for lattice dynamics. 
  Cleaned up from previous version and seperates a lot of functions.
- Wavenumbers.py: Contains all wavenumber functions.
- Expand.py: Contains functions to expand unit cells or determine the local gradient of thermal expansion.
- input.py: General input file with flags for user inputs and runs the general lattice dynamic script at the end.
  'python input.py' will run the main code as long as Run_LatticeDynamics.py is callable from the current directory.
- Run_LatticeDynamics.py: main code to call subroutines and functions and output properties.

