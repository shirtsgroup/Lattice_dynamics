
import os
import numpy as np
import subprocess
import itertools as it
import Expand as Ex
import Wavenumbers as Wvn
import ThermodynamicProperties as Pr


########################################################################################################################
######################                           General Functions                              ########################
########################################################################################################################

def Potential_energy(coordinate_file: str, program: str, parameter_file='': str):
    """
    Computes the potential energy on a per lattice basis for a given cooridnate file

    Paramaeters
    -----------
    coordinate_file : coorindate file specific to the program
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    paramater_file : optional flag for parameter file for Tinker 

    Returns
    -------
    :return: potential energy in kcal/lattice
    """
    if program == 'Tinker':
        U = Tinker_U(coordinate_file, parameter_file)
    elif program == 'Test':
        U = Test_U(coordinate_file)
    elif program == 'CP2K':
        U = CP2K_U(coordinate_file)
    elif program == 'QE':
        U = QE_U(coordinate_file)
    return U

def Lattice_parameters(coordinate_file: str, program: str):
    """
    Returns the lattice parameters of a given coordinate file

    Paramaeters
    -----------
    coordinate_file : coorindate file specific to the program
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]

    Returns
    -------
    :return: list of lattice parameters [ 3x lattice vectors, 3x lattice angles]
    """
    if program == 'Tinker':
        lattice_parameters = Tinker_Lattice_Parameters(coordinate_file)
    elif program == 'Test':
        lattice_parameters = Test_Lattice_Parameters(coordinate_file)
    elif program == 'CP2K':
        lattice_parameters = CP2K_Lattice_Parameters(coordinate_file)
    elif program == 'QE':
        lattice_parameters = QE_Lattice_Parameters(coordinate_file)
    return lattice_parameters

def atoms_count(program: str, coordinate_file: str, molecules_in_coord=1: int):
    """
    Returns either: the number of atoms per coordinate file or the number of atoms per molecule

    Paramaeters
    -----------
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    coordinate_file : coorindate file specific to the program
    molecules_in_coord : optional flag to provide the number of molecules per cooridnate file 

    Returns
    -------
    :return: number of atoms per coordinate file if molecule_in_coord is 1 or atoms per molecule
    """
    if program == 'Tinker':
        atoms_in_coord = len(Return_Tinker_Coordinates(coordinate_file)[:, 0])
    elif program == 'Test':
        atoms_in_coord = len(Test_Wavenumber(coordinate_file, True)) / 3
    elif program == 'CP2K':
        atoms_in_coord = len(Return_CP2K_Coordinates(coordinate_file)[:, 0])
    elif program == 'QE':
        atoms_in_coord = len(QE_atoms_per_molecule(coordinate_file, 1))
    return int(atoms_in_coord / molecules_in_coord)

def program_wavenumbers(coordinate_file: str, tinker_parameter_file: str, output: str, original_coordinate_file: str, program: str, method: str):
    """
    Returns the list of wavenumbers of a given coordinate file

    Paramaeters
    -----------
    coordinate_file : coorindate file specific to the program
    tinker_parameter_file : parameter file for Tinker
    output : name for output files
    original_coordinate_file : used for the Test program
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    method : QHA method being used

    Returns
    -------
    :return: list of wavenumbers in cm^-1 
    """
    if program == 'Tinker':
        wavenumbers = Tinker_Wavenumber(coordinate_file, tinker_parameter_file)
    elif program == 'CP2K':
        wavenumbers = CP2K_Wavenumber(coordinate_file, tinker_parameter_file, Output=output)
    elif program == 'QE':
        wavenumbers = QE_Wavenumber(coordinate_file, tinker_parameter_file, Output=output)
    elif program == 'Test':
        if method == 'GaQ':
            wavenumbers = Test_Wavenumber(coordinate_file,
                                          Ex.Lattice_parameters_to_Crystal_matrix(np.load(original_coordinate_file)))
        else:
            wavenumbers = Test_Wavenumber(coordinate_file, True)
    return wavenumbers

def Wavenumber_and_Vectors(program: str, coordinate_file: str, parameter_file: str):
    """
    Returns the wavenumbers and corresponding eigenvectors

    Paramaeters
    -----------
    coordinate_file : coorindate file specific to the program
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    parameter_file : program specific parameter file

    Returns
    -------
    :return: list of wavenumbers in cm^-1 and eigenvectors
    """
    if program == 'Tinker':
        wavenumbers, eigenvectors = Tinker_Wavenumber_and_Vectors(coordinate_file, parameter_file)
    elif program == 'Test':
        wavenumbers = Test_Wavenumber(coordinate_file, True)
        eigenvectors = np.diag(np.ones(len(wavenumbers)))
        eigenvectors = np.ones((len(wavenumbers), len(wavenumbers)))
        np.fill_diagonal(eigenvectors, 0)
    elif program == 'CP2K':
        wavenumbers, eigenvectors = CP2K_Wavenumber_and_Vectors(coordinate_file, parameter_file)
    elif program == 'QE':
        wavenumbers, eigenvectors = QE_Wavenumber_and_Vectors(coordinate_file, parameter_file)
    return wavenumbers, eigenvectors

def return_coordinates(program: str, coordinate_file: str, lattice_parameters: list):
    """
    Returns the systems Cartesian coordinates

    Paramaeters
    -----------
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    coordinate_file : coorindate file specific to the program
    lattice_parameters : list of lattice vectors and angles

    Returns
    -------
    :return: list of 3xN Cartesian coordinates
    """
    if program == 'Tinker':
        coordinates = Return_Tinker_Coordinates(coordinate_file)
    elif program == 'Test':
        coordinates = Return_Test_Coordinates()
    elif program == 'CP2K':
        coordinates = Return_CP2K_Coordinates(coordinate_file)
    elif program == 'QE':
        coordinates = Return_QE_Coordinates(coordinate_file, lattice_parameters)
    return coordinates

def assign_coordinate_file_ending(program: str):
    """
    Returns the coordinate file ending specific to the program

    Paramaeters
    -----------
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]

    Returns
    -------
    :return: string of the file ending
    """
    if program == 'Tinker':
        return '.xyz'
    elif program == 'Test':
        return '.npy'
    elif program == 'CP2K':
        return '.pdb'
    elif program == 'QE':
        return '.pw'

def output_new_coordinate_file(programi: str, coordinate_file: str, parameter_file: str, coordinates: list, lattice_parameters: list, output: str,
                               min_rms_gradient: float):
    """
    Creates a new program specific coordinate file

    Paramaeters
    -----------
    program : program to use to compute the potential for [Tinker, Test, CP2K, QE]
    coordinate_file : coorindate file specific to the program
    parameter_file : program specific parameter file
    coordinates : 3xN list of Cartesian coordinate files
    lattice_parameters : list of lattice vectors and angles
    output : output to label cooridnate file
    min_rms_gradient : minimization cretierion for geometry optimization
    """
    if program == 'Tinker':
        Ouput_Tinker_Coordinate_File(coordinate_file, parameter_file, coordinates, lattice_parameters, output)
        Tinker_optimization(parameter_file, output + '.xyz', output, min_rms_gradient)
    elif program == 'CP2K':
        Ouput_CP2K_Coordinate_File(coordinate_file, parameter_file, coordinates, lattice_parameters, output)
        CP2K_minimization(parameter_file, output + '.pdb', output, min_rms_gradient)
    elif program == 'QE':
        Output_QE_Coordinate_File(coordinate_file, parameter_file, coordinates, lattice_parameters, output)
        QE_minimization(parameter_file, output + '.qe', output, min_rms_gradient)
    elif program == 'Test':
        np.save(output, lattice_parameters)


########################################################################################################################
######################                         Tinker Molecular Modeling                        ########################
########################################################################################################################

def Tinker_U(Coordinate_file: str, Parameter_file: str):
    """
    Returns the lattice energy in kcal/lattice

    Paramaeters
    -----------
    Coorindate_file : Tinker .xyz file
    Parameter_file : Tinker parameter file

    Returns
    -------
    :return: lattice energy in kcal/lattice
    """
    U = float(subprocess.check_output(
        "analyze %s -k %s E | grep 'Total'| grep -oP '[-+]*[0-9]*\.[0-9]*'" % (Coordinate_file, Parameter_file),
        shell=True))
    return U

def Tinker_Lattice_Parameters(Coordinate_file: str):
    """
    Getting the lattice parameters of the Tinker coordinate file

    Paramaeters
    -----------
    Coorindate_file : Tinker .xyz file

    Returns
    -------
    :return: list of lattice parameters
    """
    with open('%s' % Coordinate_file, 'r') as l:  # Opening coordinate file
        lattice_parameters = [lines.split() for lines in l]
        lattice_parameters = np.array(list(it.zip_longest(*lattice_parameters, fillvalue=' '))).T
        lattice_parameters = lattice_parameters[1, :6].astype(float)
    return lattice_parameters

def Tinker_Wavenumber(Coordinate_file: str, Parameter_file: str):
    """
    Returns the tinker wavenumbers in cm ^-1

    Paramaeters
    -----------
    Coorindate_file : Tinker .xyz file
    Parameter_file : Tinker parameter file

    Returns
    -------
    :return: list of wavenumbers in cm^-1
    """
    # Calling Tinker's vibrate executable and extracting the eigenvalues and wavenumbers of the respective
    # Hessian and mass-weighted Hessian
    eigenvalues_and_wavenumbers = subprocess.check_output("vibrate %s -k %s  CR |  grep -oP '[-+]*[0-9]*\.[0-9]{2,9}'"
                                                          % (Coordinate_file, Parameter_file), shell=True).decode("utf-8")
    # Splitting the outputs into array form
    eigenvalues_and_wavenumbers = eigenvalues_and_wavenumbers.split('\n')
    eigenvalues_and_wavenumbers_hold = []
    for i in eigenvalues_and_wavenumbers:
        if i == '':
            pass
        else:
            eigenvalues_and_wavenumbers_hold.append(float(i))

    # Extracting the wavenumbers and assuring they're sorted from lowest to highest
    wavenumbers = np.sort(np.array(eigenvalues_and_wavenumbers_hold[len(eigenvalues_and_wavenumbers_hold)//2:]))
    return wavenumbers

def Tinker_Wavenumber_and_Vectors(Coordinate_file: str, Parameter_file: str):
    """
    Returns the tinker wavenumbers in cm ^-1 and eigenvectors

    Paramaeters
    -----------
    Coorindate_file : Tinker .xyz file
    Parameter_file : Tinker parameter file

    Returns
    -------
    :return: list of wavenumbers in cm^-1 and eigenvectors
    """

    # Calling Tinker's vibrate executable and extracting the eigenvectors and wavenumbers of the respective
    # Hessian and mass-weighted Hessian
    subprocess.call(['cp', Coordinate_file, 'vector_temp.xyz'])
    output = subprocess.check_output("vibrate vector_temp.xyz -k %s  A |  grep -oP '[-+]*[0-9]*\.[0-9]{2,9}'"
                                                          % (Parameter_file), shell=True).decode("utf-8")

    subprocess.call(['rm vector_temp.*'], shell=True)

    # Finding the number modes in the system
    number_of_modes = int(3 * atoms_count('Tinker', Coordinate_file))

    # Splitting the outputs into array form
    output = output.split('\n')
    output.remove('')
    output = np.array(output).astype(float)

    # Grabbing the wavenumbers
    wavenumbers = np.array(output[number_of_modes: number_of_modes*2]).astype(float)

    # Grabbing the eigenvectors
    eigenvectors = np.zeros((number_of_modes, number_of_modes))
    for i in range(number_of_modes):
        start = number_of_modes*(i + 2) + i + 1
        finish = start + number_of_modes
        eigenvectors[i] = output[start: finish] / np.sqrt(np.sum(output[start: finish]**2))
    return wavenumbers, eigenvectors



def Return_Tinker_Coordinates(Coordinate_file):
    """
    This function opens a Tinker .xyz for a crystal and returns the 3x(number of atoms) matrix

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    """
    with open(Coordinate_file) as f:
        # Opening xyz coordinate file to expand
        coordinates = np.array(list(it.zip_longest(*[lines.split() for lines in f], fillvalue=' '))).T
    coordinates = coordinates[2:, 2:5].astype(float)
    return coordinates

def Ouput_Tinker_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    Tinker .xyz crystal file

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """
    with open(Coordinate_file) as f:
        # Opening xyz coordinate file to expand
        coordinates_template = np.array(list(it.zip_longest(*[lines.split() for lines in f], fillvalue=' '))).T

    coordinates_template[2:, 2:5] = np.around(coordinates, decimals=8).astype('str')
    coordinates_template[1, :6] = lattice_parameters.astype(str)
    string_coordinates = ''
    for i in range(len(coordinates_template[:, 0])):
        for j in range(len(coordinates_template[i, :])):
            string_coordinates = string_coordinates + '    ' + coordinates_template[i, j]
        string_coordinates = string_coordinates + '\n'

    with open(Output + '.xyz', 'w') as file_out:
        file_out.write(string_coordinates)

def Tinker_optimization(Parameter_file, Coordinate_file, Ouptu, min_RMS_gradient):
    with open('minimization.out', 'a') as myfile:
        myfile.write("======================== New Minimization ========================\n")
    output = subprocess.check_output(['optimize', Coordinate_file, '-k', Parameter_file,
                                          str(min_RMS_gradient)]).decode("utf-8")
    subprocess.call(['mv', Coordinate_file + '_2', Coordinate_file])
    with open('minimization.out', 'a') as myfile:
        myfile.write(output)

def tinker_xtalmin(inputs):
    with open('minimization.out', 'a') as myfile:
        myfile.write("==================== Geo. & Lat. Optimization ====================\n")

    G_RMS = 1000000.0
    for i in range(5):
        output = subprocess.check_output(['xtalmin', inputs.coordinate_file, '-k', inputs.tinker_parameter_file,
                                              str(inputs.tinker_xtalmin_tol)]).decode("utf-8")
        subprocess.call(['mv', inputs.coordinate_file + '_2', inputs.coordinate_file])
        with open('minimization.out', 'a') as myfile:
            myfile.write(output)
        if (output.split('\n')[-15].split()[0] == 'OCVM') and (float(output.split('\n')[-17].split()[2]) <= G_RMS):
            G_RMS = float(output.split('\n')[-17].split()[2])
            pass
        else:
            break


########################################################################################################################
######################                               Test System                                ########################
########################################################################################################################

def Test_U(Coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns the potential energy
    Random funcitons can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters
    """
    new_lp = np.load(Coordinate_file)
    U = Test_U_poly(new_lp)
    return U

def Test_U_poly(array):
    p = np.array([  4.23707763e+03,  -4.84369902e+02,  -4.14736558e+02,  -3.52679881e+02,
                   -3.56924620e+00,   1.84204932e+01,   2.91373650e+00,   3.39706395e+01,
                    1.43083158e+01,   1.29649063e+01,   2.44591565e-02,   1.48827075e-01,
                    2.43687850e-02,   1.79715135e+01,   2.30779870e+01,   8.31777610e-02,
                   -3.37390151e+00,  -3.77035091e-01,   1.41456418e+01,   6.39700536e-02,
                   -8.04076344e-01,  -7.14652248e-02,  -1.39556833e-01,  -1.75045683e+00,
                   -3.25448447e-02,   7.63592279e-03,  -1.32279455e-02,  -2.32066032e-02])

    U = p[0] + array[0]*p[1] + array[1]*p[2] + array[2]*p[3] + array[3]*p[4] + array[4]*p[5] + array[5]*p[6] \
        + array[0]**2*p[7] + array[1]**2*p[8] + array[2]**2*p[9] + array[3]**2*p[10] + array[4]**2*p[11] + array[5]**2*p[12] \
        + array[0]*array[1]*p[13] + array[0]*array[2]*p[14] + array[0]*array[3]*p[15] + array[0]*array[4]*p[16] + array[0]*array[5]*p[17] \
        + array[1]*array[2]*p[18] + array[1]*array[3]*p[19] + array[1]*array[4]*p[20] + array[1]*array[5]*p[21] \
        + array[2]*array[3]*p[22] + array[2]*array[4]*p[23] + array[2]*array[5]*p[24] \
        + array[3]*array[4]*p[25] + array[3]*array[5]*p[26] \
        + array[4]*array[5]*p[27]
    return U

def Test_Lattice_Parameters(Coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns them

    **Required Inputs
    Coordinate_file = File containing lattice parameters and atom coordinates
    """
    lattice_parameters = np.load(Coordinate_file)
    return lattice_parameters

def Test_Wavenumber(Coordinate_file, ref_crystal_matrix, function='Test3', Gru=False):
    """
    This function takes a set of lattice parameters in a .npy file and returns a set of wavenumbers
    Random functions can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters and atom coordinates
    """
    lattice_parameters = np.load(Coordinate_file)
    if function == 'Test1':
        wavenumbers = np.array([0., 0., 0., 52., 380., 1570., 3002.])
        for i in np.arange(3, len(wavenumbers[3:])+3):  # probably should be 3?
            wavenumbers[i] = wavenumbers[i]*(1/3.)*(((lattice_parameters[0]-16)/6)**2 + ((lattice_parameters[1] -
                                                                                          12)/5)**2 +
                                                    ((lattice_parameters[2] - 23)/11)**2)
    elif function == 'Test2':
        wavenumbers = np.arange(1, 200)
        wavenumbers = wavenumbers**(5.0/3.0)  # get something in the right range = 400^(4/3) = 2941
        wavenumbers[0:3] = [0, 0, 0]  # zero translation
        [refx, refy, refz] = [lattice_parameters[0]/10., lattice_parameters[1]/12., lattice_parameters[2]/15.]
#        [refx,refy,refz] = [lattice_parameters[0]/5,lattice_parameters[1]/6,lattice_parameters[2]/8]
        for i in range(3,len(wavenumbers[3:])+3):
            wavenumbers[i] = wavenumbers[i]*(1.0/15.0)*(2*refx**4.8 + 10*refy**4.2 + 4*np.sin(2*np.pi*refx) +
                                                        3*refz**4.8)
    elif function == 'Test3':
        # Determining if the wavenumbers at the lattice minimum and the way they change are present
        if os.path.isfile('wvn0_test.npy') and os.path.isfile('wvnChange_test.npy'):
            if np.all(ref_crystal_matrix == True):
                strain = np.zeros(6)
                strain[:3] = (Pr.Volume(Program='Test', Coordinate_file=Coordinate_file)/ 570.88883075)**(1./3.) - 1.
            elif np.all(ref_crystal_matrix == False):
                strain = np.zeros(6)
            else:
                new_crystal_matrix = Ex.Lattice_parameters_to_Crystal_matrix(Lattice_parameters('Test',
                                                                                                   Coordinate_file))
                strain = Pr.RotationFree_StrainArray_from_CrystalMatrix(ref_crystal_matrix, new_crystal_matrix)
                if Gru == True:
                    for i in range(6):
                        if strain[i] != max(strain):
                            strain[i] = 0.
            wvn0 = np.load('wvn0_test.npy')
            change = np.load('wvnChange_test.npy')
            wavenumbers = np.zeros(len(wvn0))
            for i in range(3, len(wavenumbers)):
                wavenumbers[i] = wvn0[i]*np.exp(-1.*np.sum(np.dot(strain, change[i]))) # + strain**2*(change[i]**2)))
        else:
            # Setting random wavenumbers
            wavenumbers = np.zeros(303)
            wavenumbers[3:241] = np.random.uniform(10., 2000., len(wavenumbers[3:241]))
            wavenumbers[241:] = np.random.uniform(2800., 3200., len(wavenumbers[241:]))
            wavenumbers = np.sort(wavenumbers)
            np.save('wvn0_test', wavenumbers)

            # Setting gruneisen parameters
            change = np.zeros((len(wavenumbers), 6))
            for i in range(3,len(wavenumbers)):
                change[i, :3] = np.random.uniform(-2*np.exp(-wavenumbers[i]/200.) - 0.001, 7*np.exp(-wavenumbers[i]/500.) + 0.001)
                for j in range(3, 6):
                    change[j] = np.random.uniform(-0.5*np.absolute(lattice_parameters[j] - 90.)*np.exp(-wavenumbers[i]/100.) - 0.01,
                                                  0.5*np.absolute(lattice_parameters[j] - 90.)*np.exp(-wavenumbers[i]/100.) + 0.01)
            np.save('wvnChange_test.npy', change)
    return wavenumbers

def Return_Test_Coordinates():
    """
    This funciton returns coordinates for the test system
    Because there are no real coordiantes, it just returns a matrix of ones as a holder for the coordinates
    """
    coordinates = np.ones((1, 3))
    return coordinates


########################################################################################################################
######################                                   CP2K                                   ########################
########################################################################################################################

def CP2K_U(coordinate_file):
    """
    This function takes a set of lattice parameters in a .npy file and returns the potential energy
    Random funcitons can be input here to run different tests and implimented new methods efficiently

    **Required Inputs
    Coordinate_file = File containing lattice parameters
    """
    l = open(coordinate_file)
    lines = l.readlines()
    U = float(lines[0].split()[-1])*627.5
    return U

def CP2K_Lattice_Parameters(Coordinate_file):
    """
    This function extracts the lattice parameters from within the Tinker coordinate file

    **Required Inputs
    Coordinate_file = Tinker .xyz file for crystal structure
    """
    with open('%s' % Coordinate_file, 'r') as l:
        lines = l.readlines()
        lattice_parameterstemp = (lines[1].split()[1:7])
    lattice_parameters = np.zeros((6,))
    for x in range(0,6):
        lattice_parameters[x] = float(lattice_parameterstemp[x])
    return lattice_parameters

def CP2K_Wavenumber(coordinatefile, parameter_file, Output):
    import os.path
    if os.path.exists(Output+'.mol') == True:
        wavenumbers = np.zeros((3,))
        wavenumfile = open(Output+'.mol','r')
        lines = wavenumfile.readlines()
        iter = 2
        while '[FR-COORD]' not in lines[iter]:
            wave = lines[iter].split()
            wavenumbers = np.append(wavenumbers, float(wave[0]))
            iter = iter+1
    else:
        print('setting up vibrational analysis')
        subprocess.call(['setup_wavenumber', '-t', 'nma', '-h', coordinatefile[0:-4]])
        subprocess.call(['mpirun', '-np','112','cp2k.popt','-i',coordinatefile[0:-4]+'.inp'])
        subprocess.call(['mv','NMA-VIBRATIONS-1.mol',coordinatefile[0:-4]+'.mol'])
        wavenumbers = np.zeros((3,))
        wavenumfile = open(coordinatefile[0:-4]+'.mol','r')
        lines = wavenumfile.readlines()
        iter = 2
        while '[FR-COORD]' not in lines[iter]:
            wave = lines[iter].split()
            wavenumbers = np.append(wavenumbers, float(wave[0]))
            iter = iter+1
    return wavenumbers

def CP2K_Wavenumber_and_Vectors(Coordinate_file, Parameter_file):
    # Calling CP2K's vibrate executable and extracting the eigenvectors and wavenumbers of the respective
    # .mol file
    import os.path

    if os.path.exists(Coordinate_file[0:-4] + '.mol') == False:
        subprocess.call(['setup_wavenumber', '-t', 'nma', '-h', Coordinate_file[0:-4]])
        subprocess.call(['mpirun', '-np', '112', 'cp2k.popt', '-i', Coordinate_file[0:-4] + '.inp'])
        subprocess.call(['mv', 'NMA-VIBRATIONS-1.mol', Coordinate_file[0:-4] + '.mol'])
    wavenumbers = np.zeros((3,))
    wavenumfile = open(Coordinate_file[0:-4] + '.mol', 'r')
    lines = wavenumfile.readlines()
    iter = 2
    while '[FR-COORD]' not in lines[iter]:
        wave = lines[iter].split()
        wavenumbers = np.append(wavenumbers, float(wave[0]))
        iter = iter + 1
    nummodes = len(wavenumbers)
    eigenvectors = np.zeros((nummodes, nummodes))
    vect = 3
    for y in range(0, len(lines)):
        if lines[y].split()[0] == 'vibration':
            for z in range(1, int(nummodes / 3) + 1):
                modecoord = lines[y + z].split()
                start = int((z - 1) * 3)
                eigenvectors[vect, start:start + 3] = modecoord[:]
            vect += 1
    return wavenumbers, eigenvectors

def Return_CP2K_Coordinates(Coordinate_file):
    """
    This function opens a Tinker .xyz for a crystal and returns the 3x(number of atoms) matrix

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    """
    with open(Coordinate_file) as f:
        coordlines = f.readlines()[2:-1]
    coords = np.zeros((len(coordlines),3))
    for x in range(0,len(coordlines)):
        coords[x,:] = coordlines[x].split()[3:6]
        # Opening xyz coordinate file to expand
    return coords

def Ouput_CP2K_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output):
    """
    This function takes a new set of coordinates and utilizes a previous coordinate file as a template to produce a new
    .pdb crystal file

    **Required Inputs
    Coordinate_file = Tinker .xyz file for a crystal
    Parameter_file = Tinker .key file with force field parameters
    coordinates = New coordinates in a 3x(number of atoms) matrix
    lattice_parameters = lattice parameters as an array ([a,b,c,alpha,beta,gamma])
    Output = file name of new .xyz file
    """
    coordlines = open(Coordinate_file, 'r').readlines()
    xstr = []
    numatoms = np.shape(coordinates)[0]
    for d in range(3):
        xstr.append("%9.3f" % (lattice_parameters[d]))
    for d in range(3,6):
        xstr.append("%7.2f" % (lattice_parameters[d]))
    with open(Output + '.pdb', 'w') as file_out:
        file_out.write('REMARK'+ '\n')
        file_out.write('CRYST1'+str(xstr[0])+xstr[1]+xstr[2]+xstr[3]+xstr[4]+xstr[5]+'\n')
        for x in range(numatoms):
            linenum = coordlines[x+2].split()
            mol = str(linenum[2])
            ty = str(linenum[8])
            line =  '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'.format('ATOM',x+1, mol,'','','','','',coordinates[x,0],coordinates[x,1],coordinates[x,2], 0.0,0.0,ty,'')
            file_out.write(line+'\n')
        file_out.write('END')

def CP2K_minimization(Parameter_file, Coordinate_file, Output, min_RMS_gradient):
    print(Output)
    subprocess.call(['setup_wavenumber','-t','geoopt','-h',Output])
    subprocess.call(['mpirun', '-np','112','cp2k.popt','-i',Output+'.inp' ])
    subprocess.call(['pulllastframe', '-f', 'GEOOPT-GEOOPT.pdb-pos-1.pdb' ,'-n', Output+'.pdb'])


########################################################################################################################
######################                             Quantum Espresso                             ########################
########################################################################################################################

def QE_U(Coordinate_file):
    print('getting energy from'+Coordinate_file)
    with open(Coordinate_file,'r') as lines:
        rlines = lines.readlines()
        energy = float(rlines[2].split()[5]) *313.754
    return energy

def QE_Lattice_Parameters(Coordinate_file):
    lfile = open(Coordinate_file+'bv')
    lines = lfile.readlines()
    matrix = np.zeros((3,3))
    for x in range(0,3):
        vect = lines[x+1]
        matrix[x,0] = float(vect.split()[0])
        matrix[x,1] = float(vect.split()[1])
        matrix[x,2] = float(vect.split()[2])
    lattice_parameters = Ex.crystal_matrix_to_lattice_parameters(np.transpose(matrix))
    return lattice_parameters

def QE_atoms_per_molecule(Coordinate_file, molecules_in_coord):
    lfile = open(Coordinate_file)
    filelines = lfile.readlines()
    numatom = 0
    for x in range(0,len(filelines)):
        if filelines[x].split()[0] in ['C','H','O','S','I','Cl','N']:
            numatom+=1
    atoms_per_molecule = numatom / molecules_in_coord
    return atoms_per_molecule

def QE_Wavenumber(Coordinate_file, parameter_file, Output):
    import os.path
    print('Getting HA Wavenumbers')
    if os.path.exists(Coordinate_file[0:-3] + '.mold') == True:
        # if os.path.exists(Output+'.mold') == True:
        print('recovering wavenumbers')
        wavenumbers = np.zeros((0,))
        wavenumfile = open(Coordinate_file[0:-3] + '.mold', 'r')
        lines = wavenumfile.readlines()
        iter = 2
        while '[FR-COORD]' not in lines[iter]:
            wave = lines[iter].split()
            wavenumbers = np.append(wavenumbers, float(wave[0]))
            iter = iter + 1
    else:
        if os.path.exists(Coordinate_file[0:-3] + '.mold') == False:
            if 'D3' not in os.getcwd():
                subprocess.call(['setup_wavenumberQE', '-t', 'nma', '-h', Coordinate_file[0:-3]])
                subprocess.call(['mpirun', '-np', '112', 'pw.x', '-i', Coordinate_file[0:-3] + 'scf.qe'])
                subprocess.call(['mpirun', '-np', '112', 'ph.x', '-i', Coordinate_file[0:-3] + 'phonon.qe'])
                subprocess.call(['mpirun', '-np', '112', 'dynmat.x', '-i', Coordinate_file[0:-3] + 'matdyn.qe'])
            else:
                subprocess.call(['setup_wavenumberQE', '-t', 'nma', '-h', Coordinate_file[0:-3]])
                subprocess.call(
                    ['mpirun', '-np', '112', '/home/schieber/q-e/bin/pw.x', '-i', Coordinate_file[0:-3] + 'scf.qe'])
                subprocess.call(
                    ['mpirun', '-np', '112', '/home/schieber/q-e/bin/ph.x', '-i', Coordinate_file[0:-3] + 'phonon.qe'])
                subprocess.call(['mpirun', '-np', '112', '/home/schieber/q-e/bin/dynmat.x', '-i',
                                 Coordinate_file[0:-3] + 'matdyn.qe'])

        wavenumbers = np.zeros((0,))
        wavenumfile = open(Coordinate_file[0:-3] + '.mold', 'r')
        lines = wavenumfile.readlines()
        iter = 2
        while '[FR-COORD]' not in lines[iter]:
            wave = lines[iter].split()
            wavenumbers = np.append(wavenumbers, float(wave[0]))
            iter = iter + 1
    return wavenumbers

def QE_Wavenumber_and_Vectors(Coordinate_file):
    import os.path
    print('getting wavenumbers')
    if os.path.exists(Coordinate_file[0:-3]+'.mold') == False:
        print(Coordinate_file)
        subprocess.call(['setup_wavenumberQE', '-t', 'nma', '-h', Coordinate_file[0:-3]])
        #subprocess.call(['mpirun', '-np','112','pw.x','-i',Coordinate_file[0:-3]+'scf.qe'])
        #subprocess.call(['mpirun', '-np','112','ph.x','-i',Coordinate_file[0:-3]+'phonon.qe'])
        #subprocess.call(['mpirun', '-np','112','dynmat.x','-i',Coordinate_file[0:-3]+'matdyn.qe'])
        os.system('mpirun -np 112 pw.x -i '+Coordinate_file[0:-3]+'scf.qe > '+Coordinate_file[0:-3]+'scf.out')
        os.system('mpirun -np 112 ph.x -i '+Coordinate_file[0:-3]+'phonon.qe > '+Coordinate_file[0:-3]+'phonon.out')
        os.system('mpirun -np 112 dynmat.x -i '+Coordinate_file[0:-3]+'matdyn.qe > '+Coordinate_file[0:-3]+'matdyn.out')
    wavenumbers = np.zeros((0,))
    wavenumfile = open(Coordinate_file[0:-3]+'.mold','r')
    lines = wavenumfile.readlines()
    iter = 2
    while '[FR-COORD]' not in lines[iter]:
        wave = lines[iter].split()
        wavenumbers = np.append(wavenumbers, float(wave[0]))
        iter = iter+1
    nummodes = len(wavenumbers)
    eigenvectors = np.zeros((nummodes,nummodes))
    vect = 0
    for y in range(0,len(lines)):
        if lines[y].split()[0] == 'vibration':
            for z in range(1,int(nummodes/3)+1):
                modecoord = lines[y+z].split()
                start = int((z-1)*3)
                eigenvectors[vect,start:start+3] = modecoord[:]
            vect+=1
    return wavenumbers, eigenvectors

def Return_QE_Coordinates(Coordinate_file, lattice_parameters):
    with open(Coordinate_file) as f:
        coordlines = f.readlines()
    coords = np.zeros((len(coordlines)-4,3))
    for x in range(0,len(coordlines)-4):
        coords[x,:] = coordlines[x+4].split()[1:4]
        # Opening xyz coordinate file to expand
    for x in range(0,len(coordlines)):
        if '(crystal)' in coordlines[x].split():
            crystal=True
        else:
            crystal=False
    if crystal == True:
        for x in range(0,len(coordlines)-3):
            coords[x,:] = Ex.crystal_coord_to_cartesian(coords[x,:],lattice_parameters)

    return coords

def Output_QE_Coordinate_File(Coordinate_file, Parameter_file, coordinates, lattice_parameters, Output):
    coordlines = open(Coordinate_file, 'r').readlines()
    struct = open(Output + '.pw', 'w')
    latt = open(Output + '.pwbv', 'w')
    print(coordlines)
    numatoms = np.shape(coordinates)[0]
    print(numatoms)

    for x in range(0, 3):
        struct.write(coordlines[x])
    struct.write('ATOMIC_POSITIONS angstrom' + '\n')
    for x in range(4, 4 + numatoms):
        atomtype = coordlines[x].split()[0]
        struct.write(atomtype + '    ' + str(coordinates[x - 4, 0]) + '   ' + str(coordinates[x - 4, 1]) + '    ' + str(
            coordinates[x - 4, 2]) + '\n')
    latt.write('CELL_PARAMETERS angstrom' + '\n')
    lattice_parameters = np.transpose(Ex.Lattice_parameters_to_Crystal_matrix(lattice_parameters))
    for x in range(0, 3):
        latt.write(str(lattice_parameters[x, 0]) + '   ' + str(lattice_parameters[x, 1]) + '   ' + str(
            lattice_parameters[x, 2]) + '\n')

def QE_minimization(Parameter_file, Coordinate_file, Output, min_RMS_gradient):
    print(Coordinate_file)
    subprocess.call(['setup_wavenumberQE', '-t', 'geoopt', '-h', Output])
    # subprocess.call(['mpirun', '-np','112','pw.x','-i',Output+'.qe' ,'>',Output+'.out'])
    # os.system('mpirun -np 112 pw.x -i '+Output+'.qe > '+Output+'.out')
    # if os.path.exists(Coordinate_file[0:-3]+'.out') == False:
    if 'D3' in os.getcwd():
        os.system('mpirun -np 112 /home/schieber/q-e/bin/pw.x -i ' + Output + '.qe > ' + Output + '.out')
    else:
        os.system('mpirun -np 112 pw.x -i ' + Output + '.qe > ' + Output + '.out')
    subprocess.call(['pulllastframeQE', '-f', Output + '.out', '-n', Output + '.pw'])

