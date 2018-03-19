import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sys import exit
from sys import argv
from pdb import set_trace
import netCDF4 as nc
import mdtraj as md
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import pandas as pd
import pymbar as mb
from pymbar import timeseries
from collections import OrderedDict
from smarty import *
from openforcefield.typing.engines.smirnoff import *
from openforcefield.utils import get_data_filename, generateTopologyFromOEMol, read_molecules
import openmmtools.integrators as ommtoolsints
import mdtraj as md
from itertools import product
import pickle
#-------------------------------------------------
def read_traj(ncfiles,indkeep=0):
    """
    Take multiple .nc files and read in coordinates in order to re-valuate energies based on parameter changes

    Parameters
    -----------
    ncfiles - a list of trajectories in netcdf format

    Returns
    ----------
    data - all of the data contained in the netcdf file
    xyzn - the coordinates from the netcdf in nm
    """

    data = nc.Dataset(ncfiles)
    
    xyz = data.variables['coordinates']
    
    xyzn = Quantity(xyz[indkeep:-1], angstroms)   
    
    lens = data.variables['cell_lengths']
    lensn = Quantity(lens[indkeep:-1], angstroms)

    angs = data.variables['cell_angles']
    angsn = Quantity(angs[indkeep:-1], degrees)

    return data, xyzn, lensn, angsn
#------------------------------------------------------------------
def read_traj_vac(ncfiles,indkeep=0):

    data = nc.Dataset(ncfiles)

    xyz = data.variables['coordinates']
    xyzn = Quantity(xyz[indkeep:-1], angstroms)

    return data, xyzn
#------------------------------------------------------------------
def get_energy(system, positions, vecs):
    """
    Return the potential energy.
    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use
    vecs : simtk.unit.Quantity of dimension 3 with unit of length
        Box vectors to use 
    Returns
    ---------
    energy
    """
    
    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 2. * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    #platform = mm.Platform.getPlatformByName('CUDA')
    #properties = {"CudaPrecision": "mixed","DeterministicForces": "true" }

    #context = mm.Context(system, integrator, platform, properties)
    context = mm.Context(system, integrator, platform)

    context.setPeriodicBoxVectors(*vecs*angstroms)
    context.setPositions(positions)#*angstroms)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() 
    return energy
#------------------------------------------------------------------
def get_energy_vac(system, positions):
    """
    Return the potential energy.
    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use

    Returns
    ---------
    energy
    """

    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 1.5 * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    #platform = mm.Platform.getPlatformByName('CUDA')
    #properties = {"CudaPrecision": "mixed","DeterministicForces": "true" }

    #context = mm.Context(system, integrator, platform, properties)
    context = mm.Context(system, integrator, platform)
 
    context.setPositions(positions)#*angstroms)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    return energy

#---------------------------------------------------
def new_param_energy(coords, params, topology, vecs, P=1.01, T=293.15,NPT=False,V=None,P_conv=1.e5,V_conv=1.e-6,Ener_conv=1.e-3,N_part=250.):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :

             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    P: Pressure of the system. By default set to 1.01 bar.
    T: Temperature of the system. By default set to 300 K.

    Returns
    -------
    E_kn: a kxN matrix of the dimensional energies associated with the forcfield parameters used as input
    u_kn: a kxN matrix of the dimensionless energies associated with the forcfield parameters used as input
    """

    #-------------------
    # CONSTANTS
    #-------------------
    kB = 0.0083145  #Boltzmann constant (Gas constant) in kJ/(mol*K)
    beta = 1/(kB*T)

    #-------------------
    # PARAMETERS
    #-------------------
    params = params

    mol2files = []
    for i in params:
        mol2files.append('monomers/'+i.rsplit(' ',1)[0]+'.mol2')

    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    mols = []
    mol = oechem.OEMol()
    for mol2file in mol2files:
        ifs = oechem.oemolistream(mol2file)
        ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
        mol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, mol):
            oechem.OETriposAtomNames(mol)
            mols.append(oechem.OEGraphMol(mol))
    K = len(params['cyclohexane'].keys())
    
    # Load forcefield file
    #ffxml = 'smirnoff99Frosst_with_AllConstraints.ffxml'#
    #print('The forcefield being used is smirnoff99Frosst_with_AllConstraints.ffxml')
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    print('The forcefield being used is smirnoff99Frosst.ffxml')

    ff = ForceField(ffxml)

    # Generate a topology
    top = topology#generateTopologyFromOEMol(mol)

    #-----------------
    # MAIN
    #-----------------

    # Calculate energies

    E_kn = np.zeros([K,len(coords)],np.float64)
    u_kn = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j
        for k,l in enumerate(params[AlkEthOH_id]):
            print("Anotha one")
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(top,mols,nonbondedMethod=PME,nonbondedCutoff=1.125*nanometers,ewaldErrorTolerance=1.e-5)
                barostat = MonteCarloBarostat(P*bar, T*kelvin, 10)
                system.addForce(barostat)
            for o,p in enumerate(coords):
                e = get_energy(system,p,vecs[o])
               
                if not NPT:
                    E_kn[k,o] = e._value
                    u_kn[k,o] = e._value*beta
                else:
                    E_kn[k,o] = e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part
                    u_kn[k,o] = (e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part)*beta
    
    return E_kn,u_kn

#---------------------------------------------------------------------------
def new_param_energy_vac(coords, params, T=293.15):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :

             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    T: Temperature of the system. By default set to 300 K.

    Returns
    -------
    E_kn: a kxN matrix of the dimensional energies associated with the forcfield parameters used as input
    u_kn: a kxN matrix of the dimensionless energies associated with the forcfield parameters used as input
    """

    #-------------------
    # CONSTANTS
    #-------------------
    kB = 0.0083145  #Boltzmann constant (Gas constant) in kJ/(mol*K)
    beta = 1/(kB*T)

    #-------------------
    # PARAMETERS
    #-------------------
    params = params
    
    # Determine number of states we wish to estimate potential energies for
    mols = []
    for i in params:
        mols.append(i)
    mol = 'monomers/'+mols[0]+'.mol2'
    K = len(params[mols[0]].keys())


    #-------------
    # SYSTEM SETUP
    #-------------
    verbose = False # suppress echos from OEtoolkit functions
    ifs = oechem.oemolistream(mol)
    mol = oechem.OEMol()
    # This uses parm@frosst atom types, so make sure to use the forcefield-flavor reader
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    # Perceive tripos types
    oechem.OETriposAtomNames(mol)

    # Load forcefield file
    #ffxml = 'smirnoff99Frosst_with_AllConstraints.ffxml'#
    #print('The forcefield being used is smirnoff99Frosst_with_AllConstraints.ffxml')
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    print('The forcefield being used is smirnoff99Frosst.ffxml')

    ff = ForceField(ffxml)

    # Generate a topology
    topology = generateTopologyFromOEMol(mol)

    #-----------------
    # MAIN
    #-----------------

    # Calculate energies

    E_kn = np.zeros([K,len(coords)],np.float64)
    u_kn = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j
        for k,l in enumerate(params[AlkEthOH_id]):
            print("Anotha one")
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0]) 
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(topology, [mol])
            #print(newparams)
            for o,p in enumerate(coords):
                e = get_energy_vac(system,p)
                E_kn[k,o] = e._value
                u_kn[k,o] = e._value*beta


    return E_kn,u_kn

#-------------------------------------------------------------------------
kB = 0.0083145 #Boltzmann constant (kJ/mol/K)
T = 293.15 #Temperature (K)
N_Av = 6.02214085774e23 #particles per mole
N_part = 250. #particles of cyclohexane in box

files = ['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080.nc','cyclohexane_250_[#6X4:1]_epsilon0.1093_rmin_half1.9080.nc']
file_strings = [i.rsplit('.',1)[0].split('_',2)[2] for i in files]

file_tups_traj = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.nc'] for i in file_strings]
file_tups_traj_vac = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.nc'] for i in file_strings]

file_tups_sd = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.csv'] for i in file_strings]
file_tups_sd_vac = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.csv'] for i in file_strings]

params = [i.rsplit('.',1)[0].rsplit('_') for i in files]
params = [[i[3][7:],i[5][4:]] for i in params]
MMcyc = 84.15948 #g/mol

states_traj = [[] for i in file_tups_traj] 
states_sd = [[] for i in file_tups_sd]
xyz_orig = [[] for i in file_tups_traj]
xyz_orig_vac = [[] for i in file_tups_traj]
vol_orig = [[] for i in file_tups_traj]
ener_orig = [[] for i in file_tups_sd]
ener_orig_vac = [[] for i in file_tups_sd]
vecs_orig = [[] for i in file_tups_sd]
vol_box_orig = [[] for i in file_tups_sd]
steps_orig_vac = [[] for i in file_tups_sd]
"""
burnin = 500#1949
burnin_vac = 1000#3949
print('burnin bulk = %s' %(burnin))
print('burnin vac = %s' %(burnin_vac))
#print(file_tups_traj)
#print(file_tups_traj_vac)
print( 'Analyzing Cyclohexane neat liquid trajectories')
for j,i in enumerate(file_tups_traj):
    for ii in i:            
        print(ii)
        try:
            data, xyz, lens, angs = read_traj(ii,burnin)            
            #print(xyz)
            #print(lens) 
        except IndexError:
            print( "The trajectory had fewer than %s frames") %(burnin)
            continue 
            
        for m,n in zip(lens,angs):  
            
            vecs = md.utils.lengths_and_angles_to_box_vectors(float(m[0]._value),float(m[1]._value),float(m[2]._value),float(n[0]._value),float(n[1]._value),float(n[2]._value))        
            vecs_orig[j].append(vecs)#*angstroms)
            vol_box_orig[j].append(np.prod(m))

        for pos in xyz:
            xyz_orig[j].append(pos)
    states_traj[j].append(i[0].rsplit('.',1)[0])

for j,i in enumerate(file_tups_traj_vac):
    for ii in i:
        print(ii)
        try:
            data_vac, xyz_vac = read_traj_vac(ii,burnin_vac)
        except IndexError:
            print( "The trajectory had fewer than %s frames") %(burnin)
            continue

    for pos in xyz_vac:
        xyz_orig_vac[j].append(pos)

for j,i in enumerate(file_tups_sd):
    print(i)
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin:-1] for ii in i]
        merged = pd.concat(datasets)
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig[j].append(e)##*(N_Av**(-1.))*N_part) #Energy per mol box

    for dens in merged["Density (g/mL)"]:
        vol_orig[j].append(MMcyc*dens**(-1.))

    states_sd[j].append(i[0].rsplit('.',1)[0])

for j,i in enumerate(file_tups_sd_vac):
    print(i)
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin_vac:-1] for ii in i]
        merged = pd.concat(datasets)
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig_vac[j].append(e)##*N_Av**(-1)) #Energy per mol box
    for s in merged['#"Step"']:
        steps_orig_vac[j].append(s)


#state_coord = params
param_types = ['epsilon','rmin_half']

ener_orig_sub = [[] for i in ener_orig]
vol_orig_sub = [[] for i in vol_orig]
vol_box_orig_sub = [[] for i in vol_box_orig]
ener_orig_vac_sub = [[] for i in ener_orig_vac]
xyz_orig_sub = [[] for i in xyz_orig]
xyz_orig_vac_sub = [[] for i in xyz_orig_vac]
vecs_orig_sub = [[] for i in vecs_orig]
steps_orig_vac_sub = [[] for i in steps_orig_vac]
for ii,value in enumerate(ener_orig):
    ts = [value]
    g = np.zeros(len(ts),np.float64)

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
        else:
            g[i] = timeseries.statisticalInefficiency(t)

    N_k_sub = np.array([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)])
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds = ind[0]

    print("Sub-sampling")
    ener_sub = [value[j] for j in inds]
    vol_sub = [vol_orig[ii][j] for j in inds]
    vol_box_sub = [vol_box_orig[ii][j] for j in inds]
    xyz_sub = [xyz_orig[ii][j] for j in inds]
    vecs_sub = [vecs_orig[ii][j] for j in inds]

    ener_orig_sub[ii] = ener_sub
    vol_orig_sub[ii] = vol_sub
    vol_box_orig_sub[ii] = vol_box_sub
    xyz_orig_sub[ii] = xyz_sub
    vecs_orig_sub[ii] = vecs_sub
#print(vol_orig_sub)
#plt.figure()
#plt.hist(ener_orig_sub[0],bins=50,label='eps = 0.1094',alpha=0.7)
#plt.hist(ener_orig_sub[1],bins=50,label='eps = 0.1093',alpha=0.7)
#plt.legend()
#plt.xlabel('Potential energy (kJ/mole)')
#plt.ylabel('Frequency')
#plt.savefig('subsampled_potentials_comparison_2ref_0.1093.png',dpi=300)
#plt.figure()
#plt.hist(vol_orig_sub[0],bins=50,label='eps = 0.1094',alpha=0.7)
#plt.hist(vol_orig_sub[1],bins=50,label='eps = 0.1093',alpha=0.7)
#plt.legend()
#plt.xlabel('molar volume (mL/mole)')
#plt.ylabel('Frequency')
#plt.savefig('subsampled_volumes_comparison_2ref_0.1093.png',dpi=300)
#set_trace()
for ii,value in enumerate(ener_orig_vac):
    ts = [value]
    g = np.zeros(len(ts),np.float64)

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
        else:
            g[i] = timeseries.statisticalInefficiency(t)

    N_k_vac_sub = np.array([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)])
    ind_vac = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds_vac = ind_vac[0]

    print("Sub-sampling")
    ener_vac_sub = [value[j] for j in inds_vac]
    xyz_vac_sub = [xyz_orig_vac[ii][j] for j in inds_vac]
    steps_vac_sub = [steps_orig_vac[ii][j] for j in inds_vac]

    ener_orig_vac_sub[ii] = ener_vac_sub
    xyz_orig_vac_sub[ii] = xyz_vac_sub
    steps_orig_vac_sub[ii] = steps_vac_sub


#plt.figure()
#plt.plot(steps_orig_vac_sub[0],ener_orig_vac_sub[0])
#plt.xlabel('Timestep (units of 0.8 fs)')
#plt.ylabel('Potential Energy (kJ/mole)')
#plt.tight_layout()
#plt.savefig('Vacuum_potential_trace_HConstraints_10ns.png',dpi=300)

#print(ener_orig_vac_sub)
######################################################################################### WORKS^^
# Define new parameter states we wish to evaluate energies at
vol_sub = np.array(vol_orig_sub)
eps_vals = np.linspace(float(argv[1]),float(argv[2]),2)
rmin_vals = np.linspace(float(argv[3]),float(argv[4]),2)
eps_vals = [str(a) for a in eps_vals]
rmin_vals = [str(a) for a in rmin_vals]
new_states = list(product(eps_vals,rmin_vals))

new_states = list(set(new_states))

orig_state = [('0.1094', '1.9080'),('0.1093', '1.9080')]#[('0.1186','1.8642')]#[('0.1094', '1.9080'),('0.1186','1.8642')]

for i in new_states:
    if i in orig_state:
        new_states.remove(i)

N_eff_list = []
param_type_list = []
param_val_list = []

#state_coords = []
#state_coords.append(orig_state)
#orig_state.remove(('0.1094', '1.908'))
state_coords = orig_state
for i in new_states:
     state_coords.append(i)
print(state_coords)

filename = 'packmol_boxes/cyclohexane_250.pdb'
pdb = PDBFile(filename)
nBoots_work = 2
nBoots_work_vac = 2

u_kns = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_sub])],np.float64)
E_kns = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_sub])],np.float64)
u_kns_vac = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_vac_sub])],np.float64)
E_kns_vac = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_vac_sub])],np.float64)

vol_subs = np.zeros([1,sum([len(i) for i in ener_orig_sub])],np.float64)
vol_ind = 0
for i,j in enumerate(vol_sub):
    vol_subs[0,vol_ind:vol_ind+len(j)] = j 
    vol_ind+=len(j)
#print(vol_subs)
#exit() 
#vol_subs = vol_sub[0]
#print(vol_subs)
N_k = np.zeros(len(state_coords),np.int64)
N_k_vac = np.zeros(len(state_coords),np.int64)

for k,i in enumerate(ener_orig_sub): 
    N_k[k] = len(i)
for k,i in enumerate(ener_orig_vac_sub):
    N_k_vac[k] = len(i)

#N_all_k = 
print(N_k)
print(N_k_vac)
index = 0
index_vac = 0
for ii,value in enumerate(xyz_orig_sub):
    MBAR_moves = state_coords
    print( "Number of MBAR calculations for liquid cyclohexane: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D} 
        
    # Produce the u_kn matrix for MBAR based on the subsampled configurations
    E_kn, u_kn = new_param_energy(xyz_orig_sub[ii],D_mol, pdb.topology,vecs_orig_sub[ii],P=1.01,T = 293.15,NPT=True,V=vol_sub[ii])

    curr_k = 0

    for E_n,u_n in zip(E_kn,u_kn): 
        E_kns[curr_k,index:index+len(E_n)] = E_n
        u_kns[curr_k,index:index+len(u_n)] = u_n
        curr_k += 1
    
    index += len(E_kn[0])

<<<<<<< HEAD
    #K,N = np.shape(u_kn)
 
    #MBAR_moves = state_coords
    #print( "Number of MBAR calculations for liquid cyclohexane: %s" %(len(MBAR_moves)))
    #print( "starting MBAR calculations")
    #D = OrderedDict()
    #for i,val in enumerate(MBAR_moves):
    #    D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    #D_mol = {'cyclohexane' : D} 
        
    # Produce the u_kn matrix for MBAR based on the subsampled configurations
    #E_kn, u_kn = new_param_energy(xyz_orig_sub[ii], D_mol, pdb.topology, vecs_orig_sub[ii], T = 293.15)
    #E_kn_292, u_kn_292 = new_param_energy(xyz_sub[ii], D_mol, pdb.topology, vecs_orig_sub[ii], T = 292.15)
    #E_kn_294, u_kn_294 = new_param_energy(xyz_sub[ii], D_mol, pdb.topology, vecs_orig_sub[ii], T = 294.15)
    
=======
nBoots_work = 1000
u_kn = pickle.load( open( "pickles/u_kn_bulk_forMRS.pkl", "rb" ) )
u_kn_vac = pickle.load( open( "pickles/u_kn_vac_forMRS.pkl", "rb" ) )   
E_kn = pickle.load( open( "pickles/E_kn_bulk_forMRS.pkl", "rb" ) )          #kJ/mol
E_kn_vac = pickle.load( open( "pickles/E_kn_vac_forMRS.pkl", "rb" ) )       #kJ/mol
vol_sub = pickle.load( open( "pickles/vol_sub_forMRS.pkl", "rb" ) )         #mL/mol
MBAR_moves = pickle.load( open("pickles/param_states_forMRS.pkl", "rb" ) )

print(np.shape(u_kn))
print(np.shape(vol_sub))
exit()
for ii,value in enumerate(vol_sub):
>>>>>>> 2f9b12ad12e8d863f3683637ba5c124a55230f0d
    # Alter u_kn by adding reduced pV term and create an H_kn matrix
    #We then need to convert from 1 atm * 100 nm^3 to kJ/mol. Easiest is to go to liter-atm/mol, 
    # and then convert to kJ/mol.
    # 1 atm*nm^3   * 1 m^3 / 10^27 nm *  1000 L / 1 m^3  *  6.02214 x 10^23 things / 1 mol  * 
    # (0.00831446 kJ / 0.0820573 L*atm) = 0.0610194 kJ/mol. So the PV terms should be around 6 kJ/mol
    #Beta is 0.410, so the betaPV terms should be around 2.5. 
 
    #BCM: Volumes are already on a per mol basis (mL/mol). Convert from mL to m^3 *=1.e-6. 
    #     Pa*m^3/(mol particle) [=] J/(mol particle), so multiply by 1.e-3 to get units of kJ/(mol particle). 
    #     kJ/(mol particle) * (N_Av)**-1 (mol particle/particle) * 250. particle/(mol box) [=] kJ/(mol box)
    #J2kJ = 1.e-3
    #A32m3 = 1.e-30
    #cm32m3 = 1.e-6
    #P = 101000. #Pa
    #betapV = (1./(kB*T))*P*np.array(vol_sub)*cm32m3*J2kJ*N_part#(N_Av**(-1.))
    # Directly use calculated box volume trajectory
    # ang**3 = 1e-30 m**3 ::: Pa*m**3 = J ::: 1000 J = 1 kJ ::: 
    #betapVbox = (1./(kB*T))*P*np.array(vol_box_sub)*J2kJ*A32m3*N_Av
    #print(betapVmol)
    #print(betapVbox)
    
    #H_kn = E_kn + kB*T*betapV 

    #K,N = np.shape(u_kn)
                    
    
    #N_k[0] = int(N)

    print( "Number of MBAR calculations for cyclohexane in vacuum: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D}
 
    #Produce the u_kn matrix for MBAR based on the subsampled configurations
    E_kn_vac, u_kn_vac = new_param_energy_vac(xyz_orig_vac_sub[ii], D_mol, T = 293.15)

    curr_k_vac = 0

    for E_n_vac,u_n_vac in zip(E_kn_vac,u_kn_vac):
        E_kns_vac[curr_k_vac,index_vac:index_vac+len(E_n_vac)] = E_n_vac
        u_kns_vac[curr_k_vac,index_vac:index_vac+len(u_n_vac)] = u_n_vac
        curr_k_vac += 1

    index_vac += len(E_kn_vac[0])
"""
u_kns = pickle.load( open( "pickles/u_kn_test_MRS.pkl", "rb" ) )
u_kns_vac = pickle.load( open( "pickles/u_kn_vac_test_MRS.pkl", "rb" ) )
E_kns = pickle.load( open( "pickles/u_kn_test_MRS.pkl", "rb" ) )*kB*T          #kJ/mol
E_kns_vac = pickle.load( open( "pickles/u_kn_vac_test_MRS.pkl", "rb" ) )*kB*T       #kJ/mol
vol_subs = pickle.load( open( "pickles/vol_test_MRS.pkl", "rb" ) )         #mL/mol
MBAR_moves = pickle.load( open("pickles/param_states_test_MRS.pkl", "rb" ) )
N_k = pickle.load( open("pickles/N_k_test_MRS.pkl", "rb" ) )
N_k_vac = pickle.load( open("pickles/N_k_vac_test_MRS.pkl", "rb" ) )

K,N = np.shape(u_kns)
K_vac,N_vac = np.shape(u_kns_vac)

nBoots_work = 2
nBoots_work_vac = 2

#implement bootstrapping to get variance estimates
N_eff_boots = []
u_kn_boots = []
V_boots = []
dV_boots =[]
E_boots = []
dE_boots = []
for n in range(nBoots_work):
    for k in range(len(N_k)):
        if N_k[k] > 0:
            if (n == 0):
                booti = np.array(range(int(sum(N_k))),int)
            else:
                booti = np.random.randint(int(sum(N_k)), size = int(sum(N_k)))
    print("Bootstrap sample %s of 1000" %(n+1))
    E_kn_boot = E_kns[:,booti]       
    u_kn_boot = u_kns[:,booti]
    vol_sub_boot = vol_subs[:,booti]
    
    u_kn_boots.append(u_kns)
    
    # Initialize MBAR with Newton-Raphson
    # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
    ########################################################################################  
    if (n==0):
        initial_f_k = None # start from zero 
    else:
        initial_f_k = mbar.f_k # start from the previous final free energies to speed convergence        
   
    mbar = mb.MBAR(u_kn_boot, N_k, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)
    O_ij = mbar.computeOverlap()
    print(O_ij[2])      
    #N_eff = mbar.computeEffectiveSampleNumber(verbose=True)
    N_eff = mbar.computeEffectiveSampleNumber(verbose=False)
    
    N_eff_boots.append(N_eff)

    (Vol_expect,dVol_expect) = mbar.computeExpectations(vol_sub_boot,state_dependent = False)
   
    V_boots.append(Vol_expect)
    dV_boots.append(dVol_expect)

    # Calculating Cp as (<E**2> - <E>**2)/(kB*T**2). DID originally try the alternate
    # <(E - <E>)**2>/(kB*T**2) [hence the `C_p_expect_meth2` for second method], but they
    # yielded the same results and I chose one of them.

    (E_expect, dE_expect) = mbar.computeExpectations(E_kn_boot,state_dependent = True)
 
    E_boots.append(E_expect)
    dE_boots.append(dE_expect)
     
    #set_trace()     
u_kn = u_kn_boots[0]
N_eff = N_eff_boots[0]
Vol_expect = V_boots[0]
dVol_expect = dV_boots[0]
E_expect = E_boots[0]    
dE_expect = dE_boots[0]
     
E_boots_vt = np.vstack(E_boots)
V_boots_vt = np.vstack(V_boots)

E_bootstrap = [np.mean(E_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Mean of E calculated with bootstrapping
dE_bootstrap = [np.std(E_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Standard error of E from bootstrap
Vol_bootstrap = [np.mean(V_boots_vt[1:,a]) for a in range(np.shape(V_boots_vt)[1])] #Mean of V calculated with bootstrapping
dVol_bootstrap = [np.std(V_boots_vt[1:,a]) for a in range(np.shape(V_boots_vt)[1])] #Standard error of V from bootstrap   
   
N_eff_vac_boots = []
u_kn_vac_boots = []
E_vac_boots = []
dE_vac_boots = []
for n in range(nBoots_work_vac):
    for k in range(len(N_k_vac)):
        if N_k_vac[k] > 0:
            if (n == 0):
                booti = np.array(range(int(sum(N_k_vac))),int)
            else:
                booti = np.random.randint(int(sum(N_k_vac)), size = int(sum(N_k_vac)))

    E_kn_vac = E_kns_vac[:,booti]
    u_kn_vac = u_kns_vac[:,booti]
        
    u_kn_vac_boots.append(u_kn_vac) 

    # Initialize MBAR with Newton-Raphson
    # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
    ########################################################################################
    if (n==0):
        initial_f_k = None # start from zero
    else:
        initial_f_k = mbar_vac.f_k # start from the previous final free energies to speed convergence

    mbar_vac = mb.MBAR(u_kn_vac, N_k_vac, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)

    #N_eff_vac = mbar_vac.computeEffectiveSampleNumber(verbose=True)
    N_eff_vac = mbar_vac.computeEffectiveSampleNumber(verbose=False)
        
    N_eff_vac_boots.append(N_eff_vac)        

    (E_vac_expect, dE_vac_expect) = mbar_vac.computeExpectations(E_kn_vac,state_dependent = True)

    E_vac_boots.append(E_vac_expect)
    dE_vac_boots.append(dE_vac_expect)

u_kn_vac = u_kn_vac_boots[0]
N_eff_vac = N_eff_vac_boots[0]
E_vac_expect = E_vac_boots[0]
dE_vac_expect = dE_vac_boots[0]

E_vac_boots_vt = np.vstack(E_vac_boots) 

E_vac_bootstrap = [np.mean(E_vac_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Mean of E calculated with bootstrapping
dE_vac_bootstrap = [np.std(E_vac_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Standard error of E from bootstrap
 
    
#print(dE_vac_expect)
#print(dE_expect)
#print(dE_vac_bootstrap)
#print(dE_bootstrap)
#print(dVol_expect)
#print(dVol_bootstrap)
#######################################################################################
# Calculate heat of vaporization
#######################################################################################
Hvap_expect = [((ener_vac - (1/250.)*ener) + 101000.*1.e-3*(0.024465 - v*1.e-6)) for ener_vac,ener,v in zip(E_vac_expect,E_expect,Vol_expect)]
dHvap_expect = [np.sqrt(dener_vac**2 + ((1/250.)*dener)**2 + (101000.*1.e-3*1.e-6*dv)**2) for dener_vac,dener,dv in zip(dE_vac_expect,dE_expect,dVol_expect)]  
Hvap_bootstrap = [((ener_vac - (1/250.)*ener) + 101000.*1.e-3*(0.024465 - v*1.e-6)) for ener_vac,ener,v in zip(E_vac_bootstrap,E_bootstrap,Vol_bootstrap)]
dHvap_bootstrap = [np.sqrt(dener_vac**2 + ((1/250.)*dener)**2 + (101000.*1.e-3*1.e-6*dv)**2) for dener_vac,dener,dv in zip(dE_vac_bootstrap,dE_bootstrap,dVol_bootstrap)]
#print(dHvap_expect)
#print(dHvap_bootstrap)
print(Vol_expect,Hvap_expect,dVol_expect,dVol_bootstrap,dHvap_expect,dHvap_bootstrap)    

df = pd.DataFrame(
                      {'param_value': MBAR_moves,
                       'Vol_expect (mL/mol)': Vol_expect,
                       'dVol_expect (mL/mol)': dVol_expect,
                       'Hvap_expect (kJ/mol)': Hvap_expect,
                       'dHvap_expect (kJ/mol)': dHvap_expect,
                       'Vol_bootstrap (mL/mol)': Vol_bootstrap,
                       'dVol_bootstrap (mL/mol)': dVol_bootstrap,
                       'Hvap_bootstrap (kJ/mol)': Hvap_bootstrap,
                       'dHvap_bootstrap (kJ/mol)': dHvap_bootstrap,
                       'N_eff': N_eff
                      })

df.to_csv('two_state_MBAR_estimates.csv')    
#df.to_csv('Lang_2_baro10step_pme1e-5/MBAR_estimates/MBAR_estimates_[6X4:1]_eps'+argv[1]+'-'+argv[2]+'_rmin'+argv[3]+'-'+argv[4]+'_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_2ref_0.1093_HVap.csv',sep=';')
   
#with open('param_states_1fs.pkl', 'wb') as f:
#    pickle.dump(MBAR_moves, f)
#with open('u_kn_bulk_1fs.pkl', 'wb') as f:
#    pickle.dump(u_kn, f)
#with open('u_kn_vac_1fs.pkl', 'wb') as f:
#    pickle.dump(u_kn_vac, f)

