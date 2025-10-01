"""
A class for building molecular object adapters. Adapters for various approaches to build
the molecular info and properties (hamiltonian, rdms, etc...).
"""
# import operator
# import itertools
import numpy as np
import copy
from abc import ABC, abstractmethod
from qforte.helper.df_ham_helper import *

import qforte

from scipy.linalg import expm
from qforte.system.molecular_info import Molecule
from qforte.utils import transforms as tf


import json

try:
    import psi4
    use_psi4 = True
except:
    use_psi4 = False

try:
    import pyscf
    from pyscf import gto, scf, mp, fci, ao2mo, symm, mcscf
    from pyscf.mcscf import avas
    use_pyscf = True
except:
    use_pyscf = False


def create_psi_mol(**kwargs):
    """Builds a qforte Molecule object directly from a psi4 calculation.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    kwargs.setdefault('symmetry', 'c1')
    kwargs.setdefault('charge', 0)
    kwargs.setdefault('multiplicity', 1)

    mol_geometry = kwargs['mol_geometry']
    basis = kwargs['basis']
    multiplicity = kwargs['multiplicity']
    charge = kwargs['charge']

    qforte_mol = Molecule(mol_geometry = mol_geometry,
                               basis = basis,
                               multiplicity = multiplicity,
                               charge = charge)

    if not use_psi4:
        raise ImportError("Psi4 was not imported correctely.")

    # By default, the number of frozen orbitals is set to zero
    kwargs.setdefault('num_frozen_docc', 0)
    kwargs.setdefault('num_frozen_uocc', 0)

    # run_scf is not read, because we always run SCF to get a wavefunction object.
    kwargs.setdefault('run_mp2', False)
    kwargs.setdefault('run_ccsd', False)
    kwargs.setdefault('run_cisd', False)
    kwargs.setdefault('run_fci', False)

    # Setup psi4 calculation(s)
    psi4.set_memory('2 GB')
    psi4.core.set_output_file(kwargs['filename']+'.out', False)

    p4_geom_str =  f"{int(charge)}  {int(multiplicity)}"
    
    for geom_line in mol_geometry:
        p4_geom_str += f"\n{geom_line[0]}  {geom_line[1][0]}  {geom_line[1][1]}  {geom_line[1][2]}"
    p4_geom_str += f"\nsymmetry {kwargs['symmetry']}"
    p4_geom_str += f"\nunits angstrom"

    print(' ==> Psi4 geometry <==')
    print('-------------------------')
    print(p4_geom_str)

    p4_mol = psi4.geometry(p4_geom_str)

    scf_ref_type = "rhf" if multiplicity == 1 else "rohf"

    psi4.set_options({'basis': basis,
              'scf_type': 'pk',
              'reference' : scf_ref_type,
              'e_convergence': 1e-8,
              'd_convergence': 1e-8,
              'ci_maxiter': 100,
              'num_frozen_docc' : kwargs['num_frozen_docc'],
              'num_frozen_uocc' : kwargs['num_frozen_uocc'],
              'mp2_type': "conv"})

    # run psi4 caclulation
    p4_Escf, p4_wfn = psi4.energy('SCF', return_wfn=True)

    # Run additional computations requested by the user
    if kwargs['run_mp2']:
        qforte_mol.mp2_energy = psi4.energy('MP2')

    if kwargs['run_ccsd']:
        qforte_mol.ccsd_energy = psi4.energy('CCSD')

    if kwargs['run_cisd']:
        qforte_mol.cisd_energy = psi4.energy('CISD')

    if kwargs['run_fci']:
        if kwargs['num_frozen_uocc'] == 0:
            qforte_mol.fci_energy = psi4.energy('FCI')
        else:
            print('\nWARNING: Skipping FCI computation due to a Psi4 bug related to FCI with frozen virtuals.\n')

    # Get integrals using MintsHelper.
    mints = psi4.core.MintsHelper(p4_wfn.basisset())

    C = p4_wfn.Ca_subset("AO", "ALL")

    scalars = p4_wfn.scalar_variables()

    p4_Enuc_ref = scalars["NUCLEAR REPULSION ENERGY"]

    # Do MO integral transformation
    mo_teis = np.asarray(mints.mo_eri(C, C, C, C))
    mo_oeis = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    mo_oeis = np.einsum('uj,vi,uv', C, C, mo_oeis)
    nmo = np.shape(mo_oeis)[0]
    
    nalpha = p4_wfn.nalpha()
    nbeta = p4_wfn.nbeta()
    nel = nalpha + nbeta
    frozen_core = p4_wfn.frzcpi().sum()
    frozen_virtual = p4_wfn.frzvpi().sum()

    # Get symmetry information
    orbitals = []
    for irrep, block in enumerate(p4_wfn.epsilon_a_subset("MO", "ACTIVE").nph):
        for orbital in block:
            orbitals.append([orbital, irrep])

    orbitals.sort()
    hf_orbital_energies = []
    orb_irreps_to_int = []
    for row in orbitals:
        hf_orbital_energies.append(row[0])
        orb_irreps_to_int.append(row[1])

    point_group = p4_mol.symmetry_from_input().lower()
    irreps = qforte.irreps_of_point_groups(point_group)
    orb_irreps = [irreps[i] for i in orb_irreps_to_int]

    # If frozen_core > 0, compute the frozen core energy and transform one-electron integrals

    frozen_core_energy = 0

    if frozen_core > 0:
        for i in range(frozen_core):
            frozen_core_energy += 2 * mo_oeis[i, i]

        # Note that the two-electron integrals out of Psi4 are in the Mulliken notation
        for i in range(frozen_core):
            for j in range(frozen_core):
                frozen_core_energy += 2 * mo_teis[i, i, j, j] - mo_teis[i, j, j, i]

        # Incorporate in the one-electron integrals the two-electron integrals 
        # involving both frozen and non-frozen orbitals.
        # This also ensures that the correct orbital energies will be obtained.

        for p in range(frozen_core, nmo - frozen_virtual):
            for q in range(frozen_core, nmo - frozen_virtual):
                for i in range(frozen_core):
                    mo_oeis[p, q] += 2 * mo_teis[p, q, i, i] - mo_teis[p, i, i, q]

    # Make hf_reference
    hf_reference = [1] * (nel - 2 * frozen_core) + [0] * (2 * (nmo - frozen_virtual) - nel)

    # Build second quantized Hamiltonian
    Hsq = qforte.SQOperator()
    Hsq.add(p4_Enuc_ref + frozen_core_energy, [], [])

    # Note index is over active space orbitals if frozen occ or unocc
    for i in range(frozen_core, nmo - frozen_virtual):
        ia = (i - frozen_core)*2
        ib = (i - frozen_core)*2 + 1
        for j in range(frozen_core, nmo - frozen_virtual):
            ja = (j - frozen_core)*2
            jb = (j - frozen_core)*2 + 1

            Hsq.add(mo_oeis[i,j], [ia], [ja])
            Hsq.add(mo_oeis[i,j], [ib], [jb])

            for k in range(frozen_core, nmo - frozen_virtual):
                ka = (k - frozen_core)*2
                kb = (k - frozen_core)*2 + 1
                for l in range(frozen_core, nmo - frozen_virtual):
                    la = (l - frozen_core)*2
                    lb = (l - frozen_core)*2 + 1

                    if(ia!=jb and kb != la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, jb], [kb, la] ) # abba
                    if(ib!=ja and ka!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, ja], [ka, lb] ) # baab

                    if(ia!=ja and ka!=la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, ja], [ka, la] ) # aaaa
                    if(ib!=jb and kb!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, jb], [kb, lb] ) # bbbb

    # Set attributes
    qforte_mol.nuclear_repulsion_energy = p4_Enuc_ref
    qforte_mol.hf_energy = p4_Escf
    qforte_mol.hf_reference = hf_reference
    qforte_mol.sq_hamiltonian = Hsq
    if kwargs['build_qb_ham']:
        qforte_mol.hamiltonian = Hsq.jw_transform()
    else:
        Hsq.simplify()
        qforte_mol.hamiltonian = qforte.QubitOperator()

    qforte_mol.point_group = [point_group, irreps]
    qforte_mol.orb_irreps = orb_irreps
    qforte_mol.orb_irreps_to_int = orb_irreps_to_int
    qforte_mol.hf_orbital_energies = hf_orbital_energies
    qforte_mol.frozen_core = frozen_core
    qforte_mol.frozen_virtual = frozen_virtual
    qforte_mol.frozen_core_energy = frozen_core_energy

    if kwargs['build_df_ham']:
        if not kwargs['store_mo_ints']:
            raise ValueError("store_mo_ints must be True if you want to build_df_ham")
        else:
            p4_mo_oeis = copy.deepcopy(mo_oeis)
            p4_mo_teis = copy.deepcopy(mo_teis)

    if kwargs['store_mo_ints_np']:
        # Resize mo_oeis and mo_teis if there are frozen orbitals...
        if(frozen_core or frozen_virtual):
            # raise ValueError("This doesn't work..")
            start = frozen_core
            end = nmo - frozen_virtual
            mo_oeis = copy.deepcopy(mo_oeis[start:end, start:end])
            mo_teis = copy.deepcopy(mo_teis[start:end, start:end, start:end, start:end])

        # keep ordering consistant with openfermion eri tensors
        mo_teis_trans_np = copy.deepcopy(np.asarray(mo_teis.transpose(0, 2, 3, 1), order='C'))

        # save numpy copies
        qforte_mol.mo_oeis_np = copy.deepcopy(mo_oeis)
        qforte_mol.mo_teis_np = copy.deepcopy(mo_teis_trans_np)

    if kwargs['store_mo_ints']:

        # Resize mo_oeis and mo_teis if there are frozen orbitals...
        if(frozen_core or frozen_virtual):
            # raise ValueError("This doesn't work..")
            start = frozen_core
            end = nmo - frozen_virtual
            mo_oeis = copy.deepcopy(mo_oeis[start:end, start:end])
            mo_teis = copy.deepcopy(mo_teis[start:end, start:end, start:end, start:end])

        # keep ordering consistant with openfermion eri tensors
        mo_teis = np.asarray(mo_teis.transpose(0, 2, 3, 1), order='C')

        # save numpy copies
        # qforte_mol.mo_oeis_np = copy.deepcopy(mo_oeis)
        # qforte_mol.mo_teis_np = copy.deepcopy(mo_teis)

        # Save data to a file
        # np.savez(
        #     "mol_e0_h1e_h2e.npz", 
        #     e0=p4_Enuc_ref, 
        #     h1e=mo_oeis, 
        #     h2e=mo_teis)

        # need restricted version
        h2e_rest = copy.deepcopy(np.einsum("ijlk", -0.5 * mo_teis))

        # additoinal manipulation
        h1e = copy.deepcopy(mo_oeis)
        h2e = np.moveaxis(copy.deepcopy(h2e_rest), 1, 2) * (-1.0)
        h1e -= np.einsum('ikkj->ij', h2e)

        # just going to precumpute the einseum (for now)
        h2e_einsum = copy.deepcopy(h2e + np.einsum('ijkl->klij', h2e))

        # allocate qf tensors
        qf_mo_oeis = qforte.Tensor(shape=np.shape(h1e), name='mo_oeis')
        qf_mo_teis = qforte.Tensor(shape=np.shape(h2e), name='mo_teis')
        qf_mo_teis_einsum = qforte.Tensor(shape=np.shape(h2e_einsum), name='mo_teis_einsum')
        
        # fill qf tensors
        qf_mo_oeis.fill_from_nparray(h1e.ravel(), np.shape(h1e))
        qf_mo_teis.fill_from_nparray(h2e.ravel(), np.shape(h2e)) 
        qf_mo_teis_einsum.fill_from_nparray(
            h2e_einsum.ravel(), 
            np.shape(h2e_einsum)) 

        qforte_mol.mo_oeis = qf_mo_oeis
        qforte_mol.mo_teis = qf_mo_teis
        qforte_mol.mo_teis_einsum = qf_mo_teis_einsum

        # TODO(Nick), If we want better controll over this, it there shuld be a molecule member function that
        # builds the df_ham from the stored mo_oeis and mo_teis rather than building it when psi4 
        # is initially run!
        if kwargs['build_df_ham']:
            # NOTE: build_df_ham should not be called unless store_mo_ints is True, if called here,
            # mo_oeis and mo_teis are defined using openfermion ordering.

            # Load h1e and h2e for Li-H the .npz file
            # loaded_data = np.load('of_mol_e0_h1e_h2e.npz')
            # e0 = loaded_data['e0']
            # p4_mo_oeis = loaded_data['h1e']
            # p4_mo_teis = loaded_data['h2e']


            # keep ordering consistant with openfermion eri tensors
            p4_mo_teis = np.asarray(p4_mo_teis.transpose(0, 2, 3, 1), order='C')

            # # need restricted version
            # p4_mo_teis2 = copy.deepcopy(np.einsum("ijlk", -0.5 * p4_mo_teis))

            # # additoinal manipulation
            # h1e_2 = copy.deepcopy(p4_mo_oeis)
            # h2e_2 = np.moveaxis(copy.deepcopy(h2e_rest_2), 1, 2) * (-1.0)
            # h1e_2 -= np.einsum('ikkj->ij', h2e_rest_2)


            # do first factorization from integrals
            ff_eigenvalues, one_body_squares, one_body_correction = first_factorization(
                tei = p4_mo_teis,
                lmax=None, # change if we want 
                spin_basis=False,
                threshold=kwargs['df_icut']) # may be very important to play with
            
            # do second factorization based on integrals and first factorizaiotn
            scaled_density_density_matrices, basis_change_matrices = second_factorization(
                ff_eigenvalues, 
                one_body_squares)
            
            #          ===> get the trotter versions of the matricies <====

            # don't need time_scaled_rho_rho_matrices for now, 
            # will handle in FCI computer application funciton,
            # or perhaps in some DFHamiltonain helper funciton 
            # time_scaled_rho_rho_matrices = []
            
            # get the zero leaf, set to zero, this will make it more obvious if you try to evolve
            # without setting the first leaf...
            trotter_basis_change_matrices = [
                # basis_change_matrices[0] @ expm(-1.0j * (p4_mo_oeis + one_body_correction[::2, ::2]))
                np.zeros(shape=(nmo,nmo))
            ]

            # get the other "t" leaves (as Rob calls them)
            for ii in range(len(basis_change_matrices) - 1):

                trotter_basis_change_matrices.append(
                    basis_change_matrices[ii + 1] @ basis_change_matrices[ii].conj().T)
            
            trotter_basis_change_matrices.append(basis_change_matrices[ii + 1].conj().T)

            # ===> convert individual numpy arrays to qforte Tensors
            qf_ff_eigenvalues = qforte.Tensor(
                shape=np.shape(ff_eigenvalues), 
                name='first_factorization_eigenvalues')
            
            qf_one_body_squares = qforte.Tensor(
                shape=np.shape(one_body_squares), 
                name='one_body_squares')
            
            # NOTE(may want to check this later)
            qf_one_body_ints = qforte.Tensor(
                shape=np.shape(p4_mo_oeis), 
                name='one_body_ints')
            
            qf_one_body_correction = qforte.Tensor(
                shape=np.shape(one_body_correction[::2, ::2]), 
                name='one_body_correction')
            
            qf_ff_eigenvalues.fill_from_nparray(
                ff_eigenvalues.ravel(), 
                np.shape(ff_eigenvalues))
            
            qf_one_body_squares.fill_from_nparray(
                one_body_squares.ravel(), 
                np.shape(one_body_squares))
            
            qf_one_body_ints.fill_from_nparray(
                p4_mo_oeis.ravel(), 
                np.shape(p4_mo_oeis))
            
            qf_one_body_correction.fill_from_nparray(
                one_body_correction[::2, ::2].ravel(), 
                np.shape(one_body_correction[::2, ::2]))
            
            # ===> convert lists of numpy arrays to lists of qforte Tensors

            qf_scaled_density_density_matrices = []

            for l in range(len(scaled_density_density_matrices)):
            
                qf_scaled_density_density_mat = qforte.Tensor(
                    shape=np.shape(scaled_density_density_matrices[l]), 
                    name=f'scaled_density_density_matrices_{l}')
                
                qf_scaled_density_density_mat.fill_from_nparray(
                scaled_density_density_matrices[l].ravel(), 
                np.shape(scaled_density_density_matrices[l]))

                qf_scaled_density_density_matrices.append(
                    qf_scaled_density_density_mat
                )
            

            qf_basis_change_matrices = []

            for l in range(len(basis_change_matrices)):
                qf_basis_change_mat = qforte.Tensor(
                    shape=np.shape(basis_change_matrices[l]), 
                    name=f'basis_change_matrices_{l}')
                
                qf_basis_change_mat.fill_from_nparray(
                    basis_change_matrices[l].ravel(), 
                    np.shape(basis_change_matrices[l]))
            
                qf_basis_change_matrices.append(
                    qf_basis_change_mat
                )

            
            qf_trotter_basis_change_matrices = []
            for l in range(len(trotter_basis_change_matrices)):

                qf_trotter_basis_change_mat = qforte.Tensor(
                    shape=np.shape(trotter_basis_change_matrices[l]), 
                    name=f'trotter_basis_change_matrices_{l}')
                
                
                qf_trotter_basis_change_mat.fill_from_nparray(
                    trotter_basis_change_matrices[l].ravel(), 
                    np.shape(trotter_basis_change_matrices[l]))
                
                qf_trotter_basis_change_matrices.append(
                    qf_trotter_basis_change_mat
                )
            
            # build df_hamiltonain object

            # print(f"type(qf_scaled_density_density_matrices): {type(qf_scaled_density_density_matrices)}")
            # print(f"type(qf_scaled_density_density_matrices[0]): {type(qf_scaled_density_density_matrices[0])}")

            # print(f"type(qf_basis_change_matrices): {type(qf_basis_change_matrices)}")
            # print(f"type(qf_basis_change_matrices[0]): {type(qf_basis_change_matrices[0])}")

            # print(f"type(qf_trotter_basis_change_matrices): {type(qf_trotter_basis_change_matrices)}")
            # print(f"type(qf_trotter_basis_change_matrices[0]): {type(qf_trotter_basis_change_matrices[0])}")

            qforte_mol._df_ham = qforte.DFHamiltonian(
                nel=nel,
                norb=nmo,
                eigenvalues = qf_ff_eigenvalues,
                one_body_squares = qf_one_body_squares,
                one_body_ints = qf_one_body_ints,
                one_body_correction = qf_one_body_correction,
                scaled_density_density_matrices = qf_scaled_density_density_matrices,
                basis_change_matrices = qf_basis_change_matrices,
                trotter_basis_change_matrices = qf_trotter_basis_change_matrices
            )

    # Order Psi4 to delete its temporary files.
    psi4.core.clean()

    return qforte_mol

def create_pyscf_mol(**kwargs):
    """Builds a qforte Molecule object directly from a pyscf calculation.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    kwargs.setdefault('symmetry', 'c1')
    kwargs.setdefault('charge', 0)
    kwargs.setdefault('multiplicity', 1)
    kwargs.setdefault('nroots_fci', 1)

    mol_geometry = kwargs['mol_geometry']
    basis = kwargs['basis']
    multiplicity = kwargs['multiplicity']
    charge = kwargs['charge']
    nroots_fci = kwargs['nroots_fci']

    qforte_mol = Molecule(
        mol_geometry = mol_geometry,
        basis = basis,
        multiplicity = multiplicity,
        charge = charge,
        nroots_fci = nroots_fci)

    if not use_pyscf:
        raise ImportError("PySCF was not imported correctely.")

    # By default, the number of frozen orbitals is set to zero
    kwargs.setdefault('num_frozen_docc', 0)
    kwargs.setdefault('num_frozen_uocc', 0)

    # run_scf is not read, because we always run SCF to get a wavefunction object.
    kwargs.setdefault('run_mp2', False)
    kwargs.setdefault('run_ccsd', False)
    kwargs.setdefault('run_cisd', False)
    kwargs.setdefault('run_fci', False)

    kwargs.setdefault('nroots_fci', 1)
    nroots_fci = kwargs['nroots_fci']

    # ===> Run PySCF End <=== #

    pyscf_geom_str = ""

    for geom_line in mol_geometry:
        pyscf_geom_str += f"\n{geom_line[0]}  {geom_line[1][0]:+12.12f}  {geom_line[1][1]:+12.12f}  {geom_line[1][2]:+12.12f}"

    print(' ==> PySCF geometry <==')
    print('-------------------------')
    print(pyscf_geom_str)

    # Determine the spin (number of unpaired electrons)
    spin = multiplicity - 1

    # Create the molecule object
    pyscf_mol = mol = gto.Mole()
    pyscf_mol.build(
        atom=pyscf_geom_str,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry = True,
        symmetry_subgroup = kwargs.get('symmetry', 'c1')
    )

    # Determine the reference type

    if multiplicity != 1:
        raise ValueError(f"Unsupported multiplicity: {multiplicity}, currently unsupporded")

    mf = scf.RHF(pyscf_mol)

    # Set convergence options
    mf.conv_tol = 1e-8          
    mf.conv_tol_grad = 1e-8     

    # Perform the SCF calculation
    pyscf_Escf = mf.kernel()

    num_frozen_docc = kwargs.get('num_frozen_docc', 0)
    num_frozen_uocc = kwargs.get('num_frozen_uocc', 0)

    if (kwargs.get('use_avas', False)):
        if(num_frozen_docc != 0 or num_frozen_uocc != 0):
            raise ValueError(f"Presently can't use avas and freeze orbitals simultaniously.")
        if not kwargs.get('run_fci', False):
            raise ValueError(f"Presently must run casci to use avas.")
        if kwargs.get('symmetry', 'c1') != 'c1':
            raise ValueError(f"Presently use c1 symmetry with avas.")

    # Perform additional computations if requested
    if kwargs.get('run_mp2', False):
        
        nmo = mf.mo_coeff.shape[1]
        nocc = mol.nelectron // 2

        # Build list of frozen orbitals
        frozen = []
        if num_frozen_docc > 0:
            frozen += list(range(num_frozen_docc))  # Freeze core orbitals
        if num_frozen_uocc > 0:
            frozen += list(range(nmo - num_frozen_uocc, nmo))  # Freeze virtual orbitals

        if frozen:
            mymp = mp.MP2(mf).set(frozen=frozen)
        else:
            mymp = mp.MP2(mf)
        mp2_energy, _ = mymp.kernel()

        # Store the MP2 energy
        qforte_mol = lambda: None  # Placeholder for your data structure
        qforte_mol.mp2_energy = mp2_energy

    if kwargs.get('run_fci', False):
        if(num_frozen_docc or num_frozen_uocc):
            # Total number of orbitals
            nmo = mf.mo_coeff.shape[1]

            # Total number of electrons
            nelec = mf.mol.nelectron

            # Number of active orbitals
            ncas = nmo - num_frozen_docc - num_frozen_uocc

            # Number of active electrons (assuming a closed-shell system)
            nelecas = nelec - 2 * num_frozen_docc

            if nelecas < 0 or ncas <= 0:
                raise ValueError("Invalid number of frozen orbitals .")

            # Initialize the CASCI object
            casci = mcscf.CASCI(mf, ncas, nelecas)

            # Freeze the core orbitals
            casci.frozen = num_frozen_docc  

            # Run the CASCI calculation
            casci_output = casci.kernel()

            # Store the FCI energy
            qforte_mol.fci_energy = casci_output[0]

        elif (kwargs.get('use_avas', False)):
            # **Second Conditional**: Run CASCI using AVAS to select active space
            # Define the list of atoms or atomic orbitals to include in the active space
            avas_atoms_or_orbitals = kwargs.get('avas_atoms_or_orbitals', [])
            avas_threshold = kwargs.get('avas_threshold', 0.2)  # Default threshold

            if not avas_atoms_or_orbitals:
                raise ValueError("AVAS is enabled, but no atoms or orbitals are specified for the active space.")

            # Run AVAS to obtain the active space
            avas_obj = avas.AVAS(mf, avas_atoms_or_orbitals, avas_threshold)

            avas_obj.kernel()

            # Number of active orbitals
            ncas = avas_obj.ncas  

            # Number of active electrons
            nelecas = avas_obj.nelecas  

            # Reordered MO coefficients with AVAS active space
            C_avas = avas_obj.mo_coeff  
            nmo = C_avas.shape[0]

            if nelecas is None:
                raise ValueError("AVAS couldn't find any active space electrons with avas_atoms_or_orbitals provided")

            F = mf.get_fock()

            # Transform the Fock matrix into the AVAS MO basis
            F_avas = C_avas.T @ F @ C_avas

            # Get number of core orbitals
            ncore = nmo - ncas

            # Extract just the active block of the Fock matrix
            F_active = F_avas[ncore:ncore+ncas, ncore:ncore+ncas]

            # Diagonalize the active-space Fock submatrix
            mo_energy_active, _ = np.linalg.eigh(F_active)
            
            print('\n\n')
            print("------------------------------------")
            print("       ==> AVAS Settings <=== ")
            print("------------------------------------")
            print(f"  Basis set:               {kwargs['basis']}")
            print(f"  n electrons:             {mol.nelectron}")
            print(f"  n molecular orbs:        {nmo}")
            print(f"  AVAS atoms and orbs:     {avas_atoms_or_orbitals}")
            print(f"  ncas electrons:          {nelecas}")
            print(f"  ncas orbitals:           {ncas}")
            print('\n\n')
            
            # Set up CASCI with the AVAS active space
            casci = mcscf.CASCI(mf, ncas, nelecas)
            casci.mo_coeff = C_avas

            # Run the CASCI calculation
            result = casci.kernel()

            #NOTE(Nick): result stores five elements, including the FCI vector
            casci_total_energy = result[0]

            # Store the CASCI total energy
            qforte_mol.fci_energy = casci_total_energy

            # ==> Obtain the one- and two-electron integrals in the active space <==

            # Get the effective one-electron integrals in the active space, including core contributions
            h1eff, ecore = casci.get_h1eff()

            frozen_core_energy = ecore - mol.energy_nuc()

            # Get the active MO coefficients
            ncore = casci.ncore  # Number of core orbitals (integer)
            C_cas = casci.mo_coeff[:, ncore:ncore + ncas]

            # Transform the AO integrals to the active MO basis
            eri_cas = ao2mo.kernel(mf.mol, C_cas, compact=False)
            eri_cas = eri_cas.reshape(ncas, ncas, ncas, ncas)

            mo_oeis = h1eff
            mo_teis = eri_cas

        else:
            cisolver = fci.FCI(mf)

            cisolver.nroots = nroots_fci

            fci_energies, _ = cisolver.kernel()
            qforte_mol.fci_energy = fci_energies[0]
            qforte_mol.fci_energy_list = fci_energies

            if(nroots_fci > 1):
                print('\n  FCI Eigenstate Energies')
                print('======================================:')
                for i, Ei in enumerate(fci_energies):
                    print(f"  i: {i}  Ei:       {Ei:+10.10f}")


    # Retrieve the number of frozen core and virtual orbitals
    pyscf_Enuc_ref = mol.energy_nuc()
    frozen_core = kwargs.get('num_frozen_docc', 0)
    frozen_virtual = kwargs.get('num_frozen_uocc', 0)

    if kwargs.get('use_avas', False):

        # Get symmetry information, should only be c1 if using avas
        point_group = mol.groupname.lower()
        S = mol.intor('int1e_ovlp')
        irreps_pyscf = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, C_cas, S)
        irreps_qforte = qforte.irreps_of_point_groups(point_group)
        irrep_to_index = {irrep : idx for idx, irrep in enumerate(irreps_qforte)}

        # Build the list of orbitals with their energies and irrep indices
        # print(f"Pyscf irreps   {irreps_pyscf}")
        # print(f"mo_energy_avas {mo_energy_active}")

        orbitals = []
        for i in range(len(mo_energy_active)):
            orbital_energy = mo_energy_active[i]
            irrep_name = irreps_pyscf[i]
            irrep_idx = irrep_to_index[irrep_name]
            orbitals.append([orbital_energy, irrep_idx])

        # Don't sort with avas
        # orbitals.sort()

        # Extract orbital energies and irrep indices
        hf_orbital_energies = []
        orb_irreps_to_int = []
        for row in orbitals:
            hf_orbital_energies.append(row[0])
            orb_irreps_to_int.append(row[1])

        orb_irreps = [irreps_qforte[i] for i in orb_irreps_to_int]


    else:
        # Get the MO coefficients
        C = mf.mo_coeff

        # Get the scalar variables (nuclear repulsion energy)
        pyscf_Enuc_ref = mol.energy_nuc()

        # Get one-electron integrals and transform to MO basis
        h_core = mf.get_hcore()
        mo_oeis = C.T @ h_core @ C  # Transformed one-electron integrals

        # Get two-electron integrals and transform to MO basis
        nmo = C.shape[1]
        # AO to MO transformation of two-electron integrals
        mo_teis = ao2mo.kernel(mol, C, compact=False).reshape(nmo, nmo, nmo, nmo)

        # Get the number of alpha and beta electrons
        nalpha, nbeta = mol.nelec
        nel = nalpha + nbeta

        # Retrieve the number of frozen core and virtual orbitals
        frozen_core = kwargs.get('num_frozen_docc', 0)
        frozen_virtual = kwargs.get('num_frozen_uocc', 0)

        # Get symmetry information
        point_group = mol.groupname.lower()
        S = mol.intor('int1e_ovlp')
        irreps_pyscf = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, C, S)
        irreps_qforte = qforte.irreps_of_point_groups(point_group)
        irrep_to_index = {irrep : idx for idx, irrep in enumerate(irreps_qforte)}

        # Build the list of orbitals with their energies and irrep indices
        orbitals = []
        for i in range(len(mf.mo_energy)):
            orbital_energy = mf.mo_energy[i]
            irrep_name = irreps_pyscf[i]
            irrep_idx = irrep_to_index[irrep_name]
            orbitals.append([orbital_energy, irrep_idx])

        # Sort orbitals by energy
        orbitals.sort()

        # Extract orbital energies and irrep indices
        hf_orbital_energies = []
        orb_irreps_to_int = []
        for row in orbitals:
            hf_orbital_energies.append(row[0])
            orb_irreps_to_int.append(row[1])

        # print('\n\n')
        # print(f"Pyscf orbitals {orbitals}")
        # print(f"Pyscf point_group {point_group}")
        # print(f"Pyscf orb_irreps_to_int {orb_irreps_to_int}")
        # print(f"Pyscf irreps {irreps_qforte}")
        # print('\n\n')

        orb_irreps = [irreps_qforte[i] for i in orb_irreps_to_int]

    # need a conditioanl for using avas
    if (kwargs.get('use_avas', False)):
        num_active_orbitals = ncas
        num_active_electrons = nelecas
        Hsq = build_sq_hamiltonian(
            pyscf_Enuc_ref + frozen_core_energy, 
            mo_oeis, 
            mo_teis,
            ncas,
            0,
            0
            )
    else:
        num_active_orbitals = nmo - frozen_core - frozen_virtual
        num_active_electrons = nel - 2 * frozen_core

        # Compute the frozen core energy
        frozen_core_energy = 0.0
        if frozen_core > 0:
            
            # Sum over frozen core orbitals
            for i in range(frozen_core):
                frozen_core_energy += 2 * mo_oeis[i, i]

            for i in range(frozen_core):
                for j in range(frozen_core):
                    frozen_core_energy += 2 * mo_teis[i, i, j, j] - mo_teis[i, j, j, i]

            # Adjust the one-electron integrals to account for frozen core orbitals
            for p in range(frozen_core, nmo - frozen_virtual):
                for q in range(frozen_core, nmo - frozen_virtual):
                    for i in range(frozen_core):
                        mo_oeis[p, q] += 2 * mo_teis[p, q, i, i] - mo_teis[p, i, i, q]

        Hsq = build_sq_hamiltonian(
            pyscf_Enuc_ref + frozen_core_energy, 
            mo_oeis, 
            mo_teis,
            nmo,
            frozen_core,
            frozen_virtual
            )
        

    hf_reference = [1] * num_active_electrons + [0] * (2 * num_active_orbitals - num_active_electrons)

    # Set attributes
    qforte_mol.nuclear_repulsion_energy = pyscf_Enuc_ref
    qforte_mol.hf_energy = pyscf_Escf
    qforte_mol.hf_reference = hf_reference
    qforte_mol.sq_hamiltonian = Hsq
    if kwargs['build_qb_ham']:
        qforte_mol.hamiltonian = Hsq.jw_transform()
    else:
        Hsq.simplify()
        # qforte_mol.hamiltonian = None
        qforte_mol.hamiltonian = qforte.QubitOperator()

    qforte_mol.point_group = [point_group, irreps_qforte]
    qforte_mol.orb_irreps = orb_irreps
    qforte_mol.orb_irreps_to_int = orb_irreps_to_int
    qforte_mol.hf_orbital_energies = hf_orbital_energies
    qforte_mol.frozen_core = frozen_core
    qforte_mol.frozen_virtual = frozen_virtual
    qforte_mol.frozen_core_energy = frozen_core_energy

    if kwargs['build_df_ham']:
        if not kwargs['store_mo_ints']:
            raise ValueError("store_mo_ints must be True if you want to build_df_ham")
        else:
            pyscf_mo_oeis = copy.deepcopy(mo_oeis)
            pyscf_mo_teis = copy.deepcopy(mo_teis)

    if kwargs['store_mo_ints_np']:
        # Resize mo_oeis and mo_teis if there are frozen orbitals...
        if(frozen_core or frozen_virtual):
            # raise ValueError("This doesn't work..")
            start = frozen_core
            end = nmo - frozen_virtual
            mo_oeis = copy.deepcopy(mo_oeis[start:end, start:end])
            mo_teis = copy.deepcopy(mo_teis[start:end, start:end, start:end, start:end])

        # keep ordering consistant with openfermion eri tensors
        mo_teis_trans_np = copy.deepcopy(np.asarray(mo_teis.transpose(0, 2, 3, 1), order='C'))

        # save numpy copies
        qforte_mol.mo_oeis_np = copy.deepcopy(mo_oeis)
        qforte_mol.mo_teis_np = copy.deepcopy(mo_teis_trans_np)

    if kwargs['store_mo_ints']:

        # Resize mo_oeis and mo_teis if there are frozen orbitals...
        if(frozen_core or frozen_virtual):
            # raise ValueError("This doesn't work..")
            start = frozen_core
            end = nmo - frozen_virtual
            mo_oeis = copy.deepcopy(mo_oeis[start:end, start:end])
            mo_teis = copy.deepcopy(mo_teis[start:end, start:end, start:end, start:end])

        # keep ordering consistant with openfermion eri tensors
        mo_teis = np.asarray(mo_teis.transpose(0, 2, 3, 1), order='C')

        # need restricted version
        h2e_rest = copy.deepcopy(np.einsum("ijlk", -0.5 * mo_teis))

        # additoinal manipulation
        h1e = copy.deepcopy(mo_oeis)
        h2e = np.moveaxis(copy.deepcopy(h2e_rest), 1, 2) * (-1.0)
        h1e -= np.einsum('ikkj->ij', h2e)

        # just going to precumpute the einseum (for now)
        h2e_einsum = copy.deepcopy(h2e + np.einsum('ijkl->klij', h2e))

        # allocate qf tensors
        qf_mo_oeis = qforte.Tensor(shape=np.shape(h1e), name='mo_oeis')
        qf_mo_teis = qforte.Tensor(shape=np.shape(h2e), name='mo_teis')
        qf_mo_teis_einsum = qforte.Tensor(shape=np.shape(h2e_einsum), name='mo_teis_einsum')
        
        # fill qf tensors
        qf_mo_oeis.fill_from_nparray(h1e.ravel(), np.shape(h1e))
        qf_mo_teis.fill_from_nparray(h2e.ravel(), np.shape(h2e)) 
        qf_mo_teis_einsum.fill_from_nparray(
            h2e_einsum.ravel(), 
            np.shape(h2e_einsum)) 

        qforte_mol.mo_oeis = qf_mo_oeis
        qforte_mol.mo_teis = qf_mo_teis
        qforte_mol.mo_teis_einsum = qf_mo_teis_einsum

        # TODO(Nick), If we want better controll over this, it there shuld be a molecule member function that
        # builds the df_ham from the stored mo_oeis and mo_teis rather than building it when pyscf 
        # is initially run!
        if kwargs['build_df_ham']:
            if not kwargs['use_avas']:
                raise NotImplementedError("WARNING: Building DF Hamiltonain using pyscf without AVAS is not tested")
            # keep ordering consistant with openfermion eri tensors
            pyscf_mo_teis = np.asarray(pyscf_mo_teis.transpose(0, 2, 3, 1), order='C')

            # do first factorization from integrals
            ff_eigenvalues, one_body_squares, one_body_correction = first_factorization(
                tei = pyscf_mo_teis,
                lmax=None, # change if we want 
                spin_basis=False,
                threshold=kwargs['df_icut']) # may be very important to play with
            
            # do second factorization based on integrals and first factorizaiotn
            scaled_density_density_matrices, basis_change_matrices = second_factorization(
                ff_eigenvalues, 
                one_body_squares)
        
            trotter_basis_change_matrices = [
                # basis_change_matrices[0] @ expm(-1.0j * (p4_mo_oeis + one_body_correction[::2, ::2]))
                np.zeros(shape=(nmo,nmo))
            ]

            for ii in range(len(basis_change_matrices) - 1):

                trotter_basis_change_matrices.append(
                    basis_change_matrices[ii + 1] @ basis_change_matrices[ii].conj().T)
            
            trotter_basis_change_matrices.append(basis_change_matrices[ii + 1].conj().T)

            qf_ff_eigenvalues = qforte.Tensor(
                shape=np.shape(ff_eigenvalues), 
                name='first_factorization_eigenvalues')
            
            qf_one_body_squares = qforte.Tensor(
                shape=np.shape(one_body_squares), 
                name='one_body_squares')
            
            qf_one_body_ints = qforte.Tensor(
                shape=np.shape(pyscf_mo_oeis), 
                name='one_body_ints')
            
            qf_one_body_correction = qforte.Tensor(
                shape=np.shape(one_body_correction[::2, ::2]), 
                name='one_body_correction')
            
            qf_ff_eigenvalues.fill_from_nparray(
                ff_eigenvalues.ravel(), 
                np.shape(ff_eigenvalues))
            
            qf_one_body_squares.fill_from_nparray(
                one_body_squares.ravel(), 
                np.shape(one_body_squares))
            
            qf_one_body_ints.fill_from_nparray(
                pyscf_mo_oeis.ravel(), 
                np.shape(pyscf_mo_oeis))
            
            qf_one_body_correction.fill_from_nparray(
                one_body_correction[::2, ::2].ravel(), 
                np.shape(one_body_correction[::2, ::2]))
            
            # ===> convert lists of numpy arrays to lists of qforte Tensors

            qf_scaled_density_density_matrices = []

            for l in range(len(scaled_density_density_matrices)):
            
                qf_scaled_density_density_mat = qforte.Tensor(
                    shape=np.shape(scaled_density_density_matrices[l]), 
                    name=f'scaled_density_density_matrices_{l}')
                
                qf_scaled_density_density_mat.fill_from_nparray(
                scaled_density_density_matrices[l].ravel(), 
                np.shape(scaled_density_density_matrices[l]))

                qf_scaled_density_density_matrices.append(
                    qf_scaled_density_density_mat
                )
            

            qf_basis_change_matrices = []

            for l in range(len(basis_change_matrices)):
                qf_basis_change_mat = qforte.Tensor(
                    shape=np.shape(basis_change_matrices[l]), 
                    name=f'basis_change_matrices_{l}')
                
                qf_basis_change_mat.fill_from_nparray(
                    basis_change_matrices[l].ravel(), 
                    np.shape(basis_change_matrices[l]))
            
                qf_basis_change_matrices.append(
                    qf_basis_change_mat
                )

            
            qf_trotter_basis_change_matrices = []
            for l in range(len(trotter_basis_change_matrices)):

                qf_trotter_basis_change_mat = qforte.Tensor(
                    shape=np.shape(trotter_basis_change_matrices[l]), 
                    name=f'trotter_basis_change_matrices_{l}')
                
                
                qf_trotter_basis_change_mat.fill_from_nparray(
                    trotter_basis_change_matrices[l].ravel(), 
                    np.shape(trotter_basis_change_matrices[l]))
                
                qf_trotter_basis_change_matrices.append(
                    qf_trotter_basis_change_mat
                )

            qforte_mol._df_ham = qforte.DFHamiltonian(
                nel=num_active_electrons,
                norb=num_active_orbitals,
                eigenvalues = qf_ff_eigenvalues,
                one_body_squares = qf_one_body_squares,
                one_body_ints = qf_one_body_ints,
                one_body_correction = qf_one_body_correction,
                scaled_density_density_matrices = qf_scaled_density_density_matrices,
                basis_change_matrices = qf_basis_change_matrices,
                trotter_basis_change_matrices = qf_trotter_basis_change_matrices
            )

    return qforte_mol


def create_external_mol(**kwargs):
    """Builds a qforte Molecule object from an external json file containing
    the one and two electron integrals and numbers of alpha/beta electrons.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    qforte_mol = Molecule(multiplicity = kwargs['multiplicity'],
                                charge = kwargs['charge'],
                                filename = kwargs['filename'])

    # open json file
    with open(kwargs["filename"]) as f:
        external_data = json.load(f)

    # build sq hamiltonian
    qforte_sq_hamiltonian = qforte.SQOperator()
    qforte_sq_hamiltonian.add(external_data['scalar_energy']['data'], [], [])

    for p, q, h_pq in external_data['oei']['data']:
        qforte_sq_hamiltonian.add(h_pq, [p], [q])

    for p, q, r, s, h_pqrs in external_data['tei']['data']:
        qforte_sq_hamiltonian.add(h_pqrs/4.0, [p,q], [s,r]) # only works in C1 symmetry

    hf_reference = [0 for i in range(external_data['nso']['data'])]
    for n in range(external_data['na']['data'] + external_data['nb']['data']):
        hf_reference[n] = 1

    qforte_mol.point_group = ['C1', 'A']
    qforte_mol.orb_irreps = ['A'] * external_data['nso']['data']
    qforte_mol.orb_irreps_to_int = [0] * external_data['nso']['data']

    qforte_mol.hf_reference = hf_reference

    qforte_mol.sq_hamiltonian = qforte_sq_hamiltonian

    qforte_mol.hamiltonian = qforte_sq_hamiltonian.jw_transform()

    return qforte_mol

def build_sq_hamiltonian(
        zero_body_energy, 
        mo_oeis, 
        mo_teis,
        nmo,
        frozen_core,
        frozen_virtual
        ):
    
    # Build second quantized Hamiltonian
    Hsq = qforte.SQOperator()
    Hsq.add(zero_body_energy, [], [])
    for i in range(frozen_core, nmo - frozen_virtual):
        ia = (i - frozen_core)*2
        ib = (i - frozen_core)*2 + 1
        for j in range(frozen_core, nmo - frozen_virtual):
            ja = (j - frozen_core)*2
            jb = (j - frozen_core)*2 + 1

            Hsq.add(mo_oeis[i,j], [ia], [ja])
            Hsq.add(mo_oeis[i,j], [ib], [jb])

            for k in range(frozen_core, nmo - frozen_virtual):
                ka = (k - frozen_core)*2
                kb = (k - frozen_core)*2 + 1
                for l in range(frozen_core, nmo - frozen_virtual):
                    la = (l - frozen_core)*2
                    lb = (l - frozen_core)*2 + 1

                    if(ia!=jb and kb != la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, jb], [kb, la] ) # abba
                    if(ib!=ja and ka!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, ja], [ka, lb] ) # baab

                    if(ia!=ja and ka!=la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, ja], [ka, la] ) # aaaa
                    if(ib!=jb and kb!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, jb], [kb, lb] ) # bbbb

    return Hsq