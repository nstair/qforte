import qforte as qf


geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)),
    ('H', (0., 0., 6.00)),
    ('H', (0., 0., 7.00)),
    ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

geom = [
    ('H', (0., 0., 1.00)),
    ('F', (0., 0., 2.00)),
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

# alg_fock = qf.UCCNVQE(
#     mol,
#     computer_type = 'fock'
#     )

# alg_fock.run(
#     opt_thresh=1.0e-2, 
#     pool_type='GSD',
#     # optimizer='LBFGS'
#     )

# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.UCCNVQE(
    mol,
    computer_type = 'fci_gpu',
    )

alg_fci.run(opt_thresh=1.0e-2, 
            pool_type='SD',
            optimizer='bfgs_qf')
            
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

#                 ==> fci UCCN-VQE summary <==
# -----------------------------------------------------------
# Final UCCN-VQE Energy:                      -4.3060567700
# Final <S^2>:                                0.0000000124
# Number of operators in pool:                  360
# Final number of amplitudes in ansatz:         360
# Total number of Hamiltonian measurements:     23
# Total number of commutator measurements:      0
# Number of classical parameters used:          360
# Number of non-zero parameters used:           184
# Number of CNOT gates in deepest circuit:      36480
# Number of Pauli term measurements:            14002791
# Number of grad vector evaluations:            13
# Number of individual grad evaluations:        4680

#                 ==> UCCN-VQE summary <==
# -----------------------------------------------------------
# Final UCCN-VQE Energy:                      -4.2891134290
# Final <S^2>:                                0.0551869867
# Number of operators in pool:                  288
# Final number of amplitudes in ansatz:         288
# Total number of Hamiltonian measurements:     21
# Total number of commutator measurements:      0
# Number of classical parameters used:          288
# Number of non-zero parameters used:           144
# Number of CNOT gates in deepest circuit:      27648
# Number of Pauli term measurements:            9289557
# Number of grad vector evaluations:            11
# Number of individual grad evaluations:        3168
