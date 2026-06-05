import qforte as qf


def run_grad_fd_test():
    # geom = [
    # # ('Be', (0., 0., 1.00)), 
    # ('H', (0., 0., 1.00)),
    # ('H', (0., 0., 2.00)),
    # ('H', (0., 0., 3.00)),
    # ('H', (0., 0., 4.00)),
    # # ('H', (0., 0., 5.00)),
    # # ('H', (0., 0., 6.00)),
    # # ('H', (0., 0., 7.00)),
    # # ('H', (0., 0., 8.00)),
    # # ('H', (0., 0., 9.00)),
    # # ('H', (0., 0., 10.00))
    # ]

    # geom = [
    #     ('Be', (0., 0., 2.00)), 
    #     ('H', (0., 0., 1.00)),
    #     ('H', (0., 0., 3.00)),
    #     ]

    # geom = [
    #     ('N', (0., 0., 1.00)),
    #     ('N', (0., 0., 2.00)),
    #     ]

    geom = [
        ('Be', (0., 0., 0.00)),
        ('Be', (0., 0., 2.00)),
    ]

    timer = qf.local_timer()

    timer.reset()

    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        basis='sto-3g',
        symmetry='d2h',
        run_fci=1)

    timer.record("Psi4 Setup")

    kmax = 5

    pool_str = f'{kmax}-UpCCGSD'
    # pool_str = "GSD"
    pool_str = "SD"


    tUCC = qf.UCCNVQE(
        mol,
        computer_type = 'fci',
        apply_ham_as_tensor = True
        )

    timer.reset()

    tUCC.run(
        opt_thresh=1.0e-6, 
        pool_type=pool_str,
        optimizer='L-BFGS-B',
        opt_maxiter=20,
            )

    timer.record("dUCCSD FCI")

    print(timer)
                
    print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


    


def main():
    
    run_grad_fd_test()


if __name__ == "__main__":
    main()

#                     ==> UCCN-VQE summary <==
# -----------------------------------------------------------
# Final UCCN-VQE Energy:                      -28.7795906288
# Number of operators in pool:                  104
# Final number of amplitudes in ansatz:         104
# Total number of Hamiltonian measurements:     13
# Total number of commutator measurements:      0
# Number of classical parameters used:          104
# Number of non-zero parameters used:           104
# Number of CNOT gates in deepest circuit:      0
# Number of Pauli term measurements:            6083363
# Number of grad vector evaluations:            13
# Number of individual grad evaluations:        1352


#                 ==> FCI Profiling <==
# -----------------------------------------------------------
# Total FCI time:                                  3.388390 s

# FCI call breakdown (sorted by time):
#   vector_dot                                   0.958411 s  ( 28.3%)
#   apply_sqop_evolution                         0.616538 s  ( 18.2%)
#   get_exp_val_tensor                           0.498830 s  ( 14.7%)
#   evolve_pool_trotter_basic                    0.470533 s  ( 13.9%)
#   apply_tensor_spat_012bdy                     0.423759 s  ( 12.5%)
#   apply_sqop                                   0.270027 s  (  8.0%)
#   get_state_deep                               0.092860 s  (  2.7%)
#   set_state                                    0.056625 s  (  1.7%)
#   hartree_fock                                 0.000805 s  (  0.0%)



#      Process name         Time (s)          Percent
#     =============        =========          =======
#         fill_pool           0.0083             0.23
# initialize_ansatz           0.0000             0.00
#             solve           3.5997            99.77

#        Total Time           3.6080           100.00

# Process name    Time (s)     Percent
# =============   =========     =======
#   Psi4 Setup      3.2885       47.68
#   dUCCSD FCI      3.6084       52.32

#   Total Time      6.8969      100.00



#  Efci:   -28.7896241174
    