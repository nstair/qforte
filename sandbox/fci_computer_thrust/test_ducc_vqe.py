import qforte as qf
r = 1.0

geom = [
    ('H', (0., 0., 0.0*r)), 
    ('H', (0., 0., 1.0*r)),
    ('H', (0., 0., 2.0*r)),
    ('H', (0., 0., 3.0*r)),
    ('H', (0., 0., 4.0*r)), 
    ('H', (0., 0., 5.0*r)),
    ('H', (0., 0., 6.0*r)),
    ('H', (0., 0., 7.0*r)),
    ('H', (0., 0., 8.0*r)),
    ('H', (0., 0., 9.0*r)),
    ('H', (0., 0.,10.0*r)),
    ('H', (0., 0.,11.0*r)),
    # ('H', (0., 0.,12.0*r)),
    # ('H', (0., 0.,13.0*r)),
    ]

# geom = [
#     ('N', (0., 0., 0.0*r)), 
#     ('N', (0., 0., 1.0*r)),
#     ]

timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    symmetry='d2h',
    basis='sto-3g',
    build_qb_ham=True,
    store_mo_ints=True,
    store_mo_ints_np=True,
    run_fci=0)
timer.record("mol build")


timer.reset()
alg_fci_gpu = qf.UCCNVQE(
    mol,
    apply_ham_as_tensor=True,
    computer_type = 'fci_gpu',
    verbose=False,
    optimizer="jacobi"
    )
timer.record("alg setup fci")


timer.reset()
alg_fci_gpu.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    opt_maxiter=20,
    use_analytic_grad=True
    )
timer.record("run alg fci")

timer.reset()
alg_fqe = qf.UCCNVQE(
    mol,
    apply_ham_as_tensor=True,
    computer_type = 'fqe',
    verbose=False,
    optimizer="jacobi"
)

timer.record("alg setup fqe")

Eo_fci_comp = alg_fci_gpu.get_gs_energy()


timer.reset()
alg_fqe.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    opt_maxiter=20,
    use_analytic_grad=True
    )
timer.record("run alg fqe")

Eo_fqe_comp = alg_fqe.get_gs_energy()

print("\n Check Final Energy \n")
print("===========================")
print(f' Efci_comp:  {Eo_fci_comp:+12.10f}')
print(f' Efqe_comp:  {Eo_fqe_comp:+12.10f}')
print(f' E diff:     {Eo_fci_comp - Eo_fqe_comp:+12.10f}')


# print(f' Efci:    {mol.fci_energy:+12.10f}')
# print(f' Edif:    {alg_fci._Egs - mol.fci_energy:+12.10f}')

print("\n Total Script Time \n")
print(timer)

# timings pre fused dot_sqop

#                 ==> GPU Profiling <==
# -----------------------------------------------------------
# Total GPU time:                                 37.229837 s

# GPU call breakdown (sorted by time):
#   apply_sqop_gpu                              12.962960 s  ( 34.8%)
#   get_state_deep                              11.663313 s  ( 31.3%)
#   apply_sqop_evolution_gpu                     4.716158 s  ( 12.7%)
#   get_exp_val_tensor_gpu                       2.684057 s  (  7.2%)
#   apply_tensor_spat_012bdy_gpu                 2.458341 s  (  6.6%)
#   vector_dot                                   1.270026 s  (  3.4%)
#   evolve_pool_trotter_basic_gpu                0.813873 s  (  2.2%)
#   set_state_gpu                                0.659007 s  (  1.8%)
#   hartree_fock_gpu                             0.002102 s  (  0.0%)


# timings post fused dot_sqop

#                 ==> GPU Profiling <==
# -----------------------------------------------------------
# Total GPU time:                                 13.699297 s

# GPU call breakdown (sorted by time):
#   apply_sqop_evolution_gpu                     3.894527 s  ( 28.4%)
#   dot_sqop_gpu                                 3.695037 s  ( 27.0%)
#   get_exp_val_tensor_gpu                       2.862963 s  ( 20.9%)
#   apply_tensor_spat_012bdy_gpu                 2.457315 s  ( 17.9%)
#   evolve_pool_trotter_basic_gpu                0.786814 s  (  5.7%)
#   hartree_fock_gpu                             0.001999 s  (  0.0%)
#   set_state_from_other_gpu                     0.000641 s  (  0.0%)