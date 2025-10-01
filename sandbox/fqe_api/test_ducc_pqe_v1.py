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
    # ('H', (0., 0.,10.0*r)),
    # ('H', (0., 0.,11.0*r))
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
    build_qb_ham=False,
    store_mo_ints=True,
    store_mo_ints_np=True,
    run_fci=0)
timer.record("mol build")


timer.reset()
alg_fci = qf.UCCNPQE(
    mol,
    apply_ham_as_tensor=True,
    computer_type = 'fci',
    verbose=False)
timer.record("alg setup fci")


timer.reset()
alg_fci.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    )
timer.record("run alg fci")

timer.reset()
alg_fqe = qf.UCCNPQE(
    mol,
    apply_ham_as_tensor=True,
    computer_type = 'fqe',
    verbose=False)
timer.record("alg setup fqe")

Eo_fci_comp = alg_fci.get_gs_energy()


timer.reset()
alg_fqe.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
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


