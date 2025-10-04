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
    build_qb_ham=True,
    store_mo_ints=True,
    store_mo_ints_np=True,
    run_fci=0)
timer.record("mol build")


timer.reset()
alg_fci = qf.UCCNVQE(
    mol,
    apply_ham_as_tensor=True,
    computer_type = 'fci',
    verbose=False,
    )
timer.record("alg setup fci")


timer.reset()
alg_fci.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    opt_maxiter=20,
    )
timer.record("run alg fci")

timer.reset()
alg_fqe = qf.UCCNVQE(
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
    opt_maxiter=20,
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



# [-7.50366436e-12 -7.50364961e-12  2.16257890e-01 -1.91240471e-02
#   1.91240471e-02  1.73240159e-01  1.58647276e-01 -3.43658435e-12
#   3.48515661e-12  9.31062290e-02  4.89586588e-02 -6.69349620e-02
#   1.60041191e-01 -5.52856847e-02  1.04244343e-01 -3.43657741e-12
#   3.48515661e-12  1.60041191e-01 -6.69349620e-02  1.04244343e-01
#  -5.52856847e-02  9.31062290e-02  4.89586588e-02  1.72462990e-01
#   3.98445050e-02 -3.98445050e-02  1.71128142e-01  1.10999628e-01
#   4.96339081e-12  6.62672700e-02  2.18091817e-02 -7.99979007e-02
#   1.46265171e-01 -4.50720921e-02  5.92025210e-02  5.41651517e-02
#   3.23543733e-02 -4.89857149e-02  1.03150867e-01 -6.78167910e-02
#   1.00171164e-01  4.96339081e-12 -2.18091817e-02  1.46265171e-01
#  -7.99979007e-02  4.50720921e-02 -5.92025210e-02  6.62672700e-02
#   1.03150867e-01 -4.89857149e-02  1.00171164e-01 -6.78167910e-02
#   5.41651517e-02  3.23543733e-02  1.55449021e-01 -5.60401315e-02
#   5.60401315e-02  1.04871936e-01  1.38251065e-01]