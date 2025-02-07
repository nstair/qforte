# This is to test that the CIS construction is reversible


import qforte as qf
import numpy as np


geom = [
    ('H', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    # ('H', (0., 0., 5.00)), 
    # ('H', (0., 0., 6.00)),
    # ('H', (0., 0., 7.00)),
    # ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    symmetry = 'c1',
    run_fci=1)


apply_ham_as_tensor = True


alg_fci = qf.CIS(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor=apply_ham_as_tensor,
    )

timer = qf.local_timer()
timer.reset()

alg_fci.run(
    target_root=3,
    diagonalize_each_step=False,
    low_memory=False
)

timer.record(f"Run CIS FCI")

ft = qf.FCIComputer(
    alg_fci._nel, 
    alg_fci._2_spin, 
    alg_fci._norb)

ft.hartree_fock()

rand = True
if(rand):
    random_array = np.random.rand(ft.get_state().shape()[0], ft.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))
    # random = np.ones((ft.get_state().shape()[0], ft.get_state().shape()[1]))
    Crand = qf.Tensor(ft.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    ft.set_state(Crand)
    # print(ft.str(print_data=True))

IJs, IJt, angles = alg_fci.get_cis_unitary_parameters()

Ecis0 = ft.get_exp_val(alg_fci._sq_ham)
print(f"Befor Ecis from Unitary: {Ecis0}")

ft.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        False
    )

EcisA = ft.get_exp_val(alg_fci._sq_ham)
print(f"Final Ecis from Unitary: {EcisA}")

ft.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        True
    )

EcisB = ft.get_exp_val(alg_fci._sq_ham)
print(f"Final Ecis from U^dag U: {EcisB}")




# print(f"\n\nApply ham as tensor: {apply_ham_as_tensor}")
# print(timer)



# Eold = alg_fci.get_gs_energy()

# print('\n\n')
# print(f' Efci:     {mol.fci_energy:+12.10f}')
# print(f' Eold:     {Eold:+12.10f}')
# print(f' dEold:    {Eold-mol.fci_energy:+12.10f}')


#LGTM!

