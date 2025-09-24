import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
# mol = qf.system_factory(
#     build_type='psi4', 
#     mol_geometry=geom, 
#     basis='sto-3g', 
#     run_fci=1)

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    run_fci=0,
    build_qb_ham = False,
    store_mo_ints=1,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print("\n")
print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
print("\n")

timer = qf.local_timer()

fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fqe_comp1 = qf.FQEComputer(nel=nel, sz=sz, norb=norb)

fci_comp1.hartree_fock()
fqe_comp1.hartree_fock()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

time = 0.5
r = 3
order = 2

N = 4

ah = False
adj = True

print("\n Time Evo Settings")
print("===========================")
print(f" time:      {time}")
print(f" r:         {r}")
print(f" order:     {order}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

app_sqop = True
app_tens = True
app_trot = True


# ===> apply sqop <====

if(app_sqop):
    timer.reset()
    fci_comp1.apply_sqop(sqham)
    timer.record("FCI apply sqop")

    timer.reset()
    fqe_comp1.apply_sqop(sqham)
    timer.record("FQE apply sqop")

    # print(fci_comp1)
    # print(fqe_comp1)

    Cfci = fci_comp1.get_state_deep()
    print(f"\n |dC| apply sqop: {fqe_comp1.get_tensor_diff(Cfci)} \n")




# ===> apply tensor <====

if(app_tens):
    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()

    timer.reset()
    fci_comp1.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb)
    timer.record('FCI apply tensor')

    timer.reset()
    fqe_comp1.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis_np, 
        mol.mo_teis_np)
    timer.record('FQE apply tensor')

    Cfci = fci_comp1.get_state_deep()
    print(f"\n |dC| apply tensor: {fqe_comp1.get_tensor_diff(Cfci)} \n")

    # print(fci_comp1)
    # print(fqe_comp1)




# ===> evovle pool trotter <====

if(app_trot):
    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()

    timer.reset()

    fci_comp1.evolve_pool_trotter(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FCI Trotter step V1")

    Cfci1 = fci_comp1.get_state_deep()

    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()

    timer.reset()

    fci_comp1.evolve_pool_trotter_v2(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FCI Trotter step V2")

    timer.reset()

    fqe_comp1.evolve_pool_trotter(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FQE Trotter step")

    Cfci2 = fci_comp1.get_state_deep()
    print(f"\n |dC1| evolve sqop: {fqe_comp1.get_tensor_diff(Cfci1)} \n")
    print(f"\n |dC2| evolve sqop: {fqe_comp1.get_tensor_diff(Cfci2)} \n")


    # ===> evovle pool basic <====

    ah = True
    adj = False

    print("\n Time Evo Settings")
    print("===========================")
    print(f" time:      {1.0}")
    print(f" r:         {1}")
    print(f" order:     {1}")
    print(f" antiherm:  {ah}")
    print(f" adjoint:   {adj}")
    print("\n")


    sd_pool = qf.SQOpPool()
    sd_pool.set_orb_spaces(ref)
    sd_pool.fill_pool("SD")
    # print(sd_pool)

    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()


    timer.reset()

    fci_comp1.evolve_pool_trotter_basic_v2(
        sd_pool,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FCI Trotter basic")

    timer.reset()

    fqe_comp1.evolve_pool_trotter_basic(
        sd_pool,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FQE Trotter basic")

    Cfci = fci_comp1.get_state_deep()
    print(f"\n |dC| evolve sqop basic: {fqe_comp1.get_tensor_diff(Cfci)} \n")






print("\n\n")
print(timer)

