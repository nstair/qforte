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
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
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
    store_mo_ints=True,
    store_mo_ints_np=True,
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


fci_comp1.hartree_fock()


sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

time = 0.1
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
app_exact_evo = True
app_trot = True

sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("SD")

sdt_pool = qf.SQOpPool()
sdt_pool.set_orb_spaces(ref)
sdt_pool.fill_pool("SDT")

sdtq_pool = qf.SQOpPool()
sdtq_pool.set_orb_spaces(ref)
sdtq_pool.fill_pool("SDTQ")

pool_dict = {
    "SD": sd_pool,
    "SDT": sdt_pool,
    "SDTQ": sdtq_pool   
}


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

# ===> evovle pool trotter <====

for pool_name, pool in pool_dict.items():
    fci_comp1.hartree_fock()

    timer.reset()

    fci_comp1.evolve_pool_trotter_basic(
        pool,
        antiherm=True,
        adjoint=False)
    
    fci_comp1.apply_tensor_spat_012bdy(
        # mol.nuclear_repulsion_energy, 
        0.0, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb)
    
    fci_comp1.evolve_pool_trotter_basic(
        pool,
        antiherm=True,
        adjoint=True)

    timer.record(f"FCI Trotter basic inplace")

    Cfci = fci_comp1.get_state_deep()

    # print("\n\n")
    # print(Cfci)

    Cshape = Cfci.shape()

    Nfci = np.prod(Cshape)

    non_zero_idxs = Cfci.get_nonzero_tidxs()

    N_non_zero = len(non_zero_idxs)

    print(f"\n ==> Pool Evolution Results {pool_name} <==")
    print(f" N param:    {len(pool.terms())}")
    print(f" N non-zero: {N_non_zero}")
    print(f" N total:    {Nfci}")


print("\n\n")
print(timer)
