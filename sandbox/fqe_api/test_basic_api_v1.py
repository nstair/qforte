import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    # ('H', (0., 0., 3.0)), 
    # ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
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
    # build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

fqe_comp1 = qf.FQEComputer(nel=nel, sz=sz, norb=norb)

fci_comp1.hartree_fock()
fqe_comp1.hartree_fock()

# Test state getter

# C1 = fci_comp1.get_state()
# C2 = fqe_comp1.get_state()

C1 = fci_comp1.get_state_deep()
C2 = fqe_comp1.get_state_deep()

print(f"Type C1: {type(C1)}")
print(f"Type C2: {type(C2)}")

# print("\n ===> Printing returned Tensor/nparray from get_state() <===")
# print(C1)
# print("\n numpy version")
# print(C2)
# print("\n")

# Test printing functions (via __str__ method)
print(fci_comp1)
print(fqe_comp1)
# print(fci_comp1.str(print_complex=True))
# print(fqe_comp1.str(print_complex=True))

# Test diff
# fci_comp1.get_tensor_diff(C2)
print("\n")
print(f" Diff (fci - fqe): {fqe_comp1.get_tensor_diff(C1)}")
print("\n")

# Test getter and setter
# fci_01 = fci_comp1.get_element([0,1])
fqe_01 = fqe_comp1.get_element([0,1])

print("\n")
# print(f" fci_01: {fci_01}")
print(f" fqe_01: {fqe_01}")
print("\n")

# fci_01 = fci_comp1.get_element([0,1])
fci_comp1.set_element([0,1], 1.0)
fqe_comp1.set_element([0,1], 1.0)


print(fci_comp1)
print(fqe_comp1)


# Test scale








