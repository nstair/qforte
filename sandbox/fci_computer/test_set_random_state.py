import qforte as qf
import numpy as np

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# random_coeffs = np.random.randn(fci_comp.get_state_deep().size())

shape1 = fci_comp.get_state_deep().shape()

rand_arr = np.random.rand(*shape1)

norm = np.linalg.norm(rand_arr)

normalized_coeffs = rand_arr / norm

# nc = normalized_coeffs.tolist()
T1 = qf.Tensor(shape=shape1, name='steve')
for i in range(shape1[0]):
    for j in range(shape1[1]):
        T1.set([i,j], rand_arr[i,j])

fci_comp.set_state(T1)
fci_comp2.hartree_fock()


print(fci_comp2.get_state_deep())
print(fci_comp.get_state_deep())
