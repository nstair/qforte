import qforte as qf
from math import sqrt

BeH2 = [('Be', (0., 0., 0.)), 
        ('H', (0., 0., 1.334)),
        ('H', (0., 0., -1.334))]

mol = qf.system_factory(build_type='psi4', mol_geometry=BeH2, basis='sto-6g', run_fci=1)

ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nel:       {nel}")
print(f" norb:      {norb}")

val = 1.0 / sqrt(2.0)
 
fci_temp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

alpha_ex = qf.SQOperator()
alpha_ex.add_term(1.0, [6], [4])

beta_ex = qf.SQOperator()
beta_ex.add_term(1.0, [7], [5])

fci_temp.hartree_fock()
fci_temp.apply_sqop(alpha_ex)
a_ind = fci_temp.get_nonzero_idxs()[0]

fci_temp.hartree_fock()
fci_temp.apply_sqop(beta_ex)
b_ind = fci_temp.get_nonzero_idxs()[0]

fci_temp.zero_state()
fci_temp.set_element(a_ind, val)
fci_temp.set_element(b_ind, val)

print(fci_temp.get_state_deep())