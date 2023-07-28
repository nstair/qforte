import qforte as qf

nel = 2
sz = 0
norb = 2

num_qbits = norb * 2

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

normal_comp = qf.Computer(num_qbits, print_threshold=-1)

print(f"size of regular computer: {len(normal_comp.get_coeff_vec())}")
print(f"size of fci computer:     {fci_comp.get_state().size()}")

print(normal_comp)
print(fci_comp.str(print_data=True))

