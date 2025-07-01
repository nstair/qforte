import qforte as qf

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)
# fci_comp.hartree_fock()

fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
# fci_comp2.hartree_fock()

"""
print(fci_comp.str(print_data=True))
print(fci_comp2.str(print_data=True))


print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()
sqop.add_term(3.0, [5], [1])
sqop.add_term(3.0, [1], [5])

sqop.add_term(2.0, [4], [0])
sqop.add_term(2.0, [0], [4])
print(sqop)




print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)
print(fci_comp2)

fci_comp.do_on_gpu()
fci_comp.apply_sqop(sqop)
fci_comp2.apply_sqop(sqop)


print("\n Final FCIcomp Stuff")
print("===========================")
print(fci_comp)
print(fci_comp2)
"""