import qforte as qf

print("\n Initial FCIcomp Stuff")
print("===========================")
nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputerThrust(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock_cpu()

print(fci_comp.str(print_data=True))

print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()
# sqop.add_term(2.0, [5], [1])
# sqop.add_term(2.0, [1], [5])

sqop.add_term(2.0, [4], [0])
sqop.add_term(2.0, [0], [4])
print(sqop)

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqop)

print("hermitian_pairs")
print(hermitian_pairs)

time = 0.1
r = 1
order = 1

print("\n Final FCIcomp Stuff")
print("===========================")

fci_comp.to_gpu()
fci_comp.evolve_pool_trotter_gpu(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=False,
        adjoint=False)

print(fci_comp)