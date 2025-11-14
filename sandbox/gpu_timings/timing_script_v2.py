import qforte as qf

import numpy as np

# Define the reference and geometry lists.

BeH2_geom = [

    ('H', (0., 0., 1.0)),

    ('Be', (0., 0., 2.0)),

    ('H', (0., 0., 3.0)),

    ]

H20_geom = [

    ('H', (0., 0., 1.0)),

    ('O', (0., 0., 2.0)),

    ('H', (0., 0., 3.0)),

    ]

N2_geom = [

    ('N', (0., 0., 1.0)),

    ('N', (0., 0., 2.0)),

    ]

LiH_geom = [

    ('Li', (0., 0., 1.0)),

    ('H', (0., 0., 2.0)),

    ]

H2_geom = [

    ('H', (0., 0., 1.0)),

    ('H', (0., 0., 2.0)),

    ]
 
H4_geom = [

    ('H', (0., 0., 1.0)),

    ('H', (0., 0., 2.0)),

    ('H', (0., 0., 3.0)),

    ('H', (0., 0., 4.0)),

    ]

H6_geom = [

    ('H', (0., 0., 1.0)),

    ('H', (0., 0., 2.0)),

    ('H', (0., 0., 3.0)),

    ('H', (0., 0., 4.0)),

    ('H', (0., 0., 5.0)),

    ('H', (0., 0., 6.0)),

    ]

H8_geom = [

    ('H', (0., 0., 1.0)),

    ('H', (0., 0., 2.0)),

    ('H', (0., 0., 3.0)),

    ('H', (0., 0., 4.0)),

    ('H', (0., 0., 5.0)),

    ('H', (0., 0., 6.0)),

    ('H', (0., 0., 7.0)),

    ('H', (0., 0., 8.0)),

    ]

H10_geom = [

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

    ]

H12_geom = [

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

    ('H', (0., 0.,11.0)),

    ('H', (0., 0.,12.0))

    ]

H14_geom = [

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

    ('H', (0., 0.,11.0)),

    ('H', (0., 0.,12.0)),

    ('H', (0., 0.,13.0)),

    ('H', (0., 0.,14.0))

    ]

geom_list = [

    H2_geom,

    H4_geom,

    H6_geom,

    H8_geom,

    BeH2_geom,

    LiH_geom,

    N2_geom,

    H20_geom,

    H10_geom,

    H12_geom,

    # H14_geom

    ]

name_list = [

    'H2',

    'H4',

    'H6',

    'H8',

    'BeH2',

    'LiH',

    'N2',

    'H2O',

    'H10',

    'H12',

    # 'H14'

    ]

norb_lst = []

gpu_kernel_times = []
cpu_specific_times = []

timer = qf.local_timer()

for geom, name in zip(geom_list, name_list):

    # timer.reset()

    # Get the molecule object that now contains both the fermionic and qubit Hamiltonians.

    mol = qf.system_factory(

        build_type='psi4',

        mol_geometry=geom,

        basis='sto-3g',

        build_qb_ham = False,

        run_fci=0)

    # timer.record('Run Psi4 and Initialize')

    ref = mol.hf_reference

    nel = sum(ref)

    sz = 0

    norb = int(len(ref) / 2)

    norb_lst.append(norb)
 
    # if(norb < 6):

    #     print(f" nqbit:     {norb*2}")

    #     print(f" nel:       {nel}")

    fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

    fc1.hartree_fock()
   
    timer.reset()

    # fc1.apply_tensor_spat_012bdy(

    #     mol.nuclear_repulsion_energy,

    #     mol.mo_oeis,

    #     mol.mo_teis,

    #     mol.mo_teis_einsum,

    #     norb

    # )

    fc1.apply_sqop(mol.sq_hamiltonian)

    timer.record(f'fci  computer {name}')

    E1 = fc1.get_hf_dot()

    # fock computer stuff

    # fk1 = qf.Computer(norb * 2)

    # Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')

    # fk1.apply_circuit(Uhf)

    fcg1 = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)

    fcg1.hartree_fock()

    fcg1.to_gpu()

    timer.reset()


    fcg1.apply_sqop(mol.sq_hamiltonian)

    timer.record(f'fcigpu computer {name}')
    
    fcg1.to_cpu()

    times = timer.get_timings()

    acc_times_gpu = fcg1.get_acc_timer().get_acc_timings()
    acc_times_cpu = fc1.get_acc_timer().get_acc_timings()

    E1 = fc1.get_hf_dot() 
    E2 = fcg1.get_hf_dot() 



    # print("acc_times: \n")
    # print(acc_times_gpu)

    gpu_kernel_time = acc_times_gpu["calling gpu function"]
    cpu_specific_time = acc_times_cpu["cpu function"]

    gpu_kernel_times.append(gpu_kernel_time)
    cpu_specific_times.append(cpu_specific_time)

    N = int(len(times) / 2)
    # N2 = int(len(acc_times) / 2)
 


    # print(N)

    print("\n\n Timing")

    print("======================================================") 

    for n in range(N):

        ifc = 2*n
        ifgpu = 2*n + 1

        _, tval_fc = times[ifc]
        _, tval_fk = times[ifgpu]

        line = f"{name_list[n]:8}  {tval_fk:e}   {tval_fc:e}  {tval_fc/tval_fk:e}  {norb_lst[n]*2:6}  {gpu_kernel_times[n]:e}   {cpu_specific_times[n]:e}   {cpu_specific_times[n]/gpu_kernel_times[n]:e}"

        print(line)

    print("\n")

    print("\n Energetics")
    print("======================================================")
    # print(f" Efci:               {mol.fci_energy}")
    print(f" Ehf:                {mol.hf_energy}")
    print(f" Enr:                {mol.nuclear_repulsion_energy}")
    # print(f" Eelec:              {mol.hf_energy - mol.nuclear_repulsion_energy}")
    print(f" E1 (from cpu):   {E1}")
    print(f" E2 (from gpu):   {E2}")     


print("")

print(timer)

 

    # print("\n Energetics")

    # print("======================================================")

    # print(f" Efci:               {mol.fci_energy}")

    # print(f" Ehf:                {mol.hf_energy}")

    # print(f" Enr:                {mol.nuclear_repulsion_energy}")

    # print(f" Eelec:              {mol.hf_energy - mol.nuclear_repulsion_energy}")

    # print(f" E1 (from tensor):   {E1}")

 