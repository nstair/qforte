import qforte as qf
from qforte import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from matplotlib.backends.backend_pdf import PdfPages
# passes in the correlation term U, runs SRQK with 6 optimized time steps on half filled 6 site hubbard model and returns:
# data: containing step number and energy at that step
# FCI energy (exact hamiltonian ground state)
# dt (array of optimized time steps)
def var_plt(U):  
    nel = 6   # number of electrons
    norb = 6  # number of sites in the hubbard lattive 
    J = 1# tunneling 1 body
    # Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
    Escf = []
    Efci = []
    gam = []
    dat_tot=np.empty((7,7),dtype = int)
    my_hubbard_model = qf.system_factory(
        system_type='model',
        build_type='fermi_hubbard', 
        nel = nel,
        nsites = norb,
        tunneling_term = J, # J
        coulomb_term = U,   # U
        basis='sto-3g', 
        run_scf=0,
        run_fci=0,
        pbc = True)

    # define the algorithm being run on the created hubbard lattice 
    alg = SRQK(
        my_hubbard_model,
        computer_type='fci',
        print_summary_file=True)
    
    # minimizing function to optimize time steps (minimize energy)
    def E_mini(dt=float, *args):
        dts = copy.deepcopy(args[0])
        # dts = args[0]
        dt = float(dt)
        dts.append(dt)
        alg.run(s=len(dts)-1,
        dt = dts,
        target_root=0,
        diagonalize_each_step=True)
        data = np.loadtxt("summary.dat")
        Evals = data[:,1]
        return(Evals[-1])


    Evals=[0]
    dt = [0]
    # while my_hubbard_model.fci_energy+tolerance < Evals[-1]:
    # i+=1
    # running for 7 optimized steps 
    for _ in range(6):
        # Run a single reference QK calculation.
        result = minimize(E_mini, x0=.1, args = (dt,), method='L-BFGS-B')
        opt_val = result.x[0]
        dt.append(opt_val)
        alg.run(s=len(dt)-1,
        dt = dt,
        target_root=0,
        diagonalize_each_step=True)
        data = np.loadtxt("summary.dat")
    return data, my_hubbard_model.fci_energy, dt
def fix_plt(dt,U):
    nel = 6   # number of electrons
    norb = 6  # number of sites in the hubbard lattive 
    J = 1# tunneling 1 body
    # Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
    Escf = []
    Efci = []
    gam = []
    dat_tot=np.empty((7,7),dtype = int)
    my_hubbard_model = qf.system_factory(
        system_type='model',
        build_type='fermi_hubbard', 
        nel = nel,
        nsites = norb,
        tunneling_term = J, # J
        coulomb_term = U,   # U
        basis='sto-3g', 
        run_scf=0,
        run_fci=0,
        pbc = True)

    # define the algorithm being run on the created hubbard lattice 
    alg = SRQK(
        my_hubbard_model,
        computer_type='fci',
        print_summary_file=True)
    
    alg.run(s=6,
        dt = dt,
        target_root=0,
        diagonalize_each_step=True)
    data = np.loadtxt("summary.dat")
    return data, my_hubbard_model.fci_energy, dt

# calling the var_plt function 3 times for correlation values 1,2,3 and plotting them
for U in range(1, 4):
    fig = plt.figure()
    data1, E1, dt1 = var_plt(U)
    data2, E2, dt2 = fix_plt(.1,U)
    data3, E3, dt3 = fix_plt(.01,U)

    plt.plot(data1[:, 2], data1[:, 1] - E1, label='variational')
    plt.plot(data2[:, 2], data2[:, 1] - E2, label=f'dt = {dt2}')
    plt.plot(data3[:, 2], data3[:, 1] - E3, label=f'dt = {dt3}')

    plt.xlabel('N [Subspace Dimension]')
    plt.ylabel('ln(E_QK - E_FCI)')
    plt.yscale('log')
    plt.legend()
    plt.title(f'SRQK Convergence Gamma = {np.round(1/U,2)}')
    
        # Save each figure to a separate PDF file
    file_path = f'/Users/emmettjoee/Documents/Senior Project/Emmett Senior Project/output_U_{U}.pdf'
    pdf_pages = PdfPages(file_path)
    pdf_pages.savefig(fig)
    pdf_pages.close()

plt.close() 
