import qforte as qf
import numpy as np

# Define the reference and geometry lists.

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
my_hubbard_model = qf.system_factory(
    system_type='model',
    build_type='fermi_hubbard', 
    nel=4,
    nsites=4,
    tunneling_term=4.0, # J
    coulomb_term=2.0,   # U
    basis='sto-3g', 
    run_scf=0,
    run_fci=0)