import psi4

psi4.set_memory("1 GB")
psi4.core.set_output_file("output.dat", False)

h4 = psi4.geometry("""
0 1
H  0.0   0.0   0.0
H  1.0   0.0   0.0
H  2.0   0.0   0.0
H  3.0   0.0   0.0
symmetry c1
""")

# --- Global Psi4 options ---
psi4.set_options({
    'basis': 'sto-3g',
    'reference': 'rhf',
    'scf_type': 'pk'   # A direct (PK) SCF for small systems; not strictly required
})

# --- SCF (RHF) calculation ---
e_scf = psi4.energy('scf')
print(f"RHF energy: {e_scf:16.8f} Eh")

psi4.set_options({
    "NUM_ROOTS": 4,      # Request 5 roots
    "CALC_S_SQUARED": True,
    # "detci__analysis": True, # Print extra analysis in output
    # "detci__maxiter": 50     # Increase if needed for convergence
})

# --- CIS calculation ---
# The 'cis' keyword automatically uses the RHF orbitals from the prior SCF calculation.
e_cis = psi4.energy('ci1')
print(f"CIS energy: {e_cis:16.8f} Eh")