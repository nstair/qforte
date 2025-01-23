"""
Classes for system information, e.g., molecule or model.
"""

# TODO: Documentation needs to be fixed, attributes should be listed below
#       as opposed to arguments for __init__() (Nick).

class System(object):
    """Class for a generic quantum many-body system."""

    @property
    def fci_energy(self):
        return self._fci_energy

    @fci_energy.setter
    def fci_energy(self, fci_energy):
        self._fci_energy = fci_energy

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian_operator):
        self._hamiltonian = hamiltonian_operator

    @property
    def sq_hamiltonian(self):
        return self._sq_hamiltonian

    @sq_hamiltonian.setter
    def sq_hamiltonian(self, sq_hamiltonian_operator):
        self._sq_hamiltonian = sq_hamiltonian_operator

    @property
    def hf_reference(self):
        return self._hf_reference

    @hf_reference.setter
    def hf_reference(self, hf_reference):
        self._hf_reference = hf_reference

class Molecule(System):
    """Class for storing moleucular information. Should be instatiated using using
    a MolAdapter and populated by calling MolAdapter.run(**kwargs).


    Atributes
    ---------
    _mol_geometry : list of tuples
        Gives coordinates of each atom in Angstroms. Example format is
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.50))].

    _basis : string
        Gives the basis set to be used. Default is 'sto-3g'.

    _multiplicity : int
        Gives the targeted spin multiplicity of the molecular system.

    _charge : int
        Gives the targeted net charge of the molecular system (controls number of
        electrons to be considered).

    _filename : optional, string
        Specifies the name of the output files generated by Psi4.

    point_group: list of two elements
        It contains the information about the point group and its irreps.
        point_group[0] is a string holding the name of the point group.
        point_group[1] is a list that holds the irreps of the group in the Cotton ordering.
        Example: point_group = ['c2v', ['A1', 'A2', 'B1', 'B2']]

    ###### In the following attributes, the spatial orbitals are in ascending energy order. ######

    orb_irreps : list of strings
        It contains the information about the irrep of each spatial orbital of the system.
        In the case of H2/STO-3G, for example: orb_irreps = ['Ag', 'B1u']

    orb_irreps_to_int : list of integers
        It contains the same information as orb_irreps with each irrep mapped to an integer, using Cotton ordering.
        In the case of H2/STO-3G, for example: orb_irreps_to_int = [0, 5]

    hf_orbital_energies : list of floats
        It contains the RHF orbital energies (in hartree) of the system, as computed by Psi4.
        In the case of H2/STO-3G, r = 3.0 angs, for example: hf_orbital_energies = [-0.1805392218304829, 0.018071329565966215]

    ###### Attributes pertaining to frozen-orbital approximations ######

    frozen_core : integer
        Number of lowest-energy frozen core orbitals.

    frozen_virtual : integer
        Number of highest-energy frozen virtual orbitals.

    frozen_core_energy: float
        The contribution to the Hartree-Fock energy associated with the frozen core orbitals

    """

    def __init__(self, mol_geometry=None, basis='sto-3g', multiplicity=1, charge=0, nroots_fci=1,
                 filename=""):
        """Initialize a qforte molecule object.

        Arguments
        ---------
        mol_geometry : tuple of tuples
            Gives the coordinates of each atom in the moleucle.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom.

        basis : string
            Gives the basis set. Default is 'sto-3g'.

        charge : int
            Gives the total molecular charge. Defaults to 0.

        multiplicity : int
            Gives the spin multiplicity.

        filename : optional, string
            Gives name of output files generated by Psi4.
        """

        self.geometry = mol_geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.nroots_fci = nroots_fci
        self.filename = filename

    @property
    def nroots_fci(self):
        return self._nroots_fci

    @nroots_fci.setter
    def nroots_fci(self, nroots_fci):
        self._nroots_fci = nroots_fci

    @property
    def hf_energy(self):
        return self._hf_energy

    @hf_energy.setter
    def hf_energy(self, hf_energy):
        self._hf_energy = hf_energy

    @property
    def mp2_energy(self):
        return self._mp2_energy

    @mp2_energy.setter
    def mp2_energy(self, mp2_energy):
        self._mp2_energy = mp2_energy

    @property
    def cisd_energy(self):
        return self._cisd_energy

    @cisd_energy.setter
    def cisd_energy(self, cisd_energy):
        self._cisd_energy = cisd_energy

    @property
    def ccsd_energy(self):
        return self._ccsd_energy

    @ccsd_energy.setter
    def ccsd_energy(self, ccsd_energy):
        self._ccsd_energy = ccsd_energy

    @property
    def nuclear_repulsion_energy(self):
        return self._nuclear_repulsion_energy

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, nuclear_repulsion_energy):
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    @property
    def point_group(self):
        return self._point_group

    @point_group.setter
    def point_group(self, point_group):
        self._point_group = point_group

    @property
    def orb_irreps(self):
        return self._orb_irreps

    @orb_irreps.setter
    def orb_irreps(self, orb_irreps):
        self._orb_irreps = orb_irreps

    @property
    def orb_irreps_to_int(self):
        return self._orb_irreps_to_int

    @orb_irreps_to_int.setter
    def orb_irreps_to_int(self, orb_irreps_to_int):
        self._orb_irreps_to_int = orb_irreps_to_int

    @property
    def hf_orbital_energies(self):
        return self._hf_orbital_energies
    
    @property
    def mo_oeis(self):
        return self._mo_oeis
    
    @mo_oeis.setter
    def mo_oeis(self, mo_oeis):
        self._mo_oeis = mo_oeis
    
    @property
    def mo_teis(self):
        return self._mo_teis
    
    @mo_teis.setter
    def mo_teis(self, mo_teis):
        self._mo_teis = mo_teis

    @property
    def mo_teis_einsum(self):
        return self._mo_teis_einsum
    
    @mo_teis_einsum.setter
    def mo_teis_einsum(self, mo_teis_einsum):
        self._mo_teis_einsum = mo_teis_einsum

    @property
    def df_ham(self):
        return self._df_ham
    
    @df_ham.setter
    def df_ham(self, df_ham):
        self._df_ham = df_ham

    @hf_orbital_energies.setter
    def hf_orbital_energies(self, hf_orbital_energies):
        self._hf_orbital_energies = hf_orbital_energies

    @property
    def frozen_core(self):
        return self._frozen_core

    @frozen_core.setter
    def frozen_core(self, frozen_core):
        self._frozen_core = frozen_core

    @property
    def frozen_virtual(self):
        return self._frozen_virtual

    @frozen_virtual.setter
    def frozen_virtual(self, frozen_virtual):
        self._frozen_virtual = frozen_virtual

    @property
    def frozen_core_energy(self):
        return self._frozen_core_energy

    @frozen_core_energy.setter
    def frozen_core_energy(self, frozen_core_energy):
        self._frozen_core_energy = frozen_core_energy
