import numpy as np
from rdkit import Chem
from moleculegraph import molecule
from rdkit.Chem import Descriptors
from typing import List, Tuple, Dict
from feos.pcsaft import PcSaftParameters
from feos.gc_pcsaft import IdentifierOption
from feos.eos import EquationOfState, PhaseDiagram
from feos.si import KELVIN, BAR, PASCAL, KILOGRAM, METER, ANGSTROM


def get_initial_conditions(SAFT_parameter_file: str, molecule_smiles: List[str], compositions: List[float], temperature: float=None, pressure: float=None, 
                           SAFT_binary_file: str="", n_eval_saft: int=51, verbose: bool=False) -> Tuple[List, List, List, List]:
    """
    Function that uses the feos implementation of the PC-SAFT equation of state to determine suitable starting temperatures, pressures and densities for any mixture.

    Args:
        SAFT_parameter_file (str): File containing the PC-SAFT parameters
        molecule_smiles (List[str]): SMILES of each molecule in the mixture
        compositions (List[float]): Compositions that should be extracted of this mixture
        temperature (float, optional): Constant temperature of the mixture. If constant pressure should be used, temperature should equal None. Defaults to None.
        pressure (float, optional): Constant pressure of the mixture. If constant temperature should be used, pressure should equal None. Defaults to None.
        SAFT_binary_file (str, optional): File containing binary PC-SAFT parameters. Defaults to "".
        n_eval_saft (int, optional): Evaluation steps of the PC-SAFT VLE object. Defaults to 51.
        verbose (bool, optional): If the binary parameter should be printed out. Defaults to False.


    Returns:
        temperatures (List): Temperatures of the mixture at the specified compositions.
        pressures (List): Pressures of the mixture at the specified compositions.
        densities (List): Densities of the mixture at the specified compositions.
        activity_coeff (List): Symmetric activity coefficients at the specified compositions. (For each composition it contains gamma_i and gamma_j)
    """
    
    # Get SAFT paramters
    if SAFT_binary_file:
        parameters = PcSaftParameters.from_json( molecule_smiles, SAFT_parameter_file, SAFT_binary_file, search_option = IdentifierOption.Smiles )
    else:
        parameters = PcSaftParameters.from_json( molecule_smiles, SAFT_parameter_file, search_option = IdentifierOption.Smiles )

    if verbose:
        print("Binary parameter: k_ij = %f\n"%parameters.k_ij[0][1])

    # Call the PC-SAFT equation of state class
    eos = EquationOfState.pcsaft( parameters )

    # Get a VLE of either constant temperature or pressure
    key     = temperature * KELVIN if temperature else pressure * BAR
    vle     = PhaseDiagram.binary_vle( eos = eos, temperature_or_pressure = key, npoints = n_eval_saft)

    # Extract temperatures, pressures, and densities at the specified compositions
    saft_compositions = [ np.round( state.liquid.molefracs[0], 3).tolist() for state in vle.states if np.isin(round(state.liquid.molefracs[0],3),compositions)] 
    temperatures      = [ np.round( state.liquid.temperature / KELVIN, 3).tolist() for state in vle.states if np.isin(round(state.liquid.molefracs[0],3),compositions) ]
    pressures         = [ np.round( state.liquid.pressure() / PASCAL * 1e-5, 3).tolist() for state in vle.states if np.isin(round(state.liquid.molefracs[0],3),compositions) ]
    densities         = [ np.round( state.liquid.mass_density() / ( KILOGRAM / METER**3 ), 2).tolist() for state in vle.states if np.isin(round(state.liquid.molefracs[0],3),compositions) ]
    activity_coeff    = [ np.round( np.exp(state.liquid.ln_symmetric_activity_coefficient()), 3).tolist()[0] for state in vle.states if np.isin(round(state.liquid.molefracs[0],3),compositions) ]

    if not saft_compositions == compositions:
        raise KeyError("Not all specified compositions are found in the VLE object. Increase the number of VLE evaluations!\n")
    
    return temperatures, pressures, densities, activity_coeff


def create_composition_matrix(no_components: int, composition: List, precision: int=3) -> List[ List[ List ] ]:
    """
    Function that makes a compositon matrix for an arbitrary mixture.

    Args:
        no_components (int): Number of componets in the mixture.
        composition (List): Composition that should be simulated for each pure component.
        precision (int, optional): Number of decimals for rounding. Defaults to 3.

    Returns:
        composition_matrix List[ List[ List ] ]: Composition matrix. [ [x1=[0.0, ... 1.0], x2, x3, ...], [x1, x2=[0.0, ..., 1.0], x3, ... ], ... ]
    """
    # Create empty composition matrix
    composition_matrix = [ [] for _ in range(no_components) ]

    # Fill the diagonal with the original composition and off-diagonal with complementary composition
    for i in range(no_components):
        for j in range(no_components):
            composition_matrix[i].append( np.round(composition,precision).tolist() if i == j else np.round(1 - np.array(composition), precision).tolist() )

    return composition_matrix


def calculate_molecular_weight(smiles: str):
    """
    Function that uses the SMILE to obtain the molecular weight.

    Args:
        smiles (str): SMILES of a molecule

    Returns:
        float: Molecular weight in g/mol
    """
    if "LJ" or "Mie" in smiles:
        smiles = "[Ar]"
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.MolWt(mol)
    else:
        return None
    
def write_decoupling_ff(mol_list: List[molecule], settings: Dict[str, str | List | Dict], atom_numbers_ges: List[int], nonbonded: List[Dict[str,str]],
                        free_energy_method:str, lambda_vdw: float, lambda_coulomb: float, dlambda: List[float], free_energy_output_files: List[str]  ):
    """

    Args:
        mol_list (List[molecule]): List containing the moleculegraph objects for each molecule.
        settings (Dict[str, str  |  List  |  Dict]): Settings dictionary which will be used to render the LAMMPS input template. 
        atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
        nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: sigma, epsilon, name, m, charge
        free_energy_method (str): Free energy method that is utilized.
        lambda_vdw (float): Coupling lambda used for scaling the van der Waals interactions.
        lambda_coulomb (float): Coupling lambda used for scaling the Coulomb interactions.
        dlambda (List[float]): List containing the perturbation used in LAMMPS to compute the free energy output. Should be very small in case of TI (e.g.: [-0.001, 0.001])
                               or in case of FEP / BAR should be the difference to the previous and subsequent lambda points.
        free_energy_output_files (List[str]): List containing the file names for each computation output.

    Raises:
        KeyError: If the settings dictionary do not contain the subdictionary "style".
    """

    if not "style" in settings.keys():
        raise KeyError("Settings dictionary do not contain the style subdictionary!")
    
    ## Define pair interactions ##

    # Van der Waals
    pair_interactions = []

    # Define the atom numbers of the coupling molecule (this is always component one)
    atom_list_coulped_molecule = np.arange( len(mol_list[0].unique_atom_keys) ) + 1

    for i,iatom in zip(atom_numbers_ges, nonbonded):
        for j,jatom in zip(atom_numbers_ges[i-1:], nonbonded[i-1:]):
            
            # For decoupling the intramolecular interactions of the coupling molecule need to be unaltered,
            # while the intermolecular interactions need to be scaled. This is done using a lambda_ij.
            # Unaltered: lambda_ij = 1.0, coupled: 0.0 < lambda_ij < 1.0, deactiavted: lambda_ij = 0.0
            if all( np.isin( [ i, j ], atom_list_coulped_molecule) ):
                lambda_ij = 1.0 
            elif (i in atom_list_coulped_molecule and not j in atom_list_coulped_molecule) or (not i in atom_list_coulped_molecule and j in atom_list_coulped_molecule):
                lambda_ij = lambda_vdw
            else:
                lambda_ij = None

            name_ij   = "%s  %s"%( iatom["name"], jatom["name"] ) 

            if settings["style"]["mixing"] == "arithmetic": 
                sigma_ij   = ( iatom["sigma"] + jatom["sigma"] ) / 2
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif settings["style"]["mixing"] == "geometric":
                sigma_ij   = np.sqrt( iatom["sigma"] * jatom["sigma"] )
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif settings["style"]["mixing"] == "sixthpower": 
                sigma_ij   = ( 0.5 * ( iatom["sigma"]**6 + jatom["sigma"]**6 ) )**( 1 / 6 ) 
                epsilon_ij = 2 * np.sqrt( iatom["epsilon"] * jatom["epsilon"] ) * iatom["sigma"]**3 * jatom["sigma"]**3 / ( iatom["sigma"]**6 + jatom["sigma"]**6 )

            n_ij  = ( iatom["m"] + jatom["m"] ) / 2
            
            pair_interactions.append( { "i": i, "j": j, "sigma": round( sigma_ij, 4 ) , "epsilon": round( epsilon_ij, 4 ),  "m": n_ij, "name": name_ij, "lambda_vdw": lambda_ij } ) 

        settings["style"]["pairs"] = pair_interactions


        ## Save free energy related settings ##

        settings["free_energy"]                        = {}

        # Save the atom types of the coupled molecule
        settings["free_energy"]["coupled_atom_list"]   = atom_list_coulped_molecule

        # Coulomb
        # If coupling is wanted, the charges of all all atoms in the coupled molecule need to be scaled accordingly (this is done in the template with lambda_coulomb).
        # In case vdW interactions are investiagted, scale the charges to closely 0 (1e-9), and overlay a 2nd Coulomb potential to maintain unaltered intramolecular Coulomb interactions.
        # In case Coulomb interactions are investiaged, scale the charges accordingly, and also overlay the 2nd Coulomb potential.
        settings["free_energy"]["charge_list"]         = [ (i+1,iatom["charge"]) for i,iatom in enumerate( np.array( nonbonded )[ atom_list_coulped_molecule - 1 ] ) ]

        # Define the overlay lambda between all atoms of the coupled molecule (defined in https://doi.org/10.1007/s10822-020-00303-3)
        settings["free_energy"]["lambda_overlay"]      = ( 1 - lambda_coulomb**2 ) / lambda_coulomb**2

        # Define if vdW or Coulomb interactions is decoupled.
        settings["free_energy"]["couple_lambda"]       = lambda_vdw if np.isclose( lambda_coulomb, 0 ) else lambda_coulomb
        settings["free_energy"]["couple_interaction"]  = "vdw" if np.isclose( lambda_coulomb, 0 ) else "coulomb"

        # Define free energy method
        settings["free_energy"]["method"]              = free_energy_method

        # Define the lambda perpurtation that should be performed (backwards and forwards)
        settings["free_energy"]["lambda_perturbation"] = dlambda

        # Define output files for free energy computation
        settings["free_energy"]["output_files"]        = free_energy_output_files