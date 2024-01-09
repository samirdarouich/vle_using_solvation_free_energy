import numpy as np
from typing import List, Dict
from moleculegraph import molecule


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