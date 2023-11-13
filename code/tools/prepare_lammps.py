import numpy as np
import toml
import json
import os
import moleculegraph
import random
from scipy.constants import Avogadro,Boltzmann
from jinja2 import Template

#### Class that creates LAMMPS data and input file ####

class LAMMPS_input():

    def __init__(self,mol_list,ff_path,playmol_ff_path):
        """
        Initilizing LAMMPS input class. Create playmol ff and save system independent force field parameters

        Args:
            mol_list (list): List containing moleculegraph items for both components in the mixture
            ff_path (str): Path to toml file containing used force-field readable format by moleculegraph
        """

        ## Save moleclue graphs of both components class wide ##
        self.mol_list = mol_list

        ## Read in force field toml file ##
        with open(ff_path) as ff_toml_file:
            self.ff = toml.load(ff_toml_file)

        ## Specify force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions)

        # Get (unique) atom types and parameters #
        self.nonbonded = np.array([j for sub in [molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in mol_list] for j in sub])
        # Get (unique) bond types and parameters #
        self.bonds     = [j for sub in [molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in mol_list] for j in sub]
        # Get (unique) angle types and parameters #     
        self.angles    = [j for sub in [molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in mol_list] for j in sub]
        # Get (unique) torsion types and parameters #
        self.torsions  = [j for sub in [molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in mol_list] for j in sub]
        
        if not all( [ all([bool(entry) for entry in self.nonbonded]), all([bool(entry) for entry in self.bonds]), all([bool(entry) for entry in self.angles]), all([bool(entry) for entry in self.torsions]) ] ):
            txt = "nonbonded" if not all([bool(entry) for entry in self.nonbonded]) else "bonds" if not all([bool(entry) for entry in self.bonds]) else "angles" if not all([bool(entry) for entry in self.angles]) else "torsions"
            raise ValueError("Something went wrong during the force field typing for key: %s"%txt)
        
        # Get all atom types not only the unique one #
        self.ff_all    = np.array([j for sub in [molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) for molecule in mol_list] for j in sub])


        ## Define general settings that are not system size dependent ##

        molecule1,molecule2          = mol_list
        self.renderdict              = {}

        # Definitions for atoms in the system #
        self.number_of_atoms         = [mol.atom_number for mol in mol_list]
        self.atoms_running_number    = np.array(list(molecule1.unique_atom_inverse+1)+list(molecule2.unique_atom_inverse+1+len(np.unique(molecule1.unique_atom_inverse))))

        # Definitions for bonds in the system #
        self.number_of_bonds         = len(molecule1.bond_keys)+len(molecule2.bond_keys)
        self.bonds_running_number    = np.array(list(molecule1.unique_bond_inverse+1)+list(molecule2.unique_bond_inverse+1+len(molecule1.unique_bond_keys)))
        self.bond_numbers            = np.array(list(molecule1.bond_list+1)+list(molecule2.bond_list+1))
        self.bond_numbers_ges        = np.unique(self.bonds_running_number)

        self.renderdict["bond_styles"]        = list(np.unique([p["style"] for n,p in zip(self.bond_numbers_ges,self.bonds)]))
        self.renderdict["bond_type_number"]   = len(self.bond_numbers_ges)

        # Definitions for angles in the system #
        self.number_of_angles        = len(molecule1.angle_keys)+len(molecule2.angle_keys)
        self.angles_running_number   = np.array(list(molecule1.unique_angle_inverse+1)+list(molecule2.unique_angle_inverse+1+len(molecule1.unique_angle_keys)))
        self.angle_numbers           = np.array(list(molecule1.angle_list+1)+list(molecule2.angle_list+1))
        self.angle_numbers_ges       = np.unique(self.angles_running_number)

        self.renderdict["angle_styles"]        = list(np.unique([p["style"] for n,p in zip(self.angle_numbers_ges ,self.angles)]))
        self.renderdict["angle_type_number"]   = len(self.angle_numbers_ges)

        # Definitions for torsions in the system #
        self.number_of_torsions      = len(molecule1.torsion_keys)+len(molecule2.torsion_keys)
        self.torsions_running_number = np.array(list(molecule1.unique_torsion_inverse+1)+list(molecule2.unique_torsion_inverse+1+len(molecule1.unique_torsion_keys)))
        self.torsion_numbers         = np.array(list(molecule1.torsion_list+1)+list(molecule2.torsion_list+1))
        self.torsion_numbers_ges     = np.unique(self.torsions_running_number)

        self.renderdict["torsion_styles"]      = list(np.unique([p["style"] for n,p in zip(self.torsion_numbers_ges,self.torsions)]))
        self.renderdict["torsion_type_number"] = len(self.torsion_numbers_ges)


        ## Prepare dictionary for jinja2 template to write force field input for Playmol ##

        renderdict              = {}
        renderdict["nonbonded"] = list(zip([j for sub in [molecule.unique_atom_keys for molecule in mol_list] for j in sub],self.nonbonded))
        renderdict["bonds"]     = list(zip([j for sub in [molecule.unique_bond_names for molecule in mol_list] for j in sub],self.bonds))
        renderdict["angles"]    = list(zip([j for sub in [molecule.unique_angle_names for molecule in mol_list] for j in sub],self.angles))
        renderdict["torsions"]  = list(zip([j for sub in [molecule.unique_torsion_names for molecule in mol_list] for j in sub],self.torsions))
        
        ## Generate force field file for playmol using jinja2 template ##

        if not os.path.exists(playmol_ff_path[:playmol_ff_path.rfind("/")+1]): os.makedirs(playmol_ff_path[:playmol_ff_path.rfind("/")+1])

        with open("../templates/template.playmol") as file_:
            template = Template(file_.read())
        
        rendered = template.render( rd=renderdict )

        with open(playmol_ff_path, "w") as fh:
            fh.write(rendered) 
        
        return


    def data_file_init(self,nmol_list,densitiy,maindir):
        """
        Function that prepares the LAMMPS data file at a given density for a given force field and system

        Args:
            nmol_list (list): List containing the number of molecules per component
            densitiy (float): Mass density of the mixture at that state (kg/m^3)
            maindir (str): Path where notebook is exectued
        """

        ## Variables defined here are used class wide ##
        self.nmol_list      = nmol_list
        self.density        = densitiy
        self.maindir        = maindir
        
        molecule1,molecule2 = self.mol_list

        # Zip objects has to be refreshed for every system since its only possible to loop over them once #

        self.renderdict["bond_paras"]    = zip(self.bond_numbers_ges, self.bonds)
        self.renderdict["angle_paras"]   = zip(self.angle_numbers_ges, self.angles)
        self.renderdict["torsion_paras"] = zip(self.torsion_numbers_ges, self.torsions)
        

        #### System specific settings ####

        ## Definitions for atoms in the system ##

        # Differentiate the last molecule of species 1 from every other 
        # Therefore create new types for last molecule and increase the type number of every atom in species 2 aswell
        # This is just needed for pair potentials to ensure right coupling

        if nmol_list[0] >1:
            self.add_type     = np.max(self.atoms_running_number[molecule1.atom_numbers])
            self.a_list1      = np.arange(len(molecule1.unique_atom_keys))+1
            self.a_list1_add  = np.arange(len(molecule1.unique_atom_keys))+1+self.add_type
            self.a_list2      = np.arange(len(molecule2.unique_atom_keys))+1+self.add_type+len(molecule1.unique_atom_keys)

            self.pair_numbers = list(self.a_list1) + list(self.a_list1_add) + list(self.a_list2)
            self.pair_ff      = list(self.nonbonded[self.a_list1-1])*2  + list(self.nonbonded[self.a_list2-1-self.add_type])

        else:
            self.add_type     = 0
            self.a_list1      = np.arange(len(molecule1.unique_atom_keys))+1
            self.a_list1_add  = np.arange(len(molecule1.unique_atom_keys))+1+self.add_type
            self.a_list2      = np.arange(len(molecule2.unique_atom_keys))+1+self.add_type+len(molecule1.unique_atom_keys)

            self.pair_numbers = list(self.a_list1) + list(self.a_list2)
            self.pair_ff      = list(self.nonbonded)

        self.renderdict["atom_paras"]       = list(zip(self.pair_numbers,self.pair_ff))
        self.renderdict["atom_type_number"] = len(self.nonbonded) + self.add_type

        # Total atoms in system #
        self.total_number_of_atoms    = np.dot(self.number_of_atoms,nmol_list)

        # Total bonds in system #
        self.total_number_of_bonds    = np.dot([len(molecule1.bond_keys),len(molecule2.bond_keys)],nmol_list)
        
        # Total angles in system #
        self.total_number_of_angles   = np.dot([len(molecule1.angle_keys),len(molecule2.angle_keys)],nmol_list)
        
        # Total torsions in system #
        self.total_number_of_torsions = np.dot([len(molecule1.torsion_keys),len(molecule2.torsion_keys)],nmol_list)

        ## Mass, mol, volume and box size of the system ##

        # Molar masses of each species [g/mol] #
        Mol_masses = np.array([np.sum([a["mass"] for a in molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) if not a["name"] == "H"]) for molecule in self.mol_list])

        # Account for mixture density #

        # mole fraction of mixture (== numberfraction) #
        x = np.array(nmol_list)/np.sum(nmol_list)

        # Average molar weight of mixture [g/mol] #
        M_avg = np.dot(x,Mol_masses)

        # Total mole n = N/NA [mol] #
        n = np.sum(nmol_list)/Avogadro

        # Total mass m = n*M [kg] #
        mass = n*M_avg / 1000

        # Compute box volume V=m/rho and with it the box lenght L (in Angstrom)
        # With mass (kg) and rho (kg/m^3 --> convert in g/A^3)

        # Volume = mass / mass_density = mol / mol_density [A^3]

        volume = mass / self.density *10**30

        boxlen = volume**(1/3) / 2

        box = [ -boxlen, boxlen]

        self.renderdict["box_x"] = box
        self.renderdict["box_y"] = box
        self.renderdict["box_z"] = box
        
        return
    
    
    def write_lammps_data(self,xyz_path,data_path):
        """
        Function that generates LAMMPS data file, containing bond, angle and torsion parameters, as well as all the coordinates etc.

        Args:
            xyz_path (str): Path where xyz file ist located for this system
            data_path (str): Path where the LAMMPS data file should be generated
        """

        molecule1,molecule2 = self.mol_list

        atom_count       = 0
        bond_count       = 0
        angle_count      = 0
        torsion_count    = 0
        mol_count        = 0
        mol_count1       = 0
        coordinate_count = 0
        
        lmp_atom_list    = []
        lmp_bond_list    = []
        lmp_angle_list   = []
        lmp_torsion_list = []

        coordinates = moleculegraph.funcs.read_xyz(xyz_path)

        for m,mol in enumerate(self.mol_list):

            # All ff lists are created for both species --> only take part of the list for the specific component 

            if m == 0:
                idx  = mol.atom_numbers
                idx1 = np.arange(len(molecule1.bond_keys))
                idx2 = np.arange(len(molecule1.angle_keys))
                idx3 = np.arange(len(molecule1.torsion_keys))

            elif m == 1:
                idx  = mol.atom_numbers+molecule1.atom_number
                idx1 = np.arange(len(molecule2.bond_keys))+len(molecule1.bond_keys)
                idx2 = np.arange(len(molecule2.angle_keys))+len(molecule1.angle_keys)
                idx3 = np.arange(len(molecule2.torsion_keys))+len(molecule1.torsion_keys)


            ## Now write LAMMPS input for every molecule of each component ##

            for mn in range(self.nmol_list[m]):
                
                # Define atoms

                for atomtype,ff_atom in zip(self.atoms_running_number[idx],self.ff_all[idx]):
                    
                    atom_count +=1
                    
                    # If last molecule of species 1: Add to every atomtype the number of types of species 1
                    if mn == self.nmol_list[0]-1 and m == 0:
                        atomtype += self.add_type
                    # If atoms of species 2, add to there atomtype to account for the extra types introduced (if N(species)==1 no special types are added (add_type =0))
                    elif m==1:
                        atomtype += self.add_type

                    # LAMMPS INPUT: total n° of atom in system, mol n° in System, atomtype, partial charges,coordinates
                    
                    line = [ atom_count, mol_count+1, atomtype, ff_atom["charge"],*coordinates[coordinate_count]["xyz"],
                            coordinates[coordinate_count]["atom"] ]

                    lmp_atom_list.append(line)

                    coordinate_count += 1

                # Define bonds

                for bondtype,bond in zip(self.bonds_running_number[idx1],self.bond_numbers[idx1]):

                    bond_count += 1

                    # The bond counting for 2nd species needs to start after nmol1 * bonds_mol1 for right index of atoms

                    if m == 0:
                        dummy = bond + mol_count * mol.atom_number
                        txt   = [ "A"+str(x) for x in bond ]
                    elif m == 1:
                        dummy = bond + (mol_count-mol_count1) * mol.atom_number + mol_count1*molecule1.atom_number
                        txt   = [ "A"+str(x+molecule1.atom_number) for x in bond ]

                    # LAMMPS INPUT: total n° of bonds in system, bond type, atoms of system with this bond type

                    line = [ bond_count, bondtype, *dummy, *txt]

                    lmp_bond_list.append(line)

                # Define angles

                for angletype,angle in zip(self.angles_running_number[idx2],self.angle_numbers[idx2]):

                    angle_count += 1

                    # The angles counting for 2nd species needs to start after nmol1 * angles_mol1 for right index of atoms

                    if m == 0:
                        dummy = angle + mol_count * mol.atom_number
                        txt   = ["A"+str(x) for x in angle]
                    elif m == 1:
                        dummy = angle + (mol_count-mol_count1) * mol.atom_number + mol_count1*molecule1.atom_number
                        txt   = ["A"+str(x+molecule1.atom_number) for x in angle]

                    # LAMMPS INPUT: total n° of angles in system, angle type, atoms of system with this angle type

                    line = [ angle_count, angletype, *dummy, *txt]

                    lmp_angle_list.append(line)

                # Define torsions

                for torsiontype,torsion in zip(self.torsions_running_number[idx3],self.torsion_numbers[idx3]):

                    torsion_count += 1

                    # The torsion counting for 2nd species needs to start after nmol1 * torsion_mol1 for right index of atoms

                    if m == 0:
                        dummy = torsion + mol_count * mol.atom_number
                        txt   = ["A"+str(x) for x in torsion]
                    elif m == 1:
                        dummy = torsion + (mol_count-mol_count1) * mol.atom_number + mol_count1*molecule1.atom_number
                        txt   = ["A"+str(x+molecule1.atom_number) for x in torsion]

                    dummy = np.flip( dummy ) # (?)
                    torsion = np.flip( torsion)  # (?)

                    # LAMMPS INPUT: total n° of torsions in system, torsion type, atoms of system with this torsion type

                    line = [ torsion_count, torsiontype, *dummy, *txt]

                    lmp_torsion_list.append(line)


                if m == 0: mol_count1 += 1

                mol_count += 1

                
            self.renderdict["atoms"]    = lmp_atom_list
            self.renderdict["bonds"]    = lmp_bond_list
            self.renderdict["angles"]   = lmp_angle_list
            self.renderdict["torsions"] = lmp_torsion_list

            self.renderdict["atom_number"]    = atom_count
            self.renderdict["bond_number"]    = bond_count
            self.renderdict["angle_number"]   = angle_count
            self.renderdict["torsion_number"] = torsion_count
            
        # Write to jinja2 template file to create LAMMPS input file

        if not os.path.exists(data_path[:data_path.rfind("/")+1]): os.makedirs(data_path[:data_path.rfind("/")+1])
        
        with open("../templates/template.lmp") as file_:
            template = Template(file_.read())
            
        rendered = template.render( rd = self.renderdict )

        with open(data_path, "w") as fh:
            fh.write(rendered)

        return
    

    def build_playmol(self,playmol_path,name_list,name,i):
        """
        Function that generates dictionary needed for jinja2 template to create input file for playmol to build the specified system

        Args:
            playmol_path (str): Path to playmol .mol file to build the system
            name_list (list): List containing the names of the both species (used to search for xyz files)
            name (str): Name of the mixture (e.g: mix_comp1_comp2)
            i (int): Enumeration index of the densities 
        """

        molecule1,molecule2 = self.mol_list
        
        moldict   = {}
        mol_names = [j for sub in [molecule.atom_names for molecule in self.mol_list] for j in sub]
        mol_bonds = [j for sub in [molecule1.bond_list+1,molecule2.bond_list+1+molecule1.atom_number] for j in sub]

        atom_numbers       = list( np.arange(molecule1.atom_number+molecule2.atom_number) +1 )
        playmol_atom_names = [j for sub in [[ get_name(mn) for mn in molecule.atom_names] for molecule in self.mol_list] for j in sub]
        playmol_bond_names = [j for sub in [[ [ get_name( molecule.atom_names[mi]) for mi in bl ] for bl in molecule.bond_list]  for molecule in self.mol_list] for j in sub]

        moldict["name"]    = name
        moldict["atoms"]   = list(zip( atom_numbers ,mol_names, playmol_atom_names,[self.ff_all[i]["charge"] for i,a in enumerate(mol_names)]) )
        moldict["bonds"]   = list(zip( mol_bonds, playmol_bond_names))
        moldict["mol"]     = [str(moldict["atoms"][0][2])+str(moldict["atoms"][0][0]),str(moldict["atoms"][molecule1.atom_number][2])+str(moldict["atoms"][molecule1.atom_number][0])]

        # Write playmol input file to build the system with specified number of molecules for each species #

        with open("../templates/template_mix.mol") as file:
                template = Template(file.read())

        # Playmol template needs density in g/cm^3; rho given in kg/m^3 to convert in g/cm^3 divide by 1000 #

        rendered = template.render( rd=moldict,name_list=name_list,name=name,rho=str(self.density/1000),i=str(i),nmols=self.nmol_list,seed=random.randint(1,10**6) )
        
        with open(playmol_path, "w") as fh:
            fh.write(rendered) 

        # Build system with playmol #
        
        os.chdir(playmol_path[:playmol_path.rfind("/")+1])
        console = "~/.local/bin/playmol -i %s"%playmol_path[playmol_path.rfind("/")+1:]
        log = os.system( console )
        os.chdir(self.maindir)
        print(log)
        print("DONE: "+console)

        return 
    
    def input_file_init(self,settings):
        """
        Function that initialize general settings for LAMMPS input file

        Args:
            settings (dict): Settings dictionary for simulation
        """

        ## Define if special bonds for LAMMPS should be used ##
        self.sb_dict       = settings["sb_dict"]
        self.special_bonds = any([any(d_item[1]) for d_item in settings["sb_dict"].items()])
        
        ## Shake algorithm is used if keys are found in FF used in the simulation ##

        # Define used input atom, bond and angle index
        atom_paras  = zip(self.pair_numbers, self.pair_ff)
        bond_paras  = zip(self.bond_numbers_ges, self.bonds)
        angle_paras = zip(self.angle_numbers_ges, self.angles)

        key_at  = [item for sublist in [[a[0] for a in atom_paras if a_key == a[1]["name"]] for a_key in settings["shake"][0]] for item in sublist]
        key_b   = [item for sublist in [[a[0] for a in bond_paras if a_key == a[1]["list"]] for a_key in settings["shake"][1]] for item in sublist]
        key_an  = [item for sublist in [[a[0] for a in angle_paras if a_key == a[1]["list"]] for a_key in settings["shake"][2]] for item in sublist]

        self.s_dict  = {"t":key_at, "b":key_b, "a":key_an}
        self.shake = any( [ key_at, key_b, key_an ] )

        ## Define cut off radius from FF file ##

        if np.ndim(np.squeeze( np.unique( [p["cut"] for p in self.nonbonded] ) )) == 0:
            self.rcut = np.squeeze( np.unique( [p["cut"] for p in self.nonbonded] ) )
        else:
            self.rcut = np.max( np.squeeze( np.unique( [p["cut"] for p in self.nonbonded] ) ) )
            print("More then one cut off radius defined ! This cut off radius is used: %.2f"%self.rcut)

        ## Define it any atom in the system is charged, if not dont use coulomb pair style in simulation ##

        self.uncharged = all( [charge==0 for charge in np.unique([p["charge"] for p in self.nonbonded])] )

        ## Define the used styles for bonds, angles and dihedrals (necessary if several styles are used) ##

        self.style = {}

        bond_styles     = list(np.unique([a["style"] for a in self.bonds]))
        angle_styles    = list(np.unique([a["style"] for a in self.angles]))  
        dihedral_styles = list(np.unique([a["style"] for a in self.torsions]))

        if len(bond_styles) >1: bond_styles = ["hybrid"] + bond_styles
        if len(angle_styles) >1: angle_styles = ["hybrid"] + angle_styles
        if len(dihedral_styles) >1: dihedral_styles = ["hybrid"] + dihedral_styles

        self.style["bond"]     = bond_styles
        self.style["angle"]    = angle_styles
        self.style["dihedral"] = dihedral_styles

        return


    def write_lammps_input(self,t,p,nmol_list,settings,data_path,input_file_name):
        """
        Function that writes LAMMPS input file using a jinja2 template

        Args:
            t (float): Temperature
            p (float): Pressure
            nmol_list (list): List containing the number of molecules per component
            settings (dict): Settings dictionary for simulation
            data_path (str): Relative path to LAMMPS data file
            input_file_name (str): Path to save input file
        """

        ## Define pair interactions. Take care to differentiate bewteen inserted and all other molecules ##

        pair_interactions = []

        if nmol_list[0] >1:
            add_type     = np.max(self.atoms_running_number[self.mol_list[0].atom_numbers])
            a_list1      = np.arange(len(self.mol_list[0].unique_atom_keys))+1
            a_list1_add  = np.arange(len(self.mol_list[0].unique_atom_keys))+1+add_type
            a_list2      = np.arange(len(self.mol_list[1].unique_atom_keys))+1+add_type+len(self.mol_list[0].unique_atom_keys)

            pair_numbers = list(a_list1) + list(a_list1_add) + list(a_list2)
            pair_ff      = list(self.nonbonded[a_list1-1])*2 + list(self.nonbonded[a_list2-1-add_type])

        else:
            add_type     = 0
            a_list1      = []
            a_list1_add  = np.arange(len(self.mol_list[0].unique_atom_keys))+1+add_type
            a_list2      = np.arange(len(self.mol_list[1].unique_atom_keys))+1+add_type+len(self.mol_list[0].unique_atom_keys)

            pair_numbers = list(a_list1_add) + list(a_list2)
            pair_ff      = list(self.nonbonded)

        for i,iatom in zip(pair_numbers, pair_ff):
            for j,jatom in zip(pair_numbers[i-1:], pair_ff[i-1:]):
                dummy = {}
                
                dummy["i"]       = i
                dummy["j"]       = j
                dummy["sigma"]   = round( ( iatom["sigma"] + jatom["sigma"] ) / 2 if settings["mixing_rule"] == "arithmetic" 
                                        else np.sqrt( iatom["sigma"] * jatom["sigma"] ) if settings["mixing_rule"] == "geometric" 
                                        else -1.0 , 4)
                dummy["epsilon"] = round( np.sqrt( iatom["epsilon"] * jatom["epsilon"] ) , 4)
                dummy["m"]       = ( iatom["m"] + jatom["m"] ) / 2

                pair_interactions.append(dummy) 

        ## Create for each lambda intermediate an own input script ##
    
        sim_lj           = settings["sim_lj"]
        l_lj_sim,l_q_sim = settings["lambdas"]
        
        for k,(l_lj,l_q) in enumerate(zip(l_lj_sim,l_q_sim)):
            
            # Seed for velocity generation
            settings["seed"] = random.randint(1,10**6) 
            
            # Lambda of the simulation (either its the lj route or the coulomb route)

            lambda_sim  = l_lj if sim_lj else l_q
            lambda_coul = (1-l_q**2)/l_q**2

            if settings["method"] == "FEP":
                if sim_lj:
                    if k == 0: dlambda_sim = [ l_lj_sim[k+1]-l_lj, 0 ]
                    elif k == len(l_lj_sim)-1: dlambda_sim = [ 0, l_lj_sim[k-1]-l_lj ]
                    else: dlambda_sim = [ l_lj_sim[k+1]-l_lj, l_lj_sim[k-1]-l_lj ]
                else:
                    if k == 0: dlambda_sim = [ l_q_sim[k+1]-l_q, 0 ]
                    elif k == len(l_q_sim)-1: dlambda_sim = [ 0, l_q_sim[k-1]-l_q ]
                    else: dlambda_sim = [ l_q_sim[k+1]-l_q, l_q_sim[k-1]-l_q ]

            elif settings["method"] == "TI":
                dlambda_sim  = [0.0001,-0.0001]

            # Set initial charges for this simulation run depending on the choosen lambda (for the molecule thats added to the solution)

            init_charges = [ [i, self.nonbonded[i-1-add_type]["charge"]] for i in a_list1_add ]

            for dum in pair_interactions:
                # Intramolecular LJ and Coulomb interactions are always on for solute
                if all([dum["i"] in a_list1_add,dum["j"] in a_list1_add]):
                    # A-A interactions
                    dum["lambda_lj"]   = 1.0
                    dum["lambda_coul"] = lambda_coul

                # Intermolecular interactions are scaled
                elif all([dum["i"] in a_list1,dum["j"] in a_list1_add]) or all([dum["i"] in a_list1_add,dum["j"] in a_list2]):
                    # A-B interactions 
                    dum["lambda_lj"]   = l_lj
            
            ## Use jinja2 template to write LAMMPS input file ##

            input_file = input_file_name%str(k)

            if not os.path.exists(input_file[:input_file.rfind("/")+1] ): os.makedirs(input_file[:input_file.rfind("/")+1] )
            
            in_template = "../templates/free_energy.input"

            # Define name for output files according to simulated intermediate #

            file_names = [ ["fep_%s.fep"%settings["txt_add"]+str(k)+str(k+1),
                            "fep_%s.fep"%settings["txt_add"]+str(k)+str(k-1)],
                            "values.sampling" ]

            with open(in_template) as file:
                template = Template(file.read())
                
            rendered = template.render(pairs=pair_interactions,style=self.style,path=data_path,file_name=file_names,
                                        special_bonds=self.special_bonds,sp_bond=self.sb_dict,shake=self.shake,
                                        s_dict=self.s_dict,charge_list=init_charges,rcut=self.rcut,method=settings["method"],
                                        chr_flag=self.uncharged,a_list1=a_list1,a_list2=a_list2,a_list1_l=a_list1_add,
                                        lambda_sim=lambda_sim,dlambda_sim=dlambda_sim,lambda_coul=lambda_coul,
                                        set=settings,sim_lj=sim_lj,temperature=t,pressure=np.round(p/1.01325,3))
            
            with open(input_file, "w") as fh:
                fh.write(rendered)       

        return 
    

def get_name(name):
    if name[0] == "c":
        return name[1]
    else:
        return name[0]