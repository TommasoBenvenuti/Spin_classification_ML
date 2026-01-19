import os
import numpy as np
import pickle
from ase                                           import Atoms 
from Data.helper.coord_number      import get_coord_vectors
from Data.helper.safe_scalar       import safe_scalar
from Data.helper.target_single_ion import target_single_ion
from dscribe.descriptors                           import SOAP
from fairchem.core.datasets                        import AseDBDataset


# ------------------------------------------------------------------
# this is the dataset used to create both SOAP and non-SOAP datasets. Change path
base_dir = "/mnt/storage/Univ/Magistrale/Chimica/Machine_learning/Esercitazioni/ML/train_4M"
dataset = AseDBDataset({"src": base_dir})
print("Numero strutture nel dataset:", len(dataset))
# ------------------------------------------------------------------


def get_ASE_fromDB(min_atoms, max_atoms, target_metal_z, target_coord_no, target_spin_values):
    """
    Extract ase structure from AseDBDataset to be used for SOAP feature extraction,
    filtering for single ion complex with specified coordination number.
    args:
        dataset: AseDBDataset object
        min_atoms: minimum number of atoms in the structure
        max_atoms: maximum number of atoms in the structure
    returns:
         saves a compressed npz file with ase atoms, target properties (homo-lumo gap, spin state), lig_counts (for data an.)
         soap_features = vector containing SOAP descriptors for each atom in a sphere arounf metal with specified radius (cutoff)
         y_spin = vector containg spin state for each molecule
         y_gap = vector containing homo-lumo gap for each molecule
         y_s2 = vector containing s squared for each molecule
         ligand_counts_array = array containing ligand counts for each molecule in a radius of lenght cutoff
         ligand_species = array containing ligand species for each molecule

    """
    collected = {s: 0 for s in target_spin_values} # set comprehension
    soap_vectors = []
    data = []
    for idx in range(len(dataset)):
        try:
            atoms = dataset.get_atoms(idx)
            atomic_numbers = atoms.get_atomic_numbers()

            # restituisce quante volte conta un metallo di interesse
            if list(atomic_numbers).count(target_metal_z) != 1:
                continue # se non trovi il metallo di interesse torna all'inzio a cercare nuove strutture

            if not (min_atoms < len(atoms) < max_atoms):
                continue # fino a qui ok

            n_coord,lig_counts = get_coord_vectors(atoms, target_metal_z, 2.5)

            if n_coord != target_coord_no:
                continue
            
            info = atoms.info # funzione di ASE
            # fra info estrai con get il risultato della chiave 'spin', se non c'è nan
            # di questo che hai estratto fai safe-scalar (funzione che ho scritto io) 

            spin_val_raw = safe_scalar(info.get("spin", np.nan))
            if not np.isfinite(spin_val_raw):
                continue
            spin_val = int(round(spin_val_raw))

            if spin_val not in target_spin_values:
                continue
            # se hai superato tutte queste condizioni !
            data.append({
                "atoms": atoms,
                "spin_state": spin_val,
                "lig_counts": lig_counts
            })
            collected[spin_val] += 1
            if sum(collected.values()) % 1000 == 0:
                print(f"Collected {sum(collected.values())} samples")
        except Exception:
            continue
    return data

def get_soap_features(cutoff_input, lmax, n_max, min_atoms, max_atoms, target_metal_z, target_coord_no, target_spin_values):

    """
    Extract SOAP features from AseDBDataset, creating a SOAP object using DScribe. 
    it considers a sphere around a center (in this case a metal)
    
    args: 
        data: data obtained via previous extraction (!!! ase objects !!!)
        cutoff_input: cutoff radius for SOAP features
        lmax: maximum l value for SOAP
        nmax: maximum n value for SOAP
        
    returns: 
       saves a compressed npz file with SOAP features and target properties
    """
    print("Extracting structures...")
    data = get_ASE_fromDB(min_atoms, max_atoms, target_metal_z, target_coord_no, target_spin_values)
    print(f"Structures extracted: {len(data)}")
    
    rcut_cutoff = cutoff_input
    structure_list = []
    ligand_list = []

    print("Creating metal centered soap structures...")
    for d in data:
        atoms = d["atoms"]
        Z = atoms.get_atomic_numbers()
        condition = np.asarray(Z) == target_metal_z
        metal_idx = condition.nonzero()[0][0]
        pos_m = atoms.positions[metal_idx]

        # seleziona tutti gli atomi entro rcut_cutoff dallo ione metallico
        dists = np.linalg.norm(atoms.positions - pos_m, axis=1)
        local_mask = dists <= rcut_cutoff
        sub_atoms = atoms[local_mask]

        # ricalcola l'indice del ferro nel sottoinsieme
        Z_sub = sub_atoms.get_atomic_numbers()
        condition = np.asarray(Z_sub) == target_metal_z
        metal_idx_sub = condition.nonzero()[0][0]

        structure_list.append((sub_atoms, metal_idx_sub)) # lista di tuple
        
        ligand_list.append(d["lig_counts"])
    print(f"Strutture locali create: {len(structure_list)}")

    # ===  Vettori target ===
    y_spin = np.array([d["spin_state"] for d in data]) 

    # ===  Calcolo SOAP centrato sul Fe === 
    # ora devo mettere tutti gli atomi in un contenirore senza duplicati (set !)

    species = set()
    for struct in structure_list: 
        sub_atoms = struct[0]
        for atom in sub_atoms:
          istant_species = atom.symbol # è un attributo di atom
          species.add(istant_species)

    soap = SOAP(
        species=species,
        r_cut=cutoff_input,
        n_max=n_max,
        l_max=lmax,
        periodic=False,
        sparse=False
    )

    soap_vectors = []
    for sub_atoms, metal_idx_sub in structure_list:
        vec = soap.create(sub_atoms, centers=[metal_idx_sub])
        soap_vectors.append(vec[0])

    soap_features = np.array(soap_vectors)
    print("Dimensione SOAP finale:", soap_features.shape)

    np.savez_compressed(
        "./Soap_data.npz",
        soap_features=soap_features,
        y_spin=y_spin
        )
    with open ('./saved_lig_counts.pkl', 'wb') as f:
        pickle.dump(ligand_list, f)

    return soap_features, y_spin, ligand_list




