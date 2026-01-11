import os
import numpy as np
from ase                                           import Atoms 
from Data.helper.coord_number      import get_coord_vectors
from Data.helper.safe_scalar       import safe_scalar
from Data.helper.target_single_ion import target_single_ion
from dscribe.descriptors                           import SOAP
from fairchem.core.datasets                        import AseDBDataset


# ------------------------------------------------------------------
# this is the dataset used to create both SOAP and non-SOAP datasets
base_dir = "/mnt/storage/Univ/Magistrale/Chimica/Machine_learning/Esercitazioni/ML/train_4M"
dataset = AseDBDataset({"src": base_dir})
print("Numero strutture nel dataset:", len(dataset))
# ------------------------------------------------------------------

def extract_xyz_from_db( min_atoms, max_atoms):
    """
    Extract positions and atomic numbers from AseDBDataset,
    filtering for single ion complex with specified coordination number.
    
    args:
        dataset: AseDBDataset object
        min_atoms: minimum number of atoms in the structure
        max_atoms: maximum number of atoms in the structure
    returns:
        saves a compressed npz file with positions, atomic numbers, and target properties (homo-lumo gap, spin state)"""

    target_spin_values, target_coord_no, collected, target_metal_z= target_single_ion(target_spin_values, target_coord_no, target_metal_z)
    data = []


    for idx in range(len(dataset)):
        try:
            atoms = dataset.get_atoms(idx)
            atomic_numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
  
            # Controlla se c'è esattamente un atomo target 
            if list(atomic_numbers).count(target_metal_z) != 1:
                continue
            if not (min_atoms < len(atoms) < max_atoms):  # Limita il numero di atomi
                continue

            info = atoms.info
            spin_val_raw = safe_scalar(info.get("spin", np.nan))
            if not np.isfinite(spin_val_raw):
                continue
            
            spin_val = int(round(spin_val_raw))
            
            if spin_val not in target_spin_values:
                continue
            
            n_cord = get_coord_vectors(atoms, metal_z=26, bond_threshold=2.5)
            
            if n_cord != target_coord_no:
                continue  # Solo complessi target 
            
            data.append({
                "positions": positions,
                "atomic_numbers": atomic_numbers,
                "spin_state": spin_val,
                "homo_lumo_gap": safe_scalar(info.get("homo_lumo_gap", np.nan)),
                "s_squared": safe_scalar(info.get("s_squared", np.nan)),
                
            })
            collected[spin_val] += 1

        except Exception:
            continue

    # === Padding delle posizioni ===
    # Trova il numero massimo di atomi
    max_atoms = max(len(d["positions"]) for d in data)
    print(f"Numero massimo di atomi: {max_atoms}")

    # Crea array con padding
    padded_positions = []
    padded_atomic_numbers = []

    for d in data:
        pos = d["positions"]
        atom_nums = d["atomic_numbers"]
        n_atoms = len(pos)
        
        # Padding per le posizioni (aggiungi zeri)
        padded_pos = np.zeros((max_atoms, 3))
        padded_pos[:n_atoms] = pos
        padded_positions.append(padded_pos)
        
        # Padding per i numeri atomici (aggiungi zeri)
        padded_nums = np.zeros(max_atoms, dtype=int)
        padded_nums[:n_atoms] = atom_nums
        padded_atomic_numbers.append(padded_nums)

    # Converti in array numpy
    positions = np.array(padded_positions)  # Shape: (n_structures, max_atoms, 3)
    atomic_numbers_array = np.array(padded_atomic_numbers)  # Shape: (n_structures, max_atoms)
    y_spin = np.array([d["spin_state"] for d in data])
    y_gap = np.array([d["homo_lumo_gap"] for d in data])
    y_s2 = np.array([d["s_squared"] for d in data])

    # === salvataggio ===
    np.savez_compressed(
        "./Non_soap_data.npz",
        positions=positions,
        atomic_numbers=atomic_numbers_array,
        y_gap=y_gap,
        y_spin=y_spin,
        y_s2=y_s2,
        max_atoms=max_atoms,  # Salva anche il numero massimo di atomi
    )

    return 


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
    collected = {s: 0 for s in target_spin_values}
    soap_vectors = []
    data = []
    for idx in range(len(dataset)):
        try:
            atoms = dataset.get_atoms(idx)
            atomic_numbers = atoms.get_atomic_numbers()

            if list(atomic_numbers).count(target_metal_z) != 1:
                continue

            if not (min_atoms < len(atoms) < max_atoms):
                continue # fino a qui ok

            n_coord,lig_counts = get_coord_vectors(atoms, target_metal_z, 2.5)

            if n_coord != target_coord_no:
                continue
            
            info = atoms.info
            spin_val_raw = safe_scalar(info.get("spin", np.nan))
            if not np.isfinite(spin_val_raw):
                continue
            spin_val = int(round(spin_val_raw))
            if spin_val not in target_spin_values:
                continue

            data.append({
                "atoms": atoms,
                "spin_state": spin_val,
                "homo_lumo_gap": safe_scalar(info.get("homo_lumo_gap", np.nan)),
                "s_squared": safe_scalar(info.get("s_squared", np.nan)),
                "lig_counts": lig_counts
            })
            collected[spin_val] += 1
            if sum(collected.values()) % 1000 == 0:
                print(f"Collected {sum(collected.values())} samples")
        except Exception:
            continue
    return data

def get_soap_features(cutoff_input, lmax, nmax, min_atoms, max_atoms, target_metal_z, target_coord_no, target_spin_values):

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
        metal_idx = np.where(Z == target_metal_z)[0][0]
        pos_m = atoms.positions[metal_idx]

        # seleziona tutti gli atomi entro rcut_cutoff dallo ione metallico
        dists = np.linalg.norm(atoms.positions - pos_m, axis=1)
        local_mask = dists <= rcut_cutoff
        sub_atoms = atoms[local_mask]

        # ricalcola l'indice del ferro nel sottoinsieme
        Z_sub = sub_atoms.get_atomic_numbers()
        metal_idx_sub = np.where(Z_sub == target_metal_z)[0][0]

        structure_list.append((sub_atoms, metal_idx_sub))
        
        if np.any(dists <= 2.5):
            ligand_list.append(d["lig_counts"])
        else:
            ligand_list.append({})
    

    print(f"Strutture locali create: {len(structure_list)}")

    # === 3) Vettori target ===
    y_spin = np.array([d["spin_state"] for d in data[:len(structure_list)]]) # solo nel subset
    y_gap = np.array([d["homo_lumo_gap"] for d in data[:len(structure_list)]])
    y_s2 = np.array([d["s_squared"] for d in data[:len(structure_list)]])

    # === 4) Calcolo SOAP centrato sul Fe ===
    species = sorted({sym for atoms, _ in structure_list for sym in atoms.get_chemical_symbols()})
    print("Specie trovate:", species)

    soap = SOAP(
        species=species,
        r_cut=2.6,
        n_max=2,
        l_max=3,
        periodic=False,
        sparse=False
    )

    soap_vectors = []
    for sub_atoms, metal_idx_sub in structure_list:
        vec = soap.create(sub_atoms, centers=[metal_idx_sub])
        soap_vectors.append(vec[0])

    soap_features = np.array(soap_vectors)
    print("Dimensione SOAP finale:", soap_features.shape)

    # === 5) Conversione dei leganti in feature numeriche ===
    ligand_species = sorted({k for d in ligand_list for k in d.keys()})
    ligand_counts_array = np.zeros((len(ligand_list), len(ligand_species)))
    for i, lig_dict in enumerate(ligand_list):
        for j, sym in enumerate(ligand_species):
            ligand_counts_array[i, j] = lig_dict.get(sym, 0)

    # === 6) Salvataggio dati ===
    np.savez_compressed(
        "./Soap_data.npz",
        soap_features=soap_features,
        y_gap=y_gap,
        y_spin=y_spin,
        y_s2=y_s2,
        ligand_counts=ligand_counts_array,
        ligand_species=ligand_species
    )

    return soap_features, y_spin, y_gap, y_s2, ligand_counts_array, ligand_species


