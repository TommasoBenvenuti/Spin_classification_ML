import numpy as np  


def get_coord_vectors(atoms, metal_z, bond_threshold=2.5):
    symbols = atoms.get_chemical_symbols()
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()

    metal_idx = np.where(Z == metal_z)[0]
    if len(metal_idx) == 0:
        return 0, np.zeros((0, 6)), np.zeros((0, 0))
    metal_idx = metal_idx[0]

    pos_m = pos[metal_idx]
    dists = np.linalg.norm(pos - pos_m, axis=1)

    ligand_indices = [i for i in range(len(Z)) if i != metal_idx and dists[i] <= bond_threshold]
    
    ligand_symbols = [symbols[i] for i in ligand_indices]
    counts = {}
    for sym in ligand_symbols:
        counts[sym] = counts.get(sym, 0) + 1
    
    if len(ligand_indices) == 0:
        return 0, np.zeros((0, 6)), np.zeros((0, 0))

    ligand_dists = np.array([dists[i] for i in ligand_indices])
    mean_dist = ligand_dists.mean()
    ligand_indices = [i for i in ligand_indices if dists[i] <= mean_dist * 1.5]
    
    n_coord = len(ligand_indices)

    return n_coord, counts