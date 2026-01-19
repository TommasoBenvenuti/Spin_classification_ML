import numpy as np  
from collections import Counter

def get_coord_vectors(atoms, metal_z, bond_threshold=2.5):
    """ The function calculate the coordination number of a compound
    containing a metal.

    args: atoms (ase structure), Atomic number of metal, threshold to consider an effective bond

    returns: n_coord, counts of chemical species in the first coordination sphere
    """
    symbols = atoms.get_chemical_symbols()
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()

    # Z --> boolean array --> nonzero --> tupla di array --> [0] --> array --> [0] --> scalare (posizione del metallo in Z)  
    # restiuisce una tupla di array, ogni array per numero di dimensioni di Z , faccio [0] e ottengo l'array e poi lo scalare
    condition = np.asarray(Z) == metal_z
    metal_idx = condition.nonzero()[0][0] # posizione del metallo

    pos_m = pos[metal_idx]
    dists = np.linalg.norm(pos - pos_m, axis=1) 

    ligand_indices = [i for i in range(len(Z)) if i != metal_idx and dists[i] <= bond_threshold] #lista
    
    if (len(ligand_indices) == 0) : 
       n_coord = 0 
       counts = {}
       return n_coord, counts

    ligand_symbols = [symbols[i] for i in ligand_indices]
    counts = Counter(ligand_symbols) # dizionario conteggi leganti

    # In this final part i calculate the mean distance beetwen the metal and 
    # ligands. I consider a bond if distance is smaller than mean distance multiplicate for
    # a costant. Clearly the parameter makes the difference. Adjusting it the program
    # can modulate the strictness to consider a bond

    ligand_dists = np.array([dists[i] for i in ligand_indices])
    mean_dist = ligand_dists.mean()
    ligand_indices = [i for i in ligand_indices if dists[i] <= mean_dist * 1.5]
    
    n_coord = len(ligand_indices)

    return n_coord, counts

#---------------------------------------------------------------------------------------
    # manually counts! 
    #counts = {}                 # creo il dizionario vuoto
    #for simb in ligand_symbols: # per tutti i simboli nella lista
    #  if simb in counts:        # se il simbolo è presente già nella lista allora:
    #    counts[simb] +=1
    #  else:                     # altrimenti inizializza a 1!  
    #    counts[simb] = 1
#---------------------------------------------------------------------------------------    
