import numpy as np
# In alcune casi devono essere state caricate delle strutture nel
# dataset che non erano andate perfettamente a convergenza.
# Il valore dello spin non è univoco e ASE lo carica come  un array. 
# Per evitare questa cosa ne faccio il valor medio. Se il valore è unico
# restituisce il valore (ne fa la media), altrimenti fa la media fra i due


def safe_scalar(x):
    """ Takes an input (a scalar or an array) and converts to a scalar, by returing the mean value or zero if the array contains nan
    
    args: x (scalare o array)
    
    returns: The mean value of the array (the mean value of the scalar is clearly the scalar itself, and this is what we want) if everyrhing is correct,  
    0.0 if the array contains nan or if the code gets stuck because of some bug
       """  
    try: # prova a convertire in array
        arr = np.array(x)
        val = np.nanmean(arr)
        return float(val) if np.isfinite(val) else 0.0 # se è un float non infinito restiuisci altrimenti restiuisci uno 0
    except Exception:
        return 0.0 # se ci sono stati errori restuisici 0
