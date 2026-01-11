import os
import numpy as np
from sklearn.decomposition import PCA

def Convert_in_lowdim(OUT_DIR, X, y_spin, y_gap, ligand_counts):
    """
    Riduce la dimensionalità del dataset SOAP iniziale 
    utilizzando PCA per mantenere il 95% della varianza.
    Salva il dataset ridotto in un file compresso .npz.
    
    Args:
        OUT_DIR: Cartella di output per salvare i risultati
        X: Matrice delle caratteristiche SOAP
        y_spin: Etichette di spin
        y_gap: Altre etichette (es. gap di energia)
        ligand_counts: Conteggi dei leganti
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_.cumsum()
    k_95 = np.argmax(explained_variance >= 0.95) + 1
    print("Number of PCA components to explain 95.0% variance:", k_95)

    pca_opt = PCA(n_components=k_95)
    X_pca_opt = pca_opt.fit_transform(X)
    print("Shape after PCA reduction:", X_pca_opt.shape)

    OUT_FILE = os.path.join(OUT_DIR, "dataset_pca_95.npz")

    np.savez_compressed(
        OUT_FILE,
        X_pca=X_pca_opt,      # dataset ridotto
        y_spin=y_spin,        # etichette spin
        y_gap=y_gap,          # eventuali altre etichette
        ligand_counts=ligand_counts
    )

