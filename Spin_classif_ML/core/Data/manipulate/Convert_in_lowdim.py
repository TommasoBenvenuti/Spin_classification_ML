import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# ----------------------------------------------------------
# Calcolo del numero minimo di componenti PCA necessario
# per raggiungere la varianza cumulativa.
#
# 1. la pca trova una base ortonormale che è un insieme auto-
#    vettori della matrice di covarianza (ogni autovettore 
#    è comb. lineare degli altri ed è una componente princ.)
# 2. le prime compon. cattura la massima var., la seconda cat-
#    tura la seconda maggiore ecc.
# 3. pca_full.explanied. restiuisce un vettore che contiene
#    quanta varianza totale spiega ogni componente
# 4. calcolo la varianza comulativa (li sommo tutti)
# 5. individuo primo numero di componenti che supera il valore
#    target
# 6. Creazione di una PCA ridotta con solo k_opt componenti
#    per comprimere i dati senza perdere informazione signifi-
#    cativa.
# 7. plot della varianza comulativa in funz delle componenti
# ----------------------------------------------------------

def Convert_in_lowdim(OUT_DIR, X, y_spin, target_spin_values, RANDOM_STATE, TEST_SIZE, target_variance=0.9999):
    """
    Riduce la dimensionalità del dataset SOAP usando PCA e salva i risultati.
    Scala e divide in train-test set per MLP 
    Salva anche uno scree plot della varianza cumulativa.

    Args:
        OUT_DIR: cartella di output
        X: matrice delle caratteristiche SOAP (n_samples, n_features)
        y_spin: etichette di spin
        target_variance: frazione di varianza da mantenere (default 0.99)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA completa
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Numero di componenti per target_variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    k_opt = np.argmax(cumulative_variance >= target_variance) + 1
    print(f"Number of PCA components to explain {target_variance*100:.1f}% variance: {k_opt}")

    # PCA con numero ottimale di componenti
    pca_opt = PCA(n_components=k_opt)
    X_pca_opt = pca_opt.fit_transform(X_scaled)
    print("Shape after PCA reduction:", X_pca_opt.shape)

   
    # Scree plot
    plt.figure(figsize=(6,4))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=target_variance, color='r', linestyle='--', label=f'{int(target_variance*100)}% varianza')
    plt.xlabel("Numero componenti PCA")
    plt.ylabel("Varianza cumulativa spiegata")
    plt.title("Scree plot PCA")
    plt.grid(True)
    plt.legend()
    plot_file = os.path.join(OUT_DIR, f"scree_plot_pca_{int(target_variance*100)}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Scree plot saved to: {plot_file}")

    # !!!!!!!!!!!!!!!
    # split data in order to have it ready to mlp 

    X_train, X_test, y_train_spin, y_test_spin = train_test_split(
        X_pca_opt, y_spin, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_spin
    )

    y_train_spin = (y_train_spin == target_spin_values[0]).astype(int) # converto in 0 1 per class. bin.
    y_test_spin = (y_test_spin == target_spin_values[0]).astype(int) 
    print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

    return X_train, X_test, y_train_spin, y_test_spin

