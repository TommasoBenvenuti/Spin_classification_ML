# =========================================================================================
# =====================  DATA ANALYSIS & VISUALIZATION COMPLETO ===========================
# =========================================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

def analyze_data(X, y_gap, y_s2, y_spin, ligand_counts, ligand_species, OUT_DIR):


    """ Analyze and visualize data with PCA, tsne, k-means clustering, and histograms.
    pca and tsne are performed on the input features X to detect connecttion beetwen geometry and electronic properties.
    Data are plotted and figures saved in OUT_DIR. 
    Histograms, PCA and TSNE colored by ligand ratios and dominant ligands are included.
    
    args:
        X: soap features array
        y_gap: Energy gap values.
        y_s2: S squared values.
        y_spin:  Spin state values.
        ligand_counts:  Counts of atoms coordinated per sample.
        ligand_species: Species of ligands coordinated to the metal center.
        
        
        returns:
        None (plots are saved to OUT_DIR) 
        Plot:
            PCA and t-SNE colored by dominant ligand, 
            PCA and t-SNE colored by ligand ratios (N/O, N/H, O/H, O/P, N/halogen, O/halogen),
            PCA and t-SNE colored by energy gap,
            PCA and t-SNE colored by spin state and clusters (k-means),
            Histograms of energy gap colored by ligand ratios.
        
!!!!!!!!!!!!!! Sono tante immagini, l'importante si trova nelle linee consigliate sotto !!!!!!!!!!!!!!!   

    !!!!!!!!!!!!!!!     GOTO 98 for PCA and 287 for t-SNE    !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! GOTO 107/297 for k-means on reduced data !!!!!!!!!!!!!!!
    

    """
#####  All'inizio si trovano soltanto delle semplici operazioni o funzioni per analizzare i leganti. Dopo si passa alla PCA e infine alla t-SNE. #####

    # Filtra solo leganti attivi
    total_counts = ligand_counts.sum(axis=0)
    active_mask = total_counts > 0
    ligand_species = np.array(ligand_species)
    ligand_counts_active = ligand_counts[:, active_mask]
    ligand_species_active = ligand_species[active_mask]

    print("Leganti attivi:", ligand_species_active)
    
    # Funzione per conteggi aggregati per gruppi di elementi

    def get_count(species, counts, target_elems):
        idx = [np.where(species == el)[0][0] for el in target_elems if el in species]
        if len(idx) == 0:
            return np.zeros(counts.shape[0])
        return counts[:, idx].sum(axis=1)

    species = ligand_species_active
    all_elements = list(species)
    halogens = [el for el in ["F", "Cl", "Br", "I"] if el in all_elements]

    # Conteggi e rapporti
    N_count = get_count(species, ligand_counts_active, ["N"])
    O_count = get_count(species, ligand_counts_active, ["O"])
    H_count = get_count(species, ligand_counts_active, ["H"])
    P_count = get_count(species, ligand_counts_active, ["P"])
    C_count = get_count(species, ligand_counts_active, ["C"])
    Hal_count = get_count(species, ligand_counts_active, halogens)

    ratios = {
        "N-O": N_count / np.maximum(O_count, 1e-8),
        "N-Alogen": N_count / np.maximum(Hal_count, 1e-8),
        "O-Alogen": O_count / np.maximum(Hal_count, 1e-8),
        "N-H": N_count / np.maximum(H_count, 1e-8),
        "O-H": O_count / np.maximum(H_count, 1e-8),
        "O-P": O_count / np.maximum(P_count, 1e-8),
        "C-H": C_count / np.maximum(H_count, 1e-8),
        "C-O": C_count / np.maximum(O_count, 1e-8),
        "C-P": C_count / np.maximum(P_count, 1e-8),
    }


    # Legante dominante
    dominant_idx = np.argmax(ligand_counts_active, axis=1)
    dominant_ligand = np.array([ligand_species_active[i] for i in dominant_idx])
    unique_lig = np.unique(dominant_ligand)
    lig_map = {lig: i for i, lig in enumerate(unique_lig)}
    lig_idx = np.array([lig_map[l] for l in dominant_ligand])
    lig_colors = plt.cm.tab10(np.linspace(0,1,len(unique_lig)))
    lig_cmap = ListedColormap(lig_colors)

    #-----------------------------------------------------
    #                    PCA: geometria
    # ---------------------------------------------------
    X_full = X # np.hstack([X, y_gap[:, None], y_s2[:, None]])
    pca_full = PCA(n_components=2, random_state=42)
    X_pca_full = pca_full.fit_transform(X_full)

    # -----------------
    # Cluster KMeans
    # -----------------
    K_range = range(2, 9)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_pca_full)
        try:
            score = silhouette_score(X_pca_full, labs)
        except Exception:
            score = np.nan
        sil_scores.append(score)
    best_k = K_range[np.nanargmax([s if not np.isnan(s) else -999 for s in sil_scores])]
    print("Best k by silhouette:", best_k)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_pca_full)
    labels = kmeans.labels_
    n_clusters = len(np.unique(labels))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # PCA colorata per legante dominante
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(X_pca_full[:,0], X_pca_full[:,1], c=lig_idx, cmap=lig_cmap, s=20, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(unique_lig)))
    cbar.ax.set_yticklabels(unique_lig)
    cbar.set_label("Dominant ligand")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by dominant ligand")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_dominant_ligand.png"), dpi=300)
    plt.close(fig)

    # PCA colorata per ratio (subplots)
    ratio_keys_pca = ["O-P", "N-O", "N-Alogen", "O-Alogen", "N-H", "O-H"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for ax, ratio_key in zip(axs, ratio_keys_pca):
        ratio_data = ratios[ratio_key]
        sc = ax.scatter(X_pca_full[:,0], X_pca_full[:,1], c=ratio_data, cmap="coolwarm", s=12, alpha=0.8)
        ax.set_title(f"PCA colored by {ratio_key} ratio")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        # Colorbar percentuale
        min_val = np.nanmin(ratio_data)
        max_val = np.nanmax(ratio_data)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"{ratio_key} ratio (%)")
        cbar.set_ticks([min_val, max_val])
        cbar.set_ticklabels([f"{0:.0f}%", f"{100:.0f}%"])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_ratios_subplot_percent.png"), dpi=300)
    plt.close(fig)

    # PCA colorata per cluster
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_pca_full[mask,0], X_pca_full[mask,1], s=12, alpha=0.8, color=cluster_colors[i], label=f"Cluster {i+1}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by cluster")
    ax.legend(title="Clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_full_clusters.png"), dpi=300)
    plt.close(fig)
 
    # PCA colorata per energy gap
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(X_pca_full[:,0], X_pca_full[:,1], c=y_gap, cmap="viridis", s=12, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Energy gap (eV)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by energy gap")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_full_energy_gap.png"), dpi=300)
    plt.close(fig)

    # ISTOGRAMMA ENERGY GAP TOTALE (SENZA LEGANTI E RAPPORTI)
    plt.figure(figsize=(8,5))
    plt.hist(y_gap, bins=100, color="gray", alpha=0.8, edgecolor="black")
    plt.xlabel("Energy gap (eV)")
    plt.ylabel("Counts")
    plt.title("Energy gap distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "energy_gap_total.png"), dpi=300)
    plt.close()

    # ISTOGRAMMI ENERGY GAP COLORATI PER RAPPORTI CHIMICI (%)
    bins = 100
    bin_edges = np.linspace(y_gap.min(), y_gap.max(), bins+1)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for ax, ratio_key in zip(axs, ratio_keys_pca):
        ratio_data = ratios[ratio_key]
        ratio_per_bin = [np.nanmean(ratio_data[(y_gap>=bin_edges[i]) & (y_gap<bin_edges[i+1])])
                        if np.any((y_gap>=bin_edges[i]) & (y_gap<bin_edges[i+1])) else np.nan
                        for i in range(bins)]
        for i, center in enumerate(bin_centers):
            count = np.sum((y_gap>=bin_edges[i]) & (y_gap<bin_edges[i+1]))
            if count==0: continue
            ratio_val = ratio_per_bin[i]
            norm_val = (ratio_val - np.nanmin(ratio_data)) / (np.nanmax(ratio_data) - np.nanmin(ratio_data))
            color = plt.cm.coolwarm(norm_val)
            ax.bar(center, count, width=bin_edges[1]-bin_edges[0], color=color, alpha=0.9)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=np.nanmin(ratio_data), vmax=np.nanmax(ratio_data)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"{ratio_key} ratio (%)")
        cbar.set_ticks([np.nanmin(ratio_data), np.nanmax(ratio_data)])
        cbar.set_ticklabels([f"0%", f"100%"])
        ax.set_title(f"Energy gap colored by {ratio_key} ratio")
        ax.set_xlabel("Energy gap (eV)")
        ax.set_ylabel("Counts")
        fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "energy_gap_ratios_subplot.png"), dpi=300)
    plt.close(fig)


    # PCA colorata per stato di spin
    fig, ax = plt.subplots(figsize=(7,5))

    # Colori discreti (uno per ogni valore di spin)
    spin_values = np.unique(y_spin)
    spin_cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(spin_values))))

    sc = ax.scatter(
        X_pca_full[:, 0],
        X_pca_full[:, 1],
        c=y_spin,
        cmap=spin_cmap,
        s=20,
        alpha=0.85
    )

    cbar = fig.colorbar(sc, ax=ax, ticks=spin_values)
    cbar.set_label("Spin state")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by spin state")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_spin_state.png"), dpi=300)
    plt.close(fig)

    # PCA colorata per rapporto N/O (semplice)
    ratio_key = "N-O"
    ratio_data = ratios[ratio_key]

    fig, ax = plt.subplots(figsize=(7,5))

    sc = ax.scatter(
        X_pca_full[:, 0],
        X_pca_full[:, 1],
        c=ratio_data,
        cmap="coolwarm",
        s=20,
        alpha=0.85
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"{ratio_key} ratio")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by N/O ratio")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_ratio_N_O.png"), dpi=300)
    plt.close(fig)
# -------------------------------------------------
#                      t-SNE 
# -------------------------------------------------
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='pca', perplexity=30)
    X_tsne = tsne.fit_transform(X)

    # -------------------------------------------------
    # Cluster KMeans
    # -------------------------------------------------
    K_range = range(2, 9)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_tsne)
        try:
            score = silhouette_score(X_tsne, labs)
        except Exception:
            score = np.nan
        sil_scores.append(score)
    best_k = K_range[np.nanargmax([s if not np.isnan(s) else -999 for s in sil_scores])]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_tsne)
    labels = kmeans.labels_
    n_clusters = len(np.unique(labels))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # t-SNE colorata per legante dominante
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=lig_idx, cmap=lig_cmap, s=20, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(unique_lig)))
    cbar.ax.set_yticklabels(unique_lig)
    cbar.set_label("Dominant ligand")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by dominant ligand")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_dominant_ligand.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per ratio (subplots)
    ratio_keys_tsne = ["O-P", "N-O", "N-Alogen", "O-Alogen", "N-H", "O-H"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for ax, ratio_key in zip(axs, ratio_keys_tsne):
        ratio_data = ratios[ratio_key]
        sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=ratio_data, cmap="coolwarm", s=12, alpha=0.8)
        ax.set_title(f"t-SNE colored by {ratio_key} ratio")
        ax.set_xlabel("t-SNE1")
        ax.set_ylabel("t-SNE2")
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=np.nanmin(ratio_data), vmax=np.nanmax(ratio_data)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"{ratio_key} ratio (%)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_ratios_subplot.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per cluster
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_tsne[mask,0], X_tsne[mask,1], s=12, alpha=0.8, color=cluster_colors[i], label=f"Cluster {i+1}")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by cluster")
    ax.legend(title="Clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_clusters.png"), dpi=300)
    plt.close(fig)

     # t-SNE colorata per energy gap
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_gap, cmap="viridis", s=12, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Energy gap (eV)")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by energy gap")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_energy_gap.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per stato di spin
    fig, ax = plt.subplots(figsize=(7,5))
    spin_values = np.unique(y_spin)
    spin_cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(spin_values))))
    sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_spin, cmap=spin_cmap, s=20, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, ticks=spin_values)
    cbar.set_label("Spin state")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by spin state")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_spin_state.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per rapporto N/O (semplice)

    ratio_key = "N-O"
    ratio_data = ratios[ratio_key]

    fig, ax = plt.subplots(figsize=(7,5))

    sc = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=ratio_data,
        cmap="coolwarm",
        s=20,
        alpha=0.85
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"{ratio_key} ratio")

    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by N/O ratio")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_ratio_N_O.png"), dpi=300)
    plt.close(fig)

