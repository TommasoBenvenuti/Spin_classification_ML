import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def analyze_data(X_train, X_test,  y_spin_train, y_spin_test, train_index, ligand_count_dict, target_variance, OUT_DIR):

    """ Analyze and visualize data with PCA, tsne, k-means clustering
    pca and tsne are performed on the input features X_train to detect connecttion beetwen geometry and electronic properties.
    I prefer fitting pca just on train data, to avoid introducing any leakage. Plotting just train data I lose some info. 
    Data are plotted and figures saved in OUT_DIR. 
    Histograms, PCA and TSNE colored by ligand ratios and dominant ligands are included.
    
    args:
        X: soap features array
        y_spin:  Spin state values.
        ligand_species:  Counts of atoms coordinated per sample.
        returns:
        None (plots are saved to OUT_DIR)     
        plots: t-SNE and pca colored by:
        -) spin state
        -) clusters (obtained on pca data)
        -) dominant atom ligand 

    """
    # ---------------------------------------------------------------------------------------------------------------
    #                                   Analisi dati con pca e tsne
    # ---------------------------------------------------------------------------------------------------------------
    
    # pca on data to explain target variance (then I'll do on two components to visualize data.. line 72)
    pca_full = PCA(n_components=None, random_state=42)
    X_pca_train = pca_full.fit_transform(X_train) # fitto soltanto sui dati di train e ottengo gli autovalori

    # Numero di componenti per target_variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    k_opt = np.argmax(cumulative_variance >= target_variance) + 1
    print(f"Number of PCA components to explain {target_variance*100:.1f}% variance: {k_opt}")

    # PCA con numero ottimale di componenti
    # here I repeat something i've done before. I do not know how to avoid
    pca_opt = PCA(n_components=k_opt)
    X_train_pca = pca_opt.fit_transform(X_train) 
    X_test_pca  = pca_opt.transform(X_test)
    print("Shape after PCA reduction:", X_train_pca.shape)
   
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

    np.savez_compressed(
            "./Soap_data_PCA.npz",
        X_train_pca  =  X_train_pca,
        X_test_pca   =  X_test_pca ,
        y_spin_train =  y_spin_train,
        y_spin_test  =  y_spin_test 
        )
    print(f"Scree plot saved to: {plot_file}")

    # here to visualize data  
    pca_for_visualizing = PCA(n_components=2, random_state=42)
    X_pca_train = pca_for_visualizing.fit_transform(X_train) # fitto soltanto sui dati di train e ottengo gli autovalori

    # -------------------------------------------------
    #                      t-SNE 
    # -------------------------------------------------
    # non linear 2D reduction to better visualize data
    # Init PCA  gives a good intial guess to the fitting procedure 
    # Perplexity bilancia tra vicini locali e globali.                      
    tsne = TSNE(n_components=2, random_state=42, 
                learning_rate='auto', init='pca', 
                perplexity=30)
    X_tsne = tsne.fit_transform(X_pca_train)

    # -----------------
    # Cluster KMeans
    # -----------------
    # I prefer doing Kmeans clustering on pca data because my pc is suffering =/ . I hope it's the same
    # I check what is the best numbers that fit clusters
    # silhouette_score measurs how much points are closer to their cluster than to the others 


    K_range = range(2, 9)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init indica quante volte l'alg. viene rilanciato con inizial. diverse
        labs = km.fit_predict(X_pca_train)
        try:
            score = silhouette_score(X_pca_train, labs)
        except Exception:
            score = np.nan
        sil_scores.append(score)
    best_k = K_range[np.nanargmax([s if not np.isnan(s) else -999 for s in sil_scores])]
    
    #Scelgo il numero di cluster con silhouette più alta.
    print("Best k by silhouette:", best_k)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_pca_train)
    labels = kmeans.labels_
    # Calcolo i cluster finali e salvo le etichette per ogni campione.
    n_clusters =len(np.unique(labels))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))


    #                                            Creo una colormap per legante

    # mi assicuro di avere i conteggi dei leganti in una lista della stessa lunghezza di x_train
    filtered_train_compound = []
    for index in train_index: # these are original index of the compounds that became train compounds
        filtered_train_compound.append(ligand_count_dict[index])
    
    dominant_ligand = []
    for item in filtered_train_compound:
       dmn_lgn = max(item, key =item.get)
       dominant_ligand.append(dmn_lgn)

    dominant_ligand=np.array(dominant_ligand)        # una struttura lunga quanto il train con leganti dominanti per complesso
    dominant_ligand_unici=np.unique(dominant_ligand) # solo leganti dominanti unici... devo fare una mappa successivamente
    
    colormap_ligand = {} # dizionario di leganti_dominanti:colori
    for lig in dominant_ligand_unici:
        color = (np.random.rand(), np.random.rand(),np.random.rand()) # not the best choice for colors
        colormap_ligand.update({lig : color})
    # ottengo il valore associato a ciascuna chiave facendo colormap_ligand[lig]
    point_color=[colormap_ligand[lig] for lig in dominant_ligand]    # ora ho per ogni punto il proprio colore!

     #                                                         plot

    # silhuette score plt
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(K_range, sil_scores, color="blue")
    plt.axvline(x = best_k, color = "orange", label = "best k score", linestyle = '--')
    ax.set_xlabel("N. clusters")
    ax.set_ylabel("Sil. score")
    ax.set_title("Silhouette score vs number of cluster")
    plt.savefig(os.path.join(OUT_DIR,"s_score.png"), dpi=300)
    plt.close(fig)
    # PCA colorata per cluster - VERSIONE CORRETTA
    fig, ax = plt.subplots(figsize=(7,5))

    # prendo il primo valore sotto il quale trovo il 99 percento dei dati
    # e il primo sotto cui trovo il 0.5 peercento. Per escludere outlier
    x_min, x_max = np.percentile(X_pca_train[:,0], [0.1, 99.9]) 
    y_min, y_max = np.percentile(X_pca_train[:,1], [0.1, 99.9])

    unique_labels = np.unique(labels)
    
    # Plotta ogni cluster separatamente
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        ax.scatter(X_pca_train[mask, 0], 
               X_pca_train[mask, 1], 
               s=12, 
               alpha=0.8, 
               color=cluster_colors[i], 
               label=f"Cluster {cluster_id+1}")  # +1 per partire da 1 invece che da 0

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA colored by KMeans clusters (k={best_k})")
    ax.legend(title="Clusters", loc='best')
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_full_clusters.png"), dpi=300)
    plt.close(fig)
 
    # PCA colorata per stato di spin
    fig, ax = plt.subplots(figsize=(7,5))

    # Colori discreti (uno per ogni valore di spin)
    spin_values = np.unique(y_spin_train)
    spin_cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(spin_values))))

    sc = ax.scatter(
        X_pca_train[:, 0],
        X_pca_train[:, 1],
        c=y_spin_train,
        cmap=spin_cmap,
        s=20,
        alpha=0.85
    )

    cbar = fig.colorbar(sc, ax=ax, ticks=spin_values)
    cbar.set_label("Spin state")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by spin state")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_spin_state.png"), dpi=300)
    plt.close(fig)

    # PCA colorata per legante dominante
    fig, ax = plt.subplots(figsize=(7,5))

    sc = ax.scatter(
        X_pca_train[:, 0],
        X_pca_train[:, 1],
        color=point_color,
        s=20,
        alpha=0.85
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA colored by dominant ligand")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
   
    #    from stack overflow
    # -------- labels ---------
    markers = [
       plt.Line2D([0,0], [0,0],
               color=color,
               marker='o',
               linestyle='')
    for color in colormap_ligand.values()
              ]

    ax.legend(markers,
          colormap_ligand.keys(),
          title="Dominant ligand",
          bbox_to_anchor=(1.05, 1),
          loc="upper left",
          numpoints=1)

 
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_legante.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per stato di spin
    fig, ax = plt.subplots(figsize=(7,5))
    spin_values = np.unique(y_spin_train)
    spin_cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(spin_values))))
    sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_spin_train, cmap=spin_cmap, s=20, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, ticks=spin_values)
    cbar.set_label("Spin state")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by spin state")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_spin_state.png"), dpi=300)
    plt.close(fig)

    # t-SNE colorata per legante dominante
    fig, ax = plt.subplots(figsize=(7,5))

    sc = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        color=point_color,
        s=20,
        alpha=0.85
    )

    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title("t-SNE colored by dominant ligand")
    # ------ labels --------
    markers = [
       plt.Line2D([0,0], [0,0],
               color=color,
               marker='o',
               linestyle='')
    for color in colormap_ligand.values()
              ]

    ax.legend(markers,
          colormap_ligand.keys(),
          title="Dominant ligand",
          bbox_to_anchor=(1.05, 1),
          loc="upper left",
          numpoints=1)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "t-SNE_legante.png"), dpi=300)
    plt.close(fig)

 # Plotta ogni cluster separatamente
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        ax.scatter(X_tsne[mask, 0], 
               X_tsne[mask, 1], 
               s=12, 
               alpha=0.8, 
               color=cluster_colors[i], 
               label=f"Cluster {cluster_id+1}")  # +1 per partire da 1 invece che da 0

    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title(f"t-SNE colored by KMeans clusters (k={best_k})")
    ax.legend(title="Clusters", loc='best')
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsne_full_clusters.png"), dpi=300)
    plt.close(fig)

