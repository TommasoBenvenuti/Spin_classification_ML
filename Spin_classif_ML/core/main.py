import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disabilita GPU se non richiesta 
# mine functions in the module 
from Data.manipulate.balance_data import get_balanced_data
from Data.manipulate.Convert_in_lowdim import Convert_in_lowdim
from Data.helper.safe_scalar import safe_scalar
from Data.helper.target_single_ion import target_single_ion
from Data.helper.coord_number import get_coord_vectors
from Data.extract.create_data import extract_xyz_from_db
from Data.extract.create_data import get_ASE_fromDB
from Data.extract.create_data import get_soap_features
from Data_analysis.Data_analyis_SOAP import analyze_data
from preprocessing.scale_split import scale_and_split_data
from fairchem.core.datasets import AseDBDataset
from SVM_RFC.SVM_RFC import NON_NN
from MLP.MLP import Multi_Layer_Perceptron 
from end_of_file import final_image

# Set random seed for reproducibility per tutte le operazioni 
RANDOM_STATE = 42
# ----------------------------------------------------------------------
#               extract input parameters from .txt
# ----------------------------------------------------------------------

params = {} # dizionario di parametri

with open("input.txt") as f:
    for line in f:
        line = line.strip() # rimuovo spazi 
        if not line or line.startswith("#"):
           continue
        key , value = line.split("=") # ottengo chiave valore separati da =
        params[key.strip()] = value.strip()

    
spin_val_1      = int(params["spin_val_1"])              
spin_val_2      = int(params["spin_val_2"])
target_coord_no = int(params["target_coord_no"])              
target_metal_z  = int(params["target_metal_z"])
min_no_atoms    = int(params["min_no_atoms"])
max_no_atoms    = int(params["max_no_atoms"])
n_max           = int(params["soap_n_max"])
l_max           = int(params["soap_l_max"])
cutoff          = float(params["cutoff"])
TEST_SIZE       = float(params["test_size"])

if params["use_balanced_data"].lower() == "yes":
   performe_data_balancing = True
else:
   performe_data_balancing = False

if params["data_already_extracted"].lower() == "yes":
   performe_extraction    = False
else:
    performe_extraction   = True
  
if params["convert_in_lowdim"].lower() == "yes":
   use_pca_data           = True
else: 
    use_pca_data          = False 
    
target_spin_values = [spin_val_1, spin_val_2] # array con i valori di spin target 

# ----------------------------------------------------------------------
#                      Directory setup
# ----------------------------------------------------------------------
OUT_DIR_DA = "./results/data_analysis_SOAP"

if performe_data_balancing:

    if use_pca_data:
       OUT_DIR_ML      = "../results/spin_classif_ML_balanced_PCA"
    else:  
       OUT_DIR_ML      = "../results/spin_classif_ML_balanced_noPCA"

    OUT_DIR_SVM_RFC = "../results/SVM_RFC_balanced"
else:
    if use_pca_data:
       OUT_DIR_ML      = "../results/spin_classif_ML_unbalanced_PCA"
    else:   
       OUT_DIR_ML      = "../results/spin_classif_ML_unbalanced_noPCA"
    OUT_DIR_SVM_RFC = "../results/SVM_RFC_unbalanced"

os.makedirs(OUT_DIR_DA, exist_ok=True)
os.makedirs(OUT_DIR_ML, exist_ok=True)
os.makedirs(OUT_DIR_SVM_RFC, exist_ok=True)

# ----------------------------------------------------------------------
#                           get data
# ----------------------------------------------------------------------
# se i dati sono già disponibili, ovvero non è la prima volta che testo il modello su un dato ione, non sto a ricercare tutto il dataset
if performe_extraction:
    print("Extracting data from database...")
    target_single_ion(target_spin_values, target_coord_no, target_metal_z)
    
    soap_features, y_spin, y_gap, y_s2, ligand_counts_array, ligand_species = get_soap_features(
        cutoff, l_max, n_max,
        min_no_atoms, max_no_atoms,
        target_metal_z, target_coord_no,
        target_spin_values
    )
    
    X = soap_features  # il vettore di input X sono le features SOAP
    print("Data extraction completed.")
    print("SOAP features shape:", X.shape)
    print("y_spin shape:", y_spin.shape)

else:
    loaded = np.load("Soap_data.npz")
    
    X = loaded["soap_features"].astype(np.float32)
    y_gap = loaded["y_gap"].astype(np.float32)
    y_spin = loaded["y_spin"].astype(int).ravel()
    y_s2 = loaded["y_s2"].astype(np.float32)
    ligand_counts_array = loaded["ligand_counts"]
    ligand_species = loaded["ligand_species"]
    
    print("Data extraction completed.")
    print("SOAP features shape:", X.shape)
    print("y_spin shape:", y_spin.shape)

# ---------------------------------------------------------------------
#                 Analisi dati prima della ML
# ---------------------------------------------------------------------
# Analisi dati con PCA e t-SNE, k-means clustering e istogrammi. Complete info in 
# function analyze_data in Data_analysis/Data_analyis_SOAP.py
print("Performing data analysis...")
analyze_data(X, y_gap, y_s2, y_spin, ligand_counts_array, ligand_species, OUT_DIR_DA)
print("Data analysis completed. Everything saved in", OUT_DIR_DA)

# ----------------------------------------------------------------------
#                 Data balancing (if requested)
# ----------------------------------------------------------------------
if performe_data_balancing:
   print("Performing data balancing...")
   X, y_gap, y_spin = get_balanced_data(X, y_spin, y_gap)
# ---------------------------------------------------------------------
#                           Preprocessing
# ----------------------------------------------------------------------
print("Scaling and splitting data...")
X_scaled, X_train, X_test, y_train_spin, y_test_spin = scale_and_split_data(X, y_spin, TEST_SIZE, RANDOM_STATE, target_spin_values)

# --------------------------------------------------------------------
#                     SVM and RFC ML classification
# ----------------------------------------------------------------------
# algorithm svm and random forest classifier for spin-classification on soap data
# svm uses a gaussian kernel to try to get better resuslts
# salva tutte le immagini in results/SVM_RFC

NON_NN(X_train, y_train_spin, X_test, y_test_spin, OUT_DIR_SVM_RFC)
print("SVM and RFC classification completed. Everything saved in", OUT_DIR_SVM_RFC)
# ----------------------------------------------------------------------
#                 MLP classification
# ----------------------------------------------------------------------
# Multi Layer Perceptron for spin-classification on soap data
# salva tutte le immagini in results/spin_classif_ML
# Binary classification. The function is quite general. I think that, adapting
# the layers, the function  could be used for other classification tasks as well.

# is way better if i use PCA data. the dataset is very small, and the risk of overfitting
# is very very high.

if use_pca_data == True: # restituisce x e y scalate e divise, post pca sulla X
   Convert_in_lowdim(OUT_DIR_ML, X, y_spin, target_spin_values, RANDOM_STATE, TEST_SIZE, target_variance=0.9999) 
Multi_Layer_Perceptron(X_train, y_train_spin, X_test, y_test_spin, OUT_DIR_ML)
print("MLP classification completed. Everything saved in", OUT_DIR_ML)

print("!! =) End of Program =) !!")

final_image()
