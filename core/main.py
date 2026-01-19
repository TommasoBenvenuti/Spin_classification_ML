import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
#disabilito la gpu, anche se ho avuto qualche problema a capire dove precisamente devo farlo
# se qui o quando importo tensorflow. 
# ! Su cluster funziona bene commentando tutte queste righe 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# mine functions in the module 
from preprocessing.eventual_pre_ML.balance_data import get_balanced_data
from Data.helper.safe_scalar import safe_scalar
from Data.helper.target_single_ion import target_single_ion
from Data.helper.coord_number import get_coord_vectors
#from Data.extract.create_data import extract_xyz_from_db
from Data.extract.create_data import get_ASE_fromDB
from Data.extract.create_data import get_soap_features
from Data_analysis.Data_analyis_SOAP import analyze_data
from preprocessing.preprocessing import scale_and_split_data
from fairchem.core.datasets import AseDBDataset
from SVM_RFC.SVM_RFC import NON_NN
from MLP.MLP import Multi_Layer_Perceptron 
from end_of_file import final_image

# Set random seed for reproducibility per tutte le operazioni 
RANDOM_STATE = 42
# ----------------------------------------------------------------------
#               extract input parameters from .txt
# ----------------------------------------------------------------------

params = {} # dizionario di input

with open("input.txt") as f:
    for line in f:
        line = line.strip() # rimuovo spazi 
        if not line or line.startswith("#"):
           continue
        key , value = line.split("=") # ottengo chiave valore separati da =
        params[key.strip()] = value.strip()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# la classe positiva per la classificazione binaria è il primo valore riportato nell'input
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
spin_val_1      = int(params["spin_HS"])         # primo valore di spin da class. (alto spin)
spin_val_2      = int(params["spin_LS"])         # secondo valore di spin (basso spin)
target_coord_no = int(params["target_coord_no"]) # numero di coordinazione ( ottaedrico, trigonale planare ecc...)
target_metal_z  = int(params["target_metal_z"])  # metallo da riconoscere
min_no_atoms    = int(params["min_no_atoms"])    # minimo numero di atomi nel complesso
max_no_atoms    = int(params["max_no_atoms"])    # max numero di atomi nel complesso
n_max           = int(params["soap_n_max"])      # descrittore soap: n_max
l_max           = int(params["soap_l_max"])      # descrittore soap: l_max
cutoff          = float(params["cutoff"])        # descrittore soap: cutoff calcolo distr. radiale
TEST_SIZE       = float(params["test_size"])     # percentuale separazione test-train
target_variance = float(params["pca_target_variance"]) # varianza spiegata totale

if params["use_balanced_data"].lower() == "yes":       # può servire per dei test bilanciare le classi alla meno numerosa
   performe_data_balancing = True
else:
   performe_data_balancing = False

if params["data_already_extracted"].lower() == "yes":  # Se il dataset è stato già creato
   performe_extraction    = False
else:
    performe_extraction   = True
  
if params["convert_in_lowdim"].lower() == "yes":       # per test su dati PCA invece che interi 
   use_pca_data           = True
else: 
    use_pca_data          = False 
    
target_spin_values = [spin_val_1, spin_val_2] # array con i valori di spin target 

# ----------------------------------------------------------------------
#                      Directory setup
# ----------------------------------------------------------------------
OUT_DIR_DA = "../results/data_analysis_SOAP"
OUT_DIR_SVM_RFC = "../results/spin_classif_SVM_RFC"

if performe_data_balancing:
    if use_pca_data:
       OUT_DIR_ML      = "../results/spin_classif_ML_balanced_PCA"
    else:  
       OUT_DIR_ML      = "../results/spin_classif_ML_balanced_noPCA"
else:
    if use_pca_data:
       OUT_DIR_ML      = "../results/spin_classif_ML_unbalanced_PCA"
    else:   
       OUT_DIR_ML      = "../results/spin_classif_ML_unbalanced_noPCA"
os.makedirs(OUT_DIR_DA, exist_ok=True)
os.makedirs(OUT_DIR_ML, exist_ok=True)
os.makedirs(OUT_DIR_SVM_RFC, exist_ok=True)

# ----------------------------------------------------------------------
#                           get data
# ----------------------------------------------------------------------
# se i dati sono già disponibili, ovvero non è la prima volta che testo il modello su un dato ione, non sto a crearer tutto il dataset
# !!!!!!!!!!!!!!!!!!!!! Important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# data are extracted from omol25 dataset. the path is indicated in ./Data/extract/create_data.py 

if performe_extraction:
    print("Extracting data from database...")
    target_single_ion(target_spin_values, target_coord_no, target_metal_z)
   
    soap_features, y_spin, ligand_count_dict = get_soap_features(
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
    X = loaded["soap_features"].astype(np.float32) # vettore di input
    y_spin = loaded["y_spin"].astype(int).ravel()  # label
      
    print("Data extraction completed.")
    print("SOAP features shape:", X.shape)
    print("y_spin shape:", y_spin.shape)

    with open('./saved_lig_counts.pkl', 'rb') as f:
        ligand_count_dict = pickle.load(f)

# -----------------   Data balancing (if requested)   -----------
# Se richiesto, mi torna comodo che sia la prima cosa da fare
if performe_data_balancing:
   print("Performing data balancing pre MLP...")
   X, y_spin = get_balanced_data(X, y_spin, spin_val_1, spin_val_2)
   print("Scaling and splitting data after balancing dataset...")
   X_train, X_test, y_train_spin, y_test_spin, train_index = scale_and_split_data(X, y_spin, TEST_SIZE, RANDOM_STATE, target_spin_values)  
else: 
   # Preprocessing:  normalize data using mean value and split it in test and train set. ./Preprocessing/Scale_split.py
   print("Scaling and splitting data...")
   X_train, X_test, y_train_spin, y_test_spin, train_index = scale_and_split_data(X, y_spin, TEST_SIZE, RANDOM_STATE, target_spin_values)

##         ---------------------     Analisi dati   ---------------------          ##
## Analisi dati con PCA e t-SNE, k-means clustering e istogrammi. Complete info in ##
## function analyze_data in Data_analysis/Data_analyis_SOAP.py                     ##
## ------------------------------------------------------------------------------- ##

print("Performing data analysis...")
analyze_data(X_train, X_test, y_train_spin, y_test_spin, train_index, ligand_count_dict, target_variance, OUT_DIR_DA)
print("Data analysis completed. Everything saved in", OUT_DIR_DA)

##      -------------    SVM and RFC ML classification       ---------------       ##
## algorithm svm and random forest classifier for spin-classification on soap data ##
## svm uses both linear and gaussian kernel (you have to change in the code)       ##
##  -----------------------------------------------------------------------------  ##

NON_NN(X_train, y_train_spin, X_test, y_test_spin, OUT_DIR_SVM_RFC)
print("SVM and RFC classification completed. Everything saved in", OUT_DIR_SVM_RFC)

## ---------------       MLP classification       ---------------- ##
## Multi Layer Perceptron for spin-classification on soap data     ## 
## salva tutte le immagini in ../results/spin_classif_ML           ##
## Binary classification. The function is quite general. Adaptable ##
## --------------------------------------------------------------- ##


if use_pca_data == True: # carico i dati pca creati prima, normalizzati ed eventualmente bilanciati! 
    loaded2      = np.load("Soap_data_PCA.npz")
    X_train      = loaded2["X_train_pca"].astype(np.float32) # vettore di input
    X_test       = loaded2["X_test_pca"].astype(np.float32) # vettore di
    y_train_spin  = loaded2["y_spin_train"].astype(int).ravel()  # label
    y_test_spin = loaded2["y_spin_test"].astype(int).ravel()  # label


Multi_Layer_Perceptron(X_train, y_train_spin, X_test, y_test_spin, OUT_DIR_ML)
print("MLP classification completed. Everything saved in", OUT_DIR_ML)

print("!! =) End of Program =) !!")

final_image()
