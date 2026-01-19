import numpy as np


def get_balanced_data(X, y_spin, spin_val_1, spin_val_2):
    spin_val1_indices = np.where(y_spin == spin_val_1)[0] # trovo un vettore 1d di indici 
    spin_val2_indices = np.where(y_spin == spin_val_2)[0]
    np.random.seed(42)
    if len(spin_val1_indices) < len(spin_val2_indices): # bilancio alla classe più numerosa
      class_spin_val1 = spin_val1_indices
      class_spin_val2 = np.random.choice(spin_val2_indices, len(class_spin_val1) , replace=False)
    else:
      class_spin_val2 = spin_val2_indices
      class_spin_val1 = np.random.choice(spin_val1_indices, len(class_spin_val2) , replace=False)
    selected_idx = np.concatenate([class_spin_val1, class_spin_val2])

    X = X[selected_idx]

    y_spin = y_spin[selected_idx]

    return X,y_spin
