import numpy as np


def get_balanced_data(X, y_spin, y_gap):
    spin_2_indices = np.where(y_spin == 2)[0]
    spin_6_indices = np.where(y_spin == 6)[0]
    np.random.seed(42)
    class_spin_2 = spin_2_indices
    class_spin_6 = np.random.choice(spin_6_indices, len(class_spin_2) , replace=False)
    selected_idx = np.concatenate([class_spin_2, class_spin_6])

    X = X[selected_idx]
    y_gap = y_gap[selected_idx]
    y_spin = y_spin[selected_idx]

    return X, y_gap, y_spin
