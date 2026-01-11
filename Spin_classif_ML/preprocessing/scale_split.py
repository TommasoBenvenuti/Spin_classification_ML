
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scale_and_split_data(X, y_spin, TEST_SIZE, RANDOM_STATE):
    """
    Scales the features using StandardScaler and splits the dataset into training and testing sets.

    Parameters:
    X (array-like): soap matrix 
    y_spin (array-like): Target vector for spin classification.
    TEST_SIZE (float): proportion of splitting in test-train 
    RANDOM_STATE (int): Random seed for reproducibility.

    Returns:
    X_scaled : Scaled feature matrix for data analysis.
    X_test : Testing feature matrix for ML model.
    X_train : Training feature matrix for ML model.
    y_train_spin : Training target vector.
    y_test_spin : Testing target vector.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train_spin, y_test_spin = train_test_split(
        X_scaled, y_spin, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_spin
    )
    print("Train/test sizes:", X_train.shape[0], X_test.shape[0])
    return X_scaled, X_train, X_test, y_train_spin, y_test_spin