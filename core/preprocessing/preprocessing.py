from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_and_split_data(X, y_spin, TEST_SIZE, RANDOM_STATE, target_spin_values):
    """
    Scales the features using StandardScaler and splits the dataset into training and testing sets.

    Parameters:
    X (array-like)      : soap matrix 
    y_spin (array-like) :Target vector for spin classification.
    TEST_SIZE (float)   : proportion of splitting in test-train 
    RANDOM_STATE (int)  : Random seed for reproducibility.

    Returns:
    X_test       : Testing feature matrix for models.
    X_train      : Training feature matrix for models.
    y_train_spin : Training target vector.
    y_test_spin  : Testing target vector.

"""
############################################################################################################# 
###                     prima divido in test e train, poi normalizzo                                      ###
###                ------------------ from stack overflow: -----------------------                        ### 
### Don't forget that testing data points represent real-world data.                                      ###  
### Feature normalization (or data standardization) of the explanatory (or predictor) variables is a      ###
### technique used to center and normalise the data by subtracting the mean and dividing by the variance. ###
### If you take the mean and variance of the whole dataset you'll be introducing future information into  ###
###the training explanatory variables (i.e. the mean and variance).                                       ### 
#############################################################################################################

    # Binarizzazione
    # la classe positiva (1) è il primo valore in input in input.txt (viceversa lo 0)
    # split train/test
    # from S.O.F :
    # This stratify parameter makes a split so that the proportion of values in the sample produced
    # will be the same as the proportion of values provided by parameter stratify. If the original data
    # contains 30 % of 0s and 70 % of 1s, then in the train and test set the random split ensure you mantain the 
    # same proportions
 
    y_spin = (y_spin == target_spin_values[0]).astype(int)
    df_copy = pd.DataFrame(X)

    # split train/test
    X_train, X_test, y_train_spin, y_test_spin = train_test_split( 
        df_copy,
        y_spin,                                                          
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_spin
    )

    # indici originali del train
    train_index = X_train.index.values
    X_train     = X_train.values
    X_test      = X_test.values
    

    # standardizzazione
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

    return X_train, X_test, y_train_spin, y_test_spin, train_index
