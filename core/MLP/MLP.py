import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disabilita GPU 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disabilita GPU se non richiesta

def Multi_Layer_Perceptron(X_train, y_train, X_test, y_test, OUT_DIR):

    """
    Addestra e valuta un modello MLP sui dati SOAP per la classificazione dello stato di spin.
    
    Args:
        X_train: Dati di addestramento 
        y_train: Etichette di addestramento (eg stati di spin)
        X_test: Dati di test
        y_test: Etichette di test (eg stati di spin)
        OUT_DIR: Cartella di output per salvare i risultati
    """
    # -------------------------
    # Modello MLP
    # -------------------------

    reg = l2(1e-4)

    model = Sequential([Flatten(input_shape=(X_train.shape[1], ))])
    model.add(Dense(32, activation='relu', kernel_regularizer=reg, kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_regularizer=reg, kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Compilazione e training
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing = 0.05),
        metrics=['accuracy']
    )

    hist = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # -------------------------
    # Plot accuracy/loss
    # -------------------------
    plt.figure()
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(OUT_DIR, "training_validation_accuracy.png"))

    plt.figure()
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(OUT_DIR, "training_validation_loss.png"))
    plt.close()

    # -------------------------
    # Confusion Matrix
    # ----------------------:---
    # sono binari 0/1 quindi si dovrebbe poter fare
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
    label_mapping = {0 :'Low spin', 1 : 'High spin'}
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping)
    disp.plot(cmap='Blues')
    plt.title("Normalized Confusion Matrix")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close()

    # -------------------------
    # ROC curve + AUC
    # -------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
    plt.close()
 
    # -------------------------
    model.save(os.path.join(OUT_DIR, "mlp_model_spin_classification.keras"))

