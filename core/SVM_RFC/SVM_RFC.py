import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def NON_NN(X_train, y_train, X_test, y_test, OUT_DIR):
    """ SVM and Random Forest Classifier for spin classification on SOAP data.
    the function trains and evalueate SVM with gaussian kernel/linear (it is sufficient to change it at line 20)
    and Random Forest Classifier on the provided training and testing data."""

    #    SVM gaussian kernel/linear 
    svm_clf = Pipeline([('svc', SVC(kernel='rbf', C=1.0, gamma='scale',
                    class_weight='balanced'))])
    # kernel: ; C : parametro che valuta quanto sono penalizzati gli errori di class.
    # gamma : per gaussiano, determina l'ampiezza della gaussiana del Kernel, 'scale'
    # adatata ai dati e al numero di feature. calss weight: pesa gl errori in base alla distribuzione delle classi
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    # metriche di valutazione
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Linear SVM test — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

    # salva metriche base
    with open(os.path.join(OUT_DIR, "svm_metrics_rbf.txt"), "w") as fh:
        fh.write("Model: LinearSVC\n")
        fh.write(f"Accuracy:  {acc:.6f}\n")
        fh.write(f"Precision: {prec:.6f}\n")
        fh.write(f"Recall:    {rec:.6f}\n")
        fh.write(f"F1-score:  {f1:.6f}\n")

    # confusion matrix (normalizzata)
    # mappa sempre coerente anche per altre classificazioni oltre a Fe(II)

    label_mapping= {0 : 'Low spin', 1 : 'High spin'}
    y_all = np.concatenate([y_train, y_test])
    classes = np.unique(y_all)
    labels = [label_mapping[c] for c in classes] # ha bisogno di una lista, non di un dizionario 
    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, cmap='Blues', values_format=".2f")
    plt.title("Confusion Matrix (normalized) - kernel rbf SVM")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix_linearSVM.png"))
    plt.close(fig)

#Random Forest Algorithm 

    rdn_clf = Pipeline([
        ('rf_clf', RandomForestClassifier(n_estimators=500, max_depth=10,
                                          min_samples_leaf=3, random_state=42,
                                          n_jobs=-1, class_weight='balanced'))
    ])

    # n_estimators: quanti alberi considero; max_depth : massima profondità di un albero
    # min_samples_leaf : numero minimo di campioni che servono per definire una 'foglia'
    # n_jobs = -1 : distribuisce in automatico il carico su tutti i core che trova sulla macchina

    rdn_clf.fit(X_train, y_train)
    y_pred_rdn = rdn_clf.predict(X_test)

    acc_rdn = accuracy_score(y_test, y_pred_rdn)
    prec_rdn = precision_score(y_test, y_pred_rdn, average='weighted', zero_division=0)
    rec_rdn = recall_score(y_test, y_pred_rdn, average='weighted', zero_division=0)
    f1_rdn = f1_score(y_test, y_pred_rdn, average='weighted', zero_division=0)
    print(f"Random Forest test — acc: {acc_rdn:.4f}, prec: {prec_rdn:.4f}, rec: {rec_rdn:.4f}, f1: {f1_rdn:.4f}")

    # salva metriche Random Forest
    with open(os.path.join(OUT_DIR, "rf_metrics.txt"), "w") as fh:
        fh.write("Model: RandomForest\n")
        fh.write(f"Accuracy:  {acc_rdn:.6f}\n")
        fh.write(f"Precision: {prec_rdn:.6f}\n")
        fh.write(f"Recall:    {rec_rdn:.6f}\n")
        fh.write(f"F1-score:  {f1_rdn:.6f}\n")

    # confusion matrix Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rdn, labels=classes, normalize='true')
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_mapping)
    fig_rf, ax_rf = plt.subplots(figsize=(6,5))
    disp_rf.plot(ax=ax_rf, cmap='Greens', values_format=".2f")
    plt.title("Confusion Matrix (normalized) - Random Forest")
    plt.tight_layout()
    fig_rf.savefig(os.path.join(OUT_DIR, "confusion_matrix_randomforest.png"))
    plt.close(fig_rf)   

    rf_model = rdn_clf.named_steps['rf_clf']  # estrai modello

    # feature importance 
    importances = rf_model.feature_importances_
    with open(os.path.join(OUT_DIR, "rf_feature_importance.txt"), "w") as fh:
      fh.write("Feature importance (Gini) per Random Forest\n")
      for i, imp in enumerate(importances):
        fh.write(f"Feature {i+1}: {imp:.6f}\n")

    plt.figure(figsize=(10,4))
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rf_feature_importance.png"))
    plt.close()
