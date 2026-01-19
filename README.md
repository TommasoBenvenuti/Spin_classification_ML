# Spin Classification in Metal Complexes Using Data-Driven Techniques

High-performance computing (HPC) techniques for spin-state classification in inorganic complexes are computationally demanding. Recently, machine learning approaches have emerged as efficient alternatives for this task ([ACS JPCA exemple ](https://pubs.acs.org/doi/10.1021/acs.jpca.0c01458)).


This repository contains a Python program to classify spin states in **first-row transition metal complexes** using machine learning techniques.

---

### 1. Extracting and Analyzing Data
- Reads an input specifying:
  - Target metal ion
  - Spin states
  - Coordination number (4, 5, or 6)
- Builds a dataset 'distilling' **OMOL25 dataset** (~4 million structures) [OMOL25](https://huggingface.co/facebook/OMol25)
- Converts molecular structures into **SOAP descriptors** using the [DSCRIBE](https://singroup.github.io/dscribe/latest/) library
- Preprocesses data:
  - Normalizing SOAP data
  - Splitting into training and test sets 
- Performs exploratory data analysis:
  - **PCA** 
  - **t-SNE**
  - **k-Means clustering**
- Generates plots colored by spin state and dominant atom in the first coordination sphere

[t-SNE plot colored by spin state (HS is label with 1, LS with 0)](tsne_spin_state.png) 
[t-SNE plot colored by dominant atom in first sphere](t-SNE_legante.png)

### 2. SVM and Random Forest Benchmark
- Implements **Support Vector Machine (SVM)** classification:
  - Linear SVM
  - Kernel SVM
- Implements **Random Forest** classifier

### 3. Multi-Layer Perceptron (MLP)
- Performs spin-state classification using an **MLP**
- Highly regularized architecture due to small datasets
- Can work on **PCA-reduced data**
- Supports **class-balanced training** (undersampling majority class)

---

## Usage
- install dependencies in requirements.txt 
- Change input parameters in input.txt
- then simply: python3 main.py 
