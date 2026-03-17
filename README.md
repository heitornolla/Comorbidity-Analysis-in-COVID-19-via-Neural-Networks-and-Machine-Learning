# Re-evaluating COVID-19 Mortality Prediction  

This repository presents a **re-analysis and reimplementation** of the 2021 study:

> *“Analysis of Pre-existing Conditions in COVID-19 Patients: Application of Neural Networks in Patient Segmentation.” (Análise de pré-condições existentes em pacientes com COVID-19: aplicação de redes neurais na segmentação de pacientes.)

The goal of this project is to demonstrate how **machine learning design choices can artificially inflate performance metrics** and ultimately **reduce the clinical usefulness of predictive models in healthcare**.

By reconstructing the original pipeline and correcting its methodological flaws, this repository highlights **common pitfalls in clinical ML research** and proposes a more robust and interpretable modeling approach.

---

## Methodological Issues Identified

The re-analysis of the original study uncovered several flaws that significantly undermine the validity and real-world applicability of the reported results.

### 1. Improper Dataset Balancing

The dataset was balanced via **undersampling before the train/test split**.

This introduces **statistical leakage**, because the test set no longer reflects the true distribution of the population. As a result, performance metrics become **optimistically biased and clinically misleading**.

---

### 2. Data Discarding

To force a **1:1 class balance**, approximately **87% of the original data was discarded**.

Consequences include:

- Increased model variance
- Loss of valuable information
- Failure to model the **real-world class imbalance typical of clinical datasets**

---

### 3. Target Leakage

The original model used variables such as:

- `icu`
- `intubed`
- `patient_type`

to predict **mortality**.

However, these variables describe **late-stage clinical interventions**, meaning the model effectively learns to detect **patients already in critical condition**.

---

### 4. Poorly Defined Clinical Objective

The model mixes **pre-existing comorbidities** with **current clinical status**.

This makes the model unsuitable for **early risk assessment**, since some predictors only become available **after hospitalization**.

---

### 5. Pipeline Fragility

The original pipeline lacked several standard ML practices:

- No **cross-validation**
- No **baseline comparisons**
- No **systematic hyperparameter tuning**
- Over-reliance on a single neural network architecture

---

## Reimplementation Approach

We correct the mentioned pitfalls, emphasizing explainability in models and metrics such as **PR-AUC** and **ROC-AUC**.

# Disclaimer

This project is intended **for research and methodological analysis only** and is **not designed for clinical decision-making**.
