# Analysis of Pre-existing Conditions in Patients with COVID-19: Application of Neural Networks in Patient Segmentation

We reproduce the paper 'Análise de pré-condições existentes em pacientes com COVID-19:
aplicação de redes neurais na segmentação de pacientes', published at the 15th Brazilian Symposium on Intelligent Automation – SBAI 2021. Additionally, we identify and implement a few improvements, enhancing our performance.

## Project Overview

The objective is to improve predictive performance, model validation rigor, and comparative benchmarking across multiple machine learning approaches.

### Proposed Improvements

#### A. Optimization of the Feature Space

Unlike the original method, we applied scaling to continuous variables.

This prevents high-magnitude features from dominating the gradients and improves convergence when using activation functions such as Tanh.

#### B. Increased Rigor in Machine Learning Model Validation

We introduced Grid Search for systematic hyperparameter tuning. While the original approach tested kernels heuristically, our method: (i) uses a dedicated validation set and (ii) tunes the regularization parameter systematically. This leads to more reliable and generalizable models.

#### C. Extended Benchmarking

We expanded the comparison beyond Support Vector Machines. We include Random Forest, Gradient Boosting, Logistic Regression, kNN and Naive Bayes.

### Results

TBD
