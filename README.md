# Analysis of Pre-existing Conditions in Patients with COVID-19: Application of Neural Networks in Patient Segmentation

We reproduce the paper 'Análise de pré-condições existentes em pacientes com COVID-19:
aplicação de redes neurais na segmentação de pacientes', published at the 15th Brazilian Symposium on Intelligent Automation – SBAI 2021. Additionally, we identify and implement a few improvements, enhancing our performance.

A more detailed explanation of improvements is underway. For the moment, results may be seen empirically by running the training pipelines with the author's data gathering pipelines (`prepare_dataframes_deprecated.py`) vs ours (`prepare_dataframes.py`).

## Project Overview

The objective is to improve predictive performance, model validation rigor, and comparative benchmarking across multiple machine learning approaches.

### Proposed Improvements

#### A. Optimization of the Feature Space

Unlike the original method, we applied scaling to continuous variables. This prevents high-magnitude features from dominating the gradients and improves convergence when using activation functions such as Tanh.

#### B. Increased Rigor in Machine Learning Model Validation

We introduced Grid Search for systematic hyperparameter tuning. While the original approach tested kernels heuristically, our method: (i) uses a dedicated validation set and (ii) tunes the regularization parameter systematically. This leads to more reliable and generalizable models.

#### C. Extended Benchmarking

We expanded the comparison beyond Support Vector Machines. We include Random Forest, Gradient Boosting, Logistic Regression, kNN and Naive Bayes.

### Results

Of the **Machine Learning** models optimizing for recall, **Gradient Boosting** performs the best, with **94.91% recall**. A close second is **Random Forest**, with **94.80%** and third is **kNN**, with **93.70%**. SVM, Logistic Regression and Naive Bayes perform more poorly, with 90.70%, 90.04% and 88.04%, respectively. However, **all models achieve accuracy > 87% and specificity > 84%**, which is **higher than the best scores presented by the best models in the paper**. This shows that the mere pre-processing step which the authors overlooked makes a significant difference in the results.

The **MLPs** perform in a **much more similar fashion** when the data is scaled correctly. Both MLPs early-stop at ~90 epochs with **accuracy of 89.21% and recall of 94.78%**, a **near-2% improvement** from the original paper. **This is difference is not statistically significant from the Random Forest algorithm, which is significantly more explainable.**
