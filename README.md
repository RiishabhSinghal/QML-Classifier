## Quantum Machine Learning for Fraud Detection using Variational Quantum Classifier (VQC)
####📌 Objective

The objective of this project is to design and implement a Hybrid Quantum Machine Learning (QML) pipeline for fraud detection and compare its performance with classical machine learning models using robust evaluation metrics such as ROC-AUC and F2-score on an imbalanced dataset.

#### 🧠 Project Overview

This project explores the application of Variational Quantum Circuits (VQC) for binary classification in financial fraud detection. Due to the highly imbalanced nature of fraud datasets and limited qubit resources in current quantum systems (NISQ era), the workflow integrates:

Advanced data preprocessing

Feature selection (≤ 4 features for qubit compatibility)

Classical baseline benchmarking

Hybrid Quantum-Classical training

The quantum model is implemented using Qiskit’s Sampler-based QNN architecture with a hardware-efficient ansatz.

#### 📊 Dataset & Problem Characteristics

Binary classification: Fraud (1) vs Non-Fraud (0)

Highly imbalanced dataset (fraud ≪ non-fraud)

Strong feature skewness in transactional distance and price ratios

Large dataset (~100,000 samples)

Class Distribution

The dataset shows a severe imbalance with a dominant non-fraud class, which justifies the use of:

F2-score (recall-focused)

ROC-AUC (threshold-independent)

#### 🔍 Data Analysis & Preprocessing
1. Distribution Analysis

The following insights were observed from feature plots:

distance_from_home: Highly right-skewed with extreme outliers

distance_from_last_transaction: Heavy-tailed distribution

ratio_to_median_purchase_price: Positively skewed with long tail

This indicates the need for transformation and robust preprocessing before model training.

2. Preprocessing Pipeline

The preprocessing steps were carefully designed for both classical and quantum compatibility:

✔ Outlier Removal (Before Transformation)

IQR-based filtering applied on continuous variables

Removed extreme anomalies to stabilize training

✔ Skewness Correction

Applied Yeo-Johnson Power Transformation to handle non-normal distributions

Suitable for zero and positive values

✔ Feature Scaling (QML-Ready)

MinMax scaling to range [-π, π]

Ensures valid angle encoding in quantum circuits

✔ Feature Selection (≤ 4 Features)

Since quantum circuits scale exponentially with features (qubits), Mutual Information was used to select the top 4 most informative features:

ratio_to_median_purchase_price

online_order

repeat_retailer

used_chip

#### ⚛️ Quantum Circuit Design
Feature Map (Data Encoding)

Angle encoding using Ry(xᵢ) rotations

Each feature mapped to a qubit rotation

Enables efficient embedding of classical data into quantum states

Ansatz (Variational Layer)

The circuit uses a hardware-efficient ansatz consisting of:

Parameterized Ry and Rz rotations

Entanglement via CNOT gates (ring topology)

Trainable parameter vector θ

Circuit Architecture (4-Qubit VQC)

Initial layer: Feature encoding using Ry(x[i])

Entanglement layer: CNOT chain (ring structure)

Variational layer: Ry(θ) and Rz(θ) rotations

This design:

Balances expressibility and trainability

Avoids barren plateaus

Suitable for NISQ simulators

#### 🏋️ Training Pipeline
Hybrid Quantum-Classical Workflow

Classical preprocessing and feature selection

Conversion to NumPy arrays (QML compatible)

Subsampling (to control runtime)

Quantum model training using COBYLA optimizer

Evaluation on full test set

Model Components

Quantum Neural Network: SamplerQNN

Optimizer: COBYLA (gradient-free, stable for noisy landscapes)

Output: Binary classification (fraud detection)

Due to simulator constraints, a representative subset of training data was used for QML training while evaluation was done on the full test set.

#### 🤖 Classical Baseline Models

To establish a benchmark, three classical models were trained on the same selected features:

Logistic Regression (class-weight balanced)

Random Forest (ensemble, depth-controlled)

Neural Network (MLP with hidden layer)

These models provide a realistic performance reference for the quantum classifier.

#### 📈 Comparative Analysis
Model	AUC-ROC	F2-Score
Logistic Regression	High	Good
Random Forest	Very High	Excellent
Neural Network	Very High	Strong
Quantum VQC	Moderate	Competitive (Recall-focused)
Key Observations

Classical models outperform VQC in raw accuracy due to data scale

VQC remains competitive in recall-focused metrics (F2-score)

Quantum models are computationally expensive on simulators

Feature reduction is critical for feasible QML training

#### ⏱️ Runtime Considerations

Full dataset (~100k samples) is not practical for quantum simulation

Training subset sizes:

200 samples → ~5–10 minutes

1,000 samples → ~30–60 minutes

20,000 samples → several hours (simulator bottleneck)

GPU acceleration (T4) provides minimal speedup since statevector simulation is mostly CPU-bound.

#### 🧪 Evaluation Metrics

Given the class imbalance, the following metrics were used:

ROC-AUC: Overall model discrimination ability

F2-Score: Emphasizes recall (important for fraud detection)

Accuracy (secondary metric)

🧰 Tech Stack

Python

Qiskit & Qiskit Machine Learning

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

#### 🚀 Key Contributions

Designed a full end-to-end Hybrid Quantum ML pipeline

Implemented a custom 4-qubit Variational Quantum Classifier

Optimized preprocessing for quantum-compatible feature encoding

Conducted fair benchmarking against classical ML models

Analyzed scalability limits of QML on large real-world datasets

#### ⚠️ Limitations

Quantum simulation is computationally expensive

Limited qubit count restricts feature dimensionality

Performance depends heavily on optimizer and circuit depth

Not yet deployable on real quantum hardware at scale

🔮 Future Work

Experiment with deeper ansatz and different entanglement patterns

Use quantum kernels (QSVM) for comparison

Train on real quantum hardware backends

Apply advanced imbalance techniques (SMOTE, focal loss)

Hyperparameter tuning of circuit depth and learning strategy

📎 Conclusion

This project demonstrates the practical integration of Quantum Machine Learning in a real-world fraud detection task. While classical models currently outperform VQC in large-scale scenarios, the hybrid approach shows promising potential, especially under constrained feature spaces and NISQ-era quantum limitations. The study highlights both the capabilities and current bottlenecks of quantum classifiers in applied machine learning.
