# Spacecraft Telemetry Anomaly Detection Platform

## Overview

This project demonstrates an end-to-end machine learning and MLOps workflow for detecting anomalous spacecraft telemetry using the NASA SMAP/MSL telemetry dataset.

The objective is to identify abnormal spacecraft behavior from high-dimensional telemetry streams before mission-impacting failures occur. The project combines machine learning experimentation, anomaly detection techniques, experiment tracking, and Azure Machine Learning infrastructure to simulate a production-grade aerospace analytics solution.

---

## Business Problem

Modern spacecraft generate thousands of telemetry measurements that engineers must continuously monitor to identify abnormal behavior.

Traditional rule-based monitoring systems can struggle to detect novel failure modes and complex telemetry patterns.

This project explores the use of unsupervised machine learning to automatically identify anomalous telemetry behavior and provide early warning indicators that could support mission operations and maintenance teams.

---

## Dataset

**Source:** NASA SMAP/MSL Telemetry Dataset

The dataset contains telemetry channels collected from:

* SMAP (Soil Moisture Active Passive Satellite)
* MSL (Mars Science Laboratory Rover)

Each telemetry channel includes:

* Historical training telemetry
* Test telemetry containing anomalies
* Ground-truth anomaly ranges
* Anomaly classifications

Example telemetry channel:

```text
P-1
Train Shape: (2872, 25)
Test Shape: (8505, 25)
```

---

# Machine Learning Layer

## Problem Formulation

This project treats anomaly detection as an unsupervised learning problem.

Training data is assumed to represent primarily normal spacecraft behavior while test data contains both normal and anomalous observations.

The objective is to identify anomalous telemetry patterns without requiring labeled anomaly examples during training.

---

## Feature Engineering

The telemetry data consists of:

* 25 telemetry features
* Thousands of sequential observations
* Multiple spacecraft subsystems

Features are standardized using:

```python
StandardScaler
```

to normalize sensor measurements before model training.

---

## Model

### Isolation Forest

The baseline anomaly detection model uses:

```python
IsolationForest
```

Isolation Forest is commonly used in industrial anomaly detection systems because it:

* Does not require labeled training anomalies
* Scales efficiently
* Works well with high-dimensional telemetry
* Provides interpretable anomaly scoring

---

## Experimentation

Multiple contamination thresholds were evaluated:

| Contamination |
| ------------- |
| 0.01          |
| 0.02          |
| 0.05          |
| 0.10          |
| 0.15          |

Model performance was evaluated using:

* Precision
* Recall
* F1 Score

Experiments were tracked using MLflow.

---

## Results

Current baseline results:

| Metric    | Score |
| --------- | ----- |
| Precision | 0.092 |
| Recall    | 0.161 |
| F1 Score  | 0.117 |

These results establish a baseline for future improvements including:

* Autoencoders
* LSTM-based anomaly detection
* Deep learning sequence models
* Hyperparameter optimization

---

# Cloud & MLOps Layer

## Azure Machine Learning

To simulate a production-grade machine learning environment, the project leverages Azure Machine Learning services.

Infrastructure created:

### Azure Resource Group

```text
rg-spacecraft-ml
```

### Azure Machine Learning Workspace

```text
spacecraft-ml-ws
```

### Azure Compute Cluster

```text
cpu-cluster
```

Configured with:

* Auto-scaling enabled
* Minimum nodes = 0
* Cost-optimized compute management

---

## Experiment Tracking

Experiments are tracked using MLflow.

Tracked metadata includes:

* Model type
* Hyperparameters
* Precision
* Recall
* F1 Score
* Training configuration

This enables reproducible experimentation and model comparison.

---

## Production-Oriented Workflow

The project follows a modern MLOps workflow:

```text
Telemetry Data
      ↓
Data Preprocessing
      ↓
Feature Scaling
      ↓
Isolation Forest Training
      ↓
MLflow Experiment Tracking
      ↓
Azure Machine Learning Compute
      ↓
Model Registry
      ↓
Online Endpoint Deployment
```

---

## Technologies

### Machine Learning

* Python
* NumPy
* Pandas
* Scikit-Learn
* Isolation Forest
* MLflow

### Cloud & MLOps

* Azure Machine Learning
* Azure Compute Clusters
* Azure Resource Groups
* Azure ML Workspaces
* MLflow Tracking

### Development

* Jupyter Notebook
* Git
* GitHub

---

## Future Improvements

Planned enhancements include:

* Azure ML Training Jobs
* Azure Model Registry
* Managed Online Endpoints
* CI/CD Integration
* Automated Retraining Pipelines
* Deep Learning-Based Anomaly Detection
* Real-Time Telemetry Streaming

---

## Key Takeaways

This project demonstrates both machine learning engineering and cloud MLOps capabilities:

* Building anomaly detection models for aerospace telemetry
* Experiment tracking and model evaluation
* Azure Machine Learning infrastructure deployment
* Cloud-based model training workflows
* Production-oriented machine learning architecture

The goal is not only to build an accurate anomaly detector, but to demonstrate the full lifecycle of deploying machine learning systems in a scalable, enterprise-ready environment.
