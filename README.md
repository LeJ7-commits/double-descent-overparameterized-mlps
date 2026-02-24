# Double Descent in Overparameterized MLPs  
A teacher–student regression study of interpolation and modern generalization behavior.

-------

## Overview

This project investigates the **double descent phenomenon** in overparameterized neural networks using a controlled teacher–student regression framework.

We:

- Simulate data from a fixed nonlinear teacher MLP.
- Train student MLPs of increasing width.
- Analyze train and test behavior across the interpolation threshold.
- Study how noise, regularization, and optimization affect generalization.

The goal is to understand how modern deep networks behave beyond the classical bias–variance tradeoff.

---

## Double Descent Curve

> (Main figure will be inserted after experiments are finalized.)



---

## Experimental Setup

**Data Generating Process**
- Input dimension: 20  
- Inputs sampled from: 𝒩(0, I)  
- Teacher network: 2-layer MLP (hidden width 32, frozen random weights)  
- Output: teacher(x) + Gaussian noise  

**Student Networks**
- Architecture: 2 hidden layers, ReLU activation  
- Width sweep: 4 → 1024  
- Loss: Mean Squared Error (MSE)  
- Optimizer (baseline): Adam  

**Dataset Sizes**
- Training samples: 1000  
- Validation samples: 1000  
- Test samples: 10000  

**Evaluation**
- Train MSE  
- Test MSE  
- Generalization gap  
- Parameter-to-sample ratio (p/n)

All experiments use deterministic seed control for reproducibility.

---

## Key Research Questions

- Does test error exhibit double descent as model width increases?
- How does label noise affect the interpolation peak?
- How does weight decay shift the generalization curve?
- Do optimizers (SGD vs Adam) influence behavior near interpolation?

---

## Notebooks

Run in order:

1. `01_teacher_student_double_descent.ipynb`
2. `02_regularization_shift.ipynb`
3. `03_optimizer_effects.ipynb`

Each notebook corresponds to one experimental component of the study.

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt


