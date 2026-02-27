# Two-Level Stacked Ensemble for Melting Point Prediction

## Overview

This project implements a custom **two-level multi-class stacking ensemble** to predict melting points of organic compounds.

It reproduces and extends a published ensemble methodology (DOI: https://link.springer.com/article/10.1134/S1995080223010341) on a distinct dataset (for hallides).

---

## Dataset

- ~3,041 organic compounds  
- Source: Citrination  
- Target: Melting point  

---

## Feature Engineering

- SMILES parsing via RDKit  
- Custom bond-count descriptors  
- SHAP-guided feature refinement  

---

## Modeling Framework

Base learners:
- Random Forest  
- XGBoost  
- LightGBM  
- MLP  

Meta-learner trained on cross-validated predictions.

Custom stacking order and weighting were engineered manually.

---

## Results

- R² ≈ **0.83**  
- ~4% improvement over baseline stacking  
- Ensemble diversity contributed more than individual model complexity
