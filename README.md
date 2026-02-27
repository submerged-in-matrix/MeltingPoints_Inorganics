# Physics-Informed Neural Network for 2D Linear Elasticity

## Problem Setup

- 1m × 1m steel plate  
- Bottom boundary fixed  
- Uniform 4 N vertical load  
- Governing equations: Navier–Cauchy + Hooke’s Law  

The goal was to approximate FEM-quality solutions without dense supervised labels.

---

## Methodology

- TensorFlow-based PINN
- Latin Hypercube collocation sampling:
  - 1,000 interior points  
  - 50 boundary points per edge  
- Dirichlet + Neumann boundary enforcement
- 27-configuration hyperparameter sweep

Loss terms combined:
- PDE residual  
- Boundary residual  
- Stress consistency  

---

## Best Architecture

- 6 hidden layers × 15 neurons  
- Glorot initialization  
- Dropout = 0.3  
- Validation loss ≈ 2.22  

---

## Outcome

- Smooth displacement fields  
- Physically coherent stress distribution  
- Demonstrated PINN viability for classical elasticity problems

---

## Extensions

- Adaptive collocation sampling  
- Nonlinear material models
