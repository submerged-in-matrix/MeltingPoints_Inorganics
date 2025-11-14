# Melting Point Prediction of Organic Compounds
# Two-Level Ensemble Method (Inspired by Kiselyova et al. and Senko et al.) __ if needed for featurization: matminer mendeleev

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np

import os
import random