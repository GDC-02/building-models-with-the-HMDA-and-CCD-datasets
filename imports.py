# ============================================================================
# IMPORTS AND SETUP
# ============================================================================


print("=" * 50)

# Core Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV #only necessary for hyperparameters

# Fairness Libraries (AIF360)
import sys
try:
    import aif360
except ImportError:
    !pip install aif360

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.preprocessing import DisparateImpactRemover

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("All libraries imported successfully!")
print("Random state set to:", RANDOM_STATE)
