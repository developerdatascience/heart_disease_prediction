import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Data Collection and Data Processing
heart_data = pd.read_csv("input/heart_disease_data.csv")

print(heart_data.head())

