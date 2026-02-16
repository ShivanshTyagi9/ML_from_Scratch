import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate the dataset
X, y = make_classification(
    n_samples=500,       # Number of rows
    n_features=2,        # 2D for easy visualization
    n_redundant=0, 
    n_clusters_per_class=1, 
    class_sep=1.5,       # Higher = easier to separate
    random_state=42
)

# 2. Convert to a DataFrame for readability
df = pd.DataFrame(X, columns=['X', 'Y'])
df['Label'] = y

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df.head())