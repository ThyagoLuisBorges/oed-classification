import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Check if feature and target shapes match
assert X.shape[0] == y.shape[0], "Mismatch: X and y must have the same number of rows!"

# Stratified k-fold cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize array for feature importance
feature_importance_values = np.zeros(X.shape[1])

# Iterate through folds
for train_index, test_index in skf.split(X, y):
    # Split data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Accumulate feature importances
    feature_importance_values += rf.feature_importances_

# Average feature importances across folds
feature_importance_values /= skf.get_n_splits()

# Create Series of feature importances
feature_importance = pd.Series(feature_importance_values, index=X.columns)

# Select important features
selected_features = feature_importance[feature_importance > (2 * feature_importance.mean())].index

# Create new DataFrame with selected features and target
df_reduced_rf = pd.concat([X[selected_features], y.reset_index(drop=True)], axis=1)