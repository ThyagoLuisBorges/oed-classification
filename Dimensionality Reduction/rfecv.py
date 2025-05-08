import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Logistic Regression
log_reg = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5)

# Feature selection using RFECV
rfe_cv = RFECV(estimator=log_reg, step=5, cv=cv, scoring='accuracy', n_jobs=-1)
X_rfe = rfe_cv.fit_transform(X_scaled, y)

# Get selected feature names
selected_feature_names = X.columns[rfe_cv.support_]

# Create new DataFrame with selected features and label
df_rfe = pd.DataFrame(X_rfe, columns=selected_feature_names)
df_rfe['label'] = y.values