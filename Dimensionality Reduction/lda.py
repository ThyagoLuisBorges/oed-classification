from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

# Encode categorical labels to numerical
label_encoder = LabelEncoder()
y_rfe_encoded = label_encoder.fit_transform(df['label'])

# Separate features and encoded target
X_rfe = df.drop(columns=['label'])
y_rfe = y_rfe_encoded

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rfe)

# Apply Linear Discriminant Analysis for dimensionality reduction
lda = LDA(n_components=None)  # Use all possible components
X_lda = lda.fit_transform(X_scaled, y_rfe)

# Create DataFrame from LDA results
df_lda = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
# Add original labels back to the DataFrame
df_lda['label'] = label_encoder.inverse_transform(y_rfe)