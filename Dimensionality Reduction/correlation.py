import numpy as np
import pandas as pd

# Drop the 'label' column and reduce memory usage by converting to float32
df = df.drop(columns=['label']).astype(np.float32)
# Store the 'label' column in a separate variable
labels = df['label']

# Define the chunk size for processing features
chunk_size = 5000
# Get the total number of features
n_features = df.shape[1]
# Initialize an empty set to store columns to be dropped
to_drop = set()

# Iterate through the features in chunks
for start in range(0, n_features, chunk_size):
    # Determine the end index of the current chunk
    end = min(start + chunk_size, n_features)

    # Calculate the absolute correlation matrix for the current chunk of features
    corr_matrix_chunk = df.iloc[:, start:end].corr().abs()
    # Get the upper triangle of the correlation matrix (excluding the diagonal)
    upper_chunk = np.triu(corr_matrix_chunk, k=1)

    # Iterate through the columns in the current chunk
    for col_idx, col in enumerate(df.columns[start:end]):
        # If any correlation in the upper triangle for the current column is greater than 0.95
        if any(upper_chunk[:, col_idx] > 0.95):
            # Add the column name to the set of columns to drop
            to_drop.add(col)

# Create a new DataFrame by dropping the highly correlated columns
df_reduced = df.drop(columns=to_drop)
# Add the 'label' column back to the reduced DataFrame
df_reduced['label'] = labels.values