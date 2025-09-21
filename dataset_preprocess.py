print("This is small implementation of dataset preprocess")
# since the e-commerce dataset are quite large 
# this kaggle dataset is quite small as compared to them
# Cuz, i didn't find the dataset in such small scale of time

import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("D:\\Work\\Meesho_Dice\\fashion_ecommerce_features.csv")
print("Original data shape:", df.shape)
print("Original data types:")
print(df.dtypes.value_counts())

# Convert all categorical columns to numerical using Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = categorical_cols(["return_reason", "payment_method",])
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns to convert: {len(categorical_cols)}")
print(categorical_cols)
print(f"\nAlready numerical columns: {len(numerical_cols)}")
print(numerical_cols)
df = df[categorical_cols + numerical_cols]
# Apply Label Encoding to categorical columns
df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Now all columns should be numerical
print("\nAfter encoding - Data types:")
print(df_encoded.dtypes.value_counts())

# Select only int and float columns (should be all columns now)
df_final = df_encoded.select_dtypes(include=['int64', 'float64'])
print(f"\nFinal dataset shape: {df_final.shape}")
print("Final dataset data types:")
print(df_final.dtypes)

# Create pairplot with numerical data only (sample some columns to avoid overcrowding)
sample_cols = df_final.columns[:8]  # Take first 8 columns for visualization
sns_pairplot = sns.pairplot(df_final[sample_cols])
sns_pairplot.savefig("pairplot.png")
print("\nPairplot saved as 'pairplot.png'")

print("\nFirst few rows of final dataset:")
print(df_final.head())

# mfe_extract.py
"""
Given input data (CSV or Pandas DataFrame),
this script applies Mahalanobis Feature Extraction (MFE)
to generate dense features for downstream ML models.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Step 1: Build feature matrices
# ----------------------------
def build_feature_matrices(df, categorical_cols, numeric_cols):
    df = df.copy()
    df[categorical_cols] = df[categorical_cols].fillna("NA")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # One-hot encode categorical features into sparse matrix
    ohe = OneHotEncoder(sparse=True)
    X_cat_sparse = ohe.fit_transform(df[categorical_cols])
    if not sparse.issparse(X_cat_sparse):
        X_cat_sparse = sparse.csr_matrix(X_cat_sparse)

    # Scale numeric features
    scaler = StandardScaler(with_mean=False)
    X_num = np.asarray(df[numeric_cols], dtype=np.float64)
    if X_num.shape[1] > 0:
        X_num = scaler.fit_transform(X_num)

    return X_cat_sparse.tocsr(), X_num, {"ohe": ohe, "scaler": scaler}


# ----------------------------
# Step 2: Mahalanobis Feature Extraction (MFE)
# ----------------------------
def train_mfe(Xcat, y, d=10, epochs=5, batch_size=2048, lr=1e-2, verbose=True, random_state=0):
    """Train MFE projection matrix W"""
    y = np.asarray(y).reshape(-1)
    rng= np.random.RandomState(random_state) # random number generator
    I, J = Xcat.shape
    W = 0.01 * rng.randn(J, d).astype(np.float64)
    indices = np.arange(I)
    ridge = 1e-6

    for ep in range(epochs):
        rng.shuffle(indices)
        if verbose:
            print(f"Epoch {ep+1}/{epochs}")

        for start in range(0, I, batch_size):
            batch_idx = indices[start:start+batch_size]
            if batch_idx.size == 0:
                continue

            Xb = Xcat[batch_idx, :]
            yb = y[batch_idx].astype(np.float64)
            b = Xb.shape[0]
            if b == 0:
                continue

            # Forward projection
            X_ext = Xb.dot(W)
            Z = np.asarray(X_ext.T.dot(yb)).reshape(d)
            sum_y = yb.sum()
            sum_y2 = (yb**2).sum()
            sum_Xext = np.asarray(X_ext.sum(axis=0)).reshape(d)
            E_Z = (sum_Xext * sum_y) / b
            Sx = X_ext.T.dot(X_ext)
            Sy_factor = (b * sum_y2 - (sum_y**2))
            denom = (b**2 * max(1, (b - 1)))
            V = ((b * Sx - np.outer(sum_Xext, sum_Xext)) * Sy_factor) / denom

            # Regularized inverse
            V_reg = V + ridge * np.eye(d)
            try:
                V_inv = np.linalg.inv(V_reg)
            except np.linalg.LinAlgError:
                V_inv = np.linalg.inv(V_reg + 1e-3 * np.eye(d))

            delta = (Z - E_Z)

            # Gradient approximation
            s_c_y = np.asarray(Xb.T.dot(yb)).reshape(-1)
            s_c = np.asarray(Xb.sum(axis=0)).reshape(-1)
            dZ_dW = np.repeat(s_c_y[:, np.newaxis], d, axis=1)
            dE_dW = np.repeat((s_c * sum_y / b)[:, np.newaxis], d, axis=1)
            v = 2.0 * V_inv.dot(delta)
            term1_per_c = (dZ_dW - dE_dW).dot(v)
            XbT_Xext = Xb.T.dot(X_ext)
            q = V_inv.dot(delta)
            s_per_c = XbT_Xext.dot(q)
            y_factor = Sy_factor / denom
            term2_per_c = (2.0 * s_c * s_per_c * y_factor)
            grad_per_c = term1_per_c - term2_per_c
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            grad_W = grad_per_c[:, np.newaxis] * q_norm[np.newaxis, :]
            W += lr * grad_W

        if verbose:
            M_val = float(delta.T.dot(V_inv).dot(delta))
            print(f"  Completed epoch {ep+1}, sample objective M ~ {M_val:.4f}")

    return W


def transform_mfe(Xcat, W):
    """Project categorical sparse features into dense extracted features"""
    return Xcat.dot(W)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # The dataset is already processed above with all numerical columns
    # Use the processed df_final which has only int and float columns
    
    print("\n" + "="*50)
    print("MFE Feature Extraction Section")
    print("="*50)
    
    build_feat = build_feature_matrices(df, categorical_cols, numerical_cols )
    Xcat, Xnum, preprocessors = train_mfe(build_feat)
    