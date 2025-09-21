import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("Mahalanobis Feature Extraction for E-commerce Dataset")
print("="*60)

# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Load and preprocess the e-commerce dataset"""
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    print(f"Original data types:\n{df.dtypes.value_counts()}")
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variables from feature columns
    target_columns = ['returned', 'return_probability', 'return_reason']
    categorical_cols = [col for col in categorical_cols if col not in target_columns]
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in target_columns]
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    return df, categorical_cols, numerical_cols

# Build feature matrices
def build_feature_matrices(df, categorical_cols, numeric_cols):
    """Build categorical sparse matrix and numerical matrix"""
    df_clean = df.copy()
    
    # Handle missing values
    if categorical_cols:
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna("Unknown")
    if numeric_cols:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0.0)
    
    # One-hot encode categorical features
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        X_cat_sparse = ohe.fit_transform(df_clean[categorical_cols])
        print(f"Categorical features shape after OHE: {X_cat_sparse.shape}")
    else:
        X_cat_sparse = sparse.csr_matrix((len(df_clean), 0))
        ohe = None
        print("No categorical features to encode")
    
    # Scale numerical features
    if numeric_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df_clean[numeric_cols].astype(float))
        print(f"Numerical features shape after scaling: {X_num.shape}")
    else:
        X_num = np.empty((len(df_clean), 0))
        scaler = None
        print("No numerical features to scale")
    
    return X_cat_sparse.tocsr(), X_num, {"ohe": ohe, "scaler": scaler}

# Mahalanobis Feature Extraction
def train_mfe(X_cat, y, d=10, epochs=5, batch_size=1024, lr=1e-3, verbose=True, random_state=42):
    """
    Train Mahalanobis Feature Extraction projection matrix W
    
    Args:
        X_cat: Sparse categorical feature matrix
        y: Target variable (binary/continuous)
        d: Output dimension of extracted features
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        verbose: Print training progress
        random_state: Random seed
    
    Returns:
        W: Projection matrix of shape (n_features, d)
    """
    print(f"\nTraining MFE with d={d}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    y = np.asarray(y).reshape(-1).astype(np.float64)
    rng = np.random.RandomState(random_state)
    
    I, J = X_cat.shape
    print(f"Input sparse matrix shape: {X_cat.shape}")
    
    # Initialize projection matrix
    W = 0.01 * rng.randn(J, d).astype(np.float64)
    indices = np.arange(I)
    ridge = 1e-6
    
    for ep in range(epochs):
        rng.shuffle(indices)
        if verbose:
            print(f"\nEpoch {ep+1}/{epochs}")
        
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, I, batch_size):
            end = min(start + batch_size, I)
            batch_idx = indices[start:end]
            
            if len(batch_idx) == 0:
                continue
                
            # Get batch data
            X_batch = X_cat[batch_idx, :]
            y_batch = y[batch_idx]
            b = X_batch.shape[0]
            
            if b <= 1:  # Skip if batch too small
                continue
            
            try:
                # Forward projection
                X_projected = X_batch.dot(W)  # Shape: (b, d)
                
                # Compute statistics
                Z = X_projected.T.dot(y_batch)  # Shape: (d,)
                sum_y = y_batch.sum()
                sum_y2 = (y_batch ** 2).sum()
                sum_X = X_projected.sum(axis=0)  # Shape: (d,)
                
                # Expected values
                E_Z = (sum_X * sum_y) / b
                
                # Covariance-like terms
                X_centered = X_projected - sum_X / b
                S_x = X_centered.T.dot(X_centered)
                
                # Variance scaling
                y_var = max(1e-8, (sum_y2 / b - (sum_y / b) ** 2))
                
                # Regularized covariance matrix
                V = S_x * y_var + ridge * np.eye(d)
                
                try:
                    V_inv = np.linalg.inv(V)
                except np.linalg.LinAlgError:
                    V_inv = np.linalg.inv(V + 1e-3 * np.eye(d))
                
                # Compute residual
                delta = Z - E_Z
                
                # Objective value (Mahalanobis distance)
                M_val = delta.T.dot(V_inv).dot(delta)
                epoch_loss += M_val
                
                # Gradient computation
                grad_factor = 2.0 * V_inv.dot(delta)
                
                # Compute gradient w.r.t. W
                X_batch_T = X_batch.T  # Shape: (J, b)
                
                # Gradient of Z w.r.t. W
                dZ_dW = X_batch_T.dot(y_batch)  # Shape: (J,)
                
                # Gradient of E_Z w.r.t. W  
                dE_dW = X_batch_T.dot(np.ones(b)) * sum_y / b  # Shape: (J,)
                
                # Final gradient
                grad_delta = (dZ_dW - dE_dW)  # Shape: (J,)
                grad_W = np.outer(grad_delta, grad_factor)  # Shape: (J, d)
                
                # Update W
                W += lr * grad_W
                n_batches += 1
                
            except Exception as e:
                if verbose:
                    print(f"  Batch error: {e}")
                continue
        
        if verbose and n_batches > 0:
            avg_loss = epoch_loss / n_batches
            print(f"  Average batch objective: {avg_loss:.6f}")
    
    print(f"MFE training completed. Final W shape: {W.shape}")
    return W

def transform_mfe(X_cat, W):
    """Transform categorical features using trained MFE projection matrix"""
    return X_cat.dot(W)

def evaluate_features(X_original, X_mfe, X_num, y, target_type='classification', test_size=0.2, random_state=42):
    """
    Evaluate the quality of MFE features by comparing model performance
    Supports both classification and regression tasks
    """
    print("\n" + "="*50)
    print("Feature Quality Evaluation")
    print("="*50)
    
    # Prepare different feature sets
    feature_sets = {}
    
    if X_original.shape[1] > 0:
        feature_sets['Original_Categorical'] = X_original.toarray() if sparse.issparse(X_original) else X_original
    
    if X_mfe.shape[1] > 0:
        feature_sets['MFE_Features'] = X_mfe
    
    if X_num.shape[1] > 0:
        feature_sets['Numerical'] = X_num
    
    if X_mfe.shape[1] > 0 and X_num.shape[1] > 0:
        feature_sets['MFE_plus_Numerical'] = np.hstack([X_mfe, X_num])
    
    if X_original.shape[1] > 0 and X_num.shape[1] > 0:
        orig_dense = X_original.toarray() if sparse.issparse(X_original) else X_original
        feature_sets['All_Original'] = np.hstack([orig_dense, X_num])
    
    results = {}
    
    for name, features in feature_sets.items():
        try:
            # Skip if features are too large (for computational efficiency)
            if features.shape[1] > 1000:
                print(f"Skipping {name} (too many features: {features.shape[1]})")
                continue
                
            # Split data based on target type
            if target_type == 'regression':
                X_train, X_test, y_train, y_test = train_test_split(
                    features, y, test_size=test_size, random_state=random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, y, test_size=test_size, random_state=random_state, stratify=y
                )
            
            # Train model based on target type
            if target_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_squared_error, r2_score
                
                model = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'r2_score': r2,
                    'mse': mse,
                    'rmse': rmse,
                    'n_features': features.shape[1],
                    'feature_shape': features.shape
                }
                
                print(f"{name:20s}: R² = {r2:.4f}, RMSE = {rmse:.4f}, Features = {features.shape[1]}")
                
            else:
                # Classification
                clf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
                clf.fit(X_train, y_train)
                
                # Make predictions
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'n_features': features.shape[1],
                    'feature_shape': features.shape
                }
                
                print(f"{name:20s}: Accuracy = {accuracy:.4f}, Features = {features.shape[1]}")
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    return results

# Main execution function
def main():
    """Main function to run the complete MFE pipeline"""
    
    # Note: Update this path to your actual dataset file
    file_path = "fashion_ecommerce_features.csv"  # Update this path
    
    try:
        # Load data
        df, categorical_cols, numerical_cols = load_and_preprocess_data(file_path)
        
        # Use return_probability as target (continuous regression target)
        if 'return_probability' in df.columns:
            y = df['return_probability'].values
            print(f"Using 'return_probability' as target (continuous)")
            print(f"Target range: {y.min():.3f} to {y.max():.3f}")
            print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")
            target_type = 'regression'
        elif 'returned' in df.columns:
            # Fallback to binary target
            y = df['returned'].values
            print(f"Using 'returned' as target (binary)")
            print(f"Target distribution: {np.bincount(y)}")
            target_type = 'classification'
        elif 'return_reason' in df.columns:
            # Use return_reason as target
            le_target = LabelEncoder()
            y = le_target.fit_transform(df['return_reason'].fillna('none'))
            print(f"Using 'return_reason' as target. Classes: {le_target.classes_}")
            target_type = 'classification'
        else:
            # Create synthetic binary target
            np.random.seed(42)
            y = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
            print("Created synthetic binary target for demonstration")
            target_type = 'classification'
        
        # Build feature matrices
        X_cat_sparse, X_num, preprocessors = build_feature_matrices(df, categorical_cols, numerical_cols)
        
        if X_cat_sparse.shape[1] == 0:
            print("No categorical features found. MFE requires categorical features.")
            return
        
        # Train MFE
        mfe_dim = min(20, X_cat_sparse.shape[1] // 10, 50)  # Reasonable dimension
        mfe_dim = max(5, mfe_dim)  # Minimum 5 dimensions
        
        W = train_mfe(
            X_cat_sparse, y, 
            d=mfe_dim, 
            epochs=10, 
            batch_size=1024, 
            lr=1e-3,
            verbose=True
        )
        
        # Transform features
        X_mfe = transform_mfe(X_cat_sparse, W)
        print(f"MFE transformed features shape: {X_mfe.shape}")
        
        # Evaluate feature quality
        results = evaluate_features(X_cat_sparse, X_mfe, X_num, y, target_type)
        
        # Save results
        print(f"\nSaving MFE projection matrix (shape: {W.shape})")
        np.save('mfe_projection_matrix.npy', W)
        
        # Create visualization if possible
        if X_mfe.shape[1] >= 2:
            plt.figure(figsize=(12, 5))
            
            # Plot first two MFE dimensions
            plt.subplot(1, 2, 1)
            plt.scatter(X_mfe[:, 0], X_mfe[:, 1], c=y, alpha=0.6, cmap='viridis')
            plt.xlabel('MFE Dimension 1')
            plt.ylabel('MFE Dimension 2')
            plt.title('MFE Features (First 2 Dimensions)')
            plt.colorbar()
            
            # Plot feature importance (first 10 features of W)
            plt.subplot(1, 2, 2)
            feature_importance = np.abs(W).mean(axis=1)
            top_features = np.argsort(feature_importance)[-10:]
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.xlabel('Average Absolute Weight')
            plt.title('Top 10 Feature Importance in MFE')
            plt.ylabel('Feature Index')
            
            plt.tight_layout()
            plt.savefig('mfe_analysis.png', dpi=150, bbox_inches='tight')
            print("Saved analysis plot as 'mfe_analysis.png'")
        
        print("\n" + "="*60)
        print("MFE Feature Extraction Complete!")
        print("="*60)
        print("Files saved:")
        print("- mfe_projection_matrix.npy: Trained MFE projection matrix")
        print("- mfe_analysis.png: Visualization of results")
        
        return W, X_mfe, results
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please update the file_path variable with the correct path to your dataset.")
        return None, None, None
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None, None

# Run the complete pipeline
if __name__ == "__main__":
    W, X_mfe, results = main()