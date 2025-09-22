"""
Quick Test Script for Neural Network Model
==========================================

This script provides a quick test of the neural network architecture
without full hyperparameter tuning for rapid prototyping.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

print("🧪 Quick Neural Network Test")
print("=" * 40)

def create_simple_model(input_dim, learning_rate=0.001):
    """Create a simple neural network for testing"""
    
    model = keras.Sequential([
        # Input layer
        layers.Dense(128, input_dim=input_dim, activation='relu', name='input_layer'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(64, activation='relu', name='hidden_1'),
        layers.BatchNormalization(), 
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu', name='hidden_2'),
        layers.Dropout(0.2),
        
        # Output layer (sigmoid for 0-1 range, then scale to 0-100%)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def quick_test():
    """Run a quick test of the neural network"""
    
    try:
        # Load MFE features
        print("📁 Loading MFE features...")
        X = np.load('features_matrix.npy')
        y = np.load('target_vector.npy')
        print(f"✅ Features loaded: {X.shape}")
        print(f"✅ Targets loaded: {y.shape}")
        
    except FileNotFoundError:
        print("❌ MFE features not found. Please run data_prepro.py first.")
        return None
    
    # Data info
    print(f"\n📊 Data Overview:")
    print(f"Features: {X.shape[1]} dimensions")
    print(f"Samples: {X.shape[0]}")
    print(f"Target range: {y.min():.3f} - {y.max():.3f}")
    print(f"Target mean: {y.mean():.3f} ± {y.std():.3f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔀 Data Split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Create model
    print(f"\n🏗️ Creating Neural Network...")
    model = create_simple_model(X.shape[1])
    
    print("📋 Model Summary:")
    model.summary()
    
    # Train model
    print(f"\n🚀 Training Model...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print(f"\n📊 Evaluating Model...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Convert to percentage (0-100%)
    y_test_pct = y_test * 100
    y_pred_pct = y_pred.flatten() * 100
    
    # Metrics
    mse = mean_squared_error(y_test_pct, y_pred_pct)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_pct, y_pred_pct)
    r2 = r2_score(y_test_pct, y_pred_pct)
    
    # Accuracy metrics
    abs_errors = np.abs(y_test_pct - y_pred_pct)
    within_5pct = np.mean(abs_errors <= 5.0) * 100
    within_10pct = np.mean(abs_errors <= 10.0) * 100
    
    print(f"\n📈 Results (Return Rate %):")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  MAE: {mae:.2f}%")
    print(f"  Within 5% accuracy: {within_5pct:.1f}%")
    print(f"  Within 10% accuracy: {within_10pct:.1f}%")
    print(f"  Prediction range: {y_pred_pct.min():.1f}% - {y_pred_pct.max():.1f}%")
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    
    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predictions vs Actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_pct, y_pred_pct, alpha=0.6)
    plt.plot([y_test_pct.min(), y_test_pct.max()], 
             [y_test_pct.min(), y_test_pct.max()], 'r--', lw=2)
    plt.xlabel('Actual Return Rate (%)')
    plt.ylabel('Predicted Return Rate (%)')
    plt.title(f'Predictions vs Actual\\nR² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(1, 3, 3)
    plt.hist(abs_errors, bins=20, alpha=0.7, color='orange')
    plt.xlabel('Absolute Error (%)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution\\nMAE = {mae:.2f}%')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_nn_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save model for later use
    model.save('quick_test_model.h5')
    print(f"\n💾 Model saved as 'quick_test_model.h5'")
    print(f"📊 Results plot saved as 'quick_nn_test_results.png'")
    
    return {
        'model': model,
        'history': history,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct
        }
    }

if __name__ == "__main__":
    results = quick_test()
    
    if results:
        print(f"\n✅ Quick test completed successfully!")
        print(f"🎯 Key Result: R² = {results['metrics']['r2']:.3f}")
    else:
        print(f"\n❌ Test failed. Please check MFE features are available.")
