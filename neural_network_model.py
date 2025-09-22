"""
Neural Network Architecture for Return Rate Prediction (0-100%)
================================================================

This module implements a deep neural network for predicting e-commerce return rates
using the dense features extracted from MFE (Mahalanobis Feature Extraction).

Features:
- Flexible neural network architecture
- Random search hyperparameter tuning
- Weight and bias optimization
- Regression for continuous return rate prediction (0-100%)
- Early stopping and regularization
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import uniform, randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import warnings
warnings.filterwarnings('ignore')

print("Neural Network for Return Rate Prediction (0-100%)")
print("=" * 60)

class ReturnRatePredictor:
    """
    Neural Network model for predicting return rates using MFE features
    """
    
    def __init__(self, input_dim=35, random_state=42):
        """
        Initialize the predictor
        
        Args:
            input_dim: Number of input features (MFE + numerical)
            random_state: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.history = None
        self.best_params = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def create_model(self, 
                     n_layers=3, 
                     layer_sizes=[128, 64, 32],
                     dropout_rate=0.3,
                     l2_reg=0.001,
                     activation='relu',
                     learning_rate=0.001,
                     optimizer='adam'):
        """
        Create neural network architecture
        
        Args:
            n_layers: Number of hidden layers
            layer_sizes: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            activation: Activation function
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type
        
        Returns:
            Compiled Keras model
        """
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            layer_sizes[0], 
            input_dim=self.input_dim,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name='input_layer'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, min(n_layers, len(layer_sizes))):
            model.add(layers.Dense(
                layer_sizes[i],
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer='he_normal',
                name=f'hidden_layer_{i}'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for regression (0-100% return rate)
        model.add(layers.Dense(
            1, 
            activation='sigmoid',  # Sigmoid to constrain output to [0,1]
            name='output_layer'
        ))
        
        # Compile model
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae', 'mse']  # Mean Absolute Error and MSE
        )
        
        return model
    
    def get_hyperparameter_space(self):
        """
        Define hyperparameter search space for random search
        
        Returns:
            Dictionary of hyperparameter distributions
        """
        param_space = {
            # Architecture parameters
            'n_layers': randint(2, 6),  # 2-5 hidden layers
            'layer_1_size': randint(64, 512),  # First layer size
            'layer_2_size': randint(32, 256),  # Second layer size  
            'layer_3_size': randint(16, 128),  # Third layer size
            'layer_4_size': randint(8, 64),    # Fourth layer size
            
            # Regularization parameters
            'dropout_rate': uniform(0.1, 0.4),  # Dropout rate 0.1-0.5
            'l2_reg': uniform(0.0001, 0.01),    # L2 regularization
            
            # Training parameters
            'learning_rate': uniform(0.0001, 0.01),  # Learning rate
            'batch_size': [16, 32, 64, 128],         # Batch sizes
            
            # Activation and optimizer
            'activation': ['relu', 'tanh', 'elu'],
            'optimizer': ['adam', 'rmsprop']
        }
        
        return param_space
    
    def create_model_for_search(self, 
                               n_layers=3,
                               layer_1_size=128,
                               layer_2_size=64, 
                               layer_3_size=32,
                               layer_4_size=16,
                               dropout_rate=0.3,
                               l2_reg=0.001,
                               activation='relu',
                               learning_rate=0.001,
                               optimizer='adam',
                               **kwargs):
        """
        Model creation function for hyperparameter search
        """
        # Build layer sizes list based on n_layers
        layer_sizes = [layer_1_size, layer_2_size, layer_3_size, layer_4_size][:n_layers]
        
        return self.create_model(
            n_layers=n_layers,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            activation=activation,
            learning_rate=learning_rate,
            optimizer=optimizer
        )
    
    def custom_scorer(self, y_true, y_pred):
        """
        Custom scoring function for hyperparameter tuning
        Combines MSE and MAE with return rate specific metrics
        """
        # Convert back to percentage scale (0-100%)
        y_true_pct = y_true * 100
        y_pred_pct = y_pred * 100
        
        mse = mean_squared_error(y_true_pct, y_pred_pct)
        mae = mean_absolute_error(y_true_pct, y_pred_pct)
        r2 = r2_score(y_true_pct, y_pred_pct)
        
        # Combined score: prioritize R² and penalize high errors
        score = r2 - (mse / 1000) - (mae / 100)
        return score
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, 
                             n_iter=50, cv_folds=3, verbose=1):
        """
        Perform random search hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training targets (scaled 0-1)
            X_val: Validation features  
            y_val: Validation targets (scaled 0-1)
            n_iter: Number of parameter combinations to try
            cv_folds: Cross-validation folds
            verbose: Verbosity level
        
        Returns:
            Best parameters found
        """
        print(f"\n🔍 Starting Random Search Hyperparameter Tuning")
        print(f"Search space: {n_iter} iterations, {cv_folds}-fold CV")
        
        # Create KerasRegressor wrapper
        keras_regressor = KerasRegressor(
            build_fn=self.create_model_for_search,
            epochs=100,
            verbose=0
        )
        
        # Get parameter space
        param_space = self.get_hyperparameter_space()
        
        # Random search
        random_search = RandomizedSearchCV(
            estimator=keras_regressor,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='neg_mean_squared_error',  # Negative MSE for minimization
            verbose=verbose,
            random_state=self.random_state,
            n_jobs=1  # Keep at 1 for neural networks
        )
        
        # Fit random search
        random_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = random_search.best_params_
        
        print(f"\n✅ Best Parameters Found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 Best CV Score: {-random_search.best_score_:.4f}")
        
        return self.best_params
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   params=None, epochs=200, verbose=1):
        """
        Train the neural network model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model parameters (if None, use best_params)
            epochs: Training epochs
            verbose: Verbosity level
        
        Returns:
            Trained model and training history
        """
        
        if params is None:
            params = self.best_params or {}
        
        print(f"\n🚀 Training Neural Network Model")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Epochs: {epochs}")
        
        # Create model with best parameters
        self.model = self.create_model_for_search(**params)
        
        print(f"\n🏗️ Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_return_rate_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        batch_size = params.get('batch_size', 32)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.model, self.history
    
    def evaluate_model(self, X_test, y_test, plot_results=True):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets (scaled 0-1)
            plot_results: Whether to create evaluation plots
        
        Returns:
            Dictionary of evaluation metrics
        """
        
        print(f"\n📊 Model Evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Convert to percentage scale (0-100%)
        y_test_pct = y_test * 100
        y_pred_pct = y_pred.flatten() * 100
        
        # Calculate metrics
        mse = mean_squared_error(y_test_pct, y_pred_pct)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_pct, y_pred_pct)
        r2 = r2_score(y_test_pct, y_pred_pct)
        
        # Return rate specific metrics
        abs_errors = np.abs(y_test_pct - y_pred_pct)
        within_5pct = np.mean(abs_errors <= 5.0) * 100  # % within 5% error
        within_10pct = np.mean(abs_errors <= 10.0) * 100  # % within 10% error
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'within_5pct_accuracy': within_5pct,
            'within_10pct_accuracy': within_10pct,
            'mean_absolute_error_pct': mae,
            'predictions_range': (y_pred_pct.min(), y_pred_pct.max()),
            'actual_range': (y_test_pct.min(), y_test_pct.max())
        }
        
        # Print results
        print(f"📈 Performance Metrics (Return Rate %):")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}%")
        print(f"  MAE: {mae:.2f}%")
        print(f"  Within 5% accuracy: {within_5pct:.1f}%")
        print(f"  Within 10% accuracy: {within_10pct:.1f}%")
        print(f"  Prediction range: {y_pred_pct.min():.1f}% - {y_pred_pct.max():.1f}%")
        
        if plot_results:
            self.plot_evaluation(y_test_pct, y_pred_pct, metrics)
        
        return metrics
    
    def plot_evaluation(self, y_true, y_pred, metrics):
        """
        Create comprehensive evaluation plots
        """
        
        plt.figure(figsize=(15, 10))
        
        # 1. Predictions vs Actual
        plt.subplot(2, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Return Rate (%)')
        plt.ylabel('Predicted Return Rate (%)')
        plt.title(f'Predictions vs Actual\nR² = {metrics["r2_score"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        plt.subplot(2, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Return Rate (%)')
        plt.ylabel('Residuals (%)')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 3. Error distribution
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='orange')
        plt.xlabel('Prediction Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\nMAE = {metrics["mae"]:.2f}%')
        plt.grid(True, alpha=0.3)
        
        # 4. Training history (if available)
        if hasattr(self, 'history') and self.history:
            plt.subplot(2, 3, 4)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Accuracy within thresholds
        plt.subplot(2, 3, 5)
        abs_errors = np.abs(y_true - y_pred)
        thresholds = [1, 2, 5, 10, 15, 20]
        accuracies = [np.mean(abs_errors <= t) * 100 for t in thresholds]
        plt.bar(range(len(thresholds)), accuracies, color='purple', alpha=0.7)
        plt.xlabel('Error Threshold (%)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy at Different Thresholds')
        plt.xticks(range(len(thresholds)), thresholds)
        plt.grid(True, alpha=0.3)
        
        # 6. Feature importance (if possible)
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.5, f"""Model Summary:
        
RMSE: {metrics['rmse']:.2f}%
MAE: {metrics['mae']:.2f}%
R² Score: {metrics['r2_score']:.3f}

Accuracy:
Within 5%: {metrics['within_5pct_accuracy']:.1f}%
Within 10%: {metrics['within_10pct_accuracy']:.1f}%

Range:
Actual: {metrics['actual_range'][0]:.1f}% - {metrics['actual_range'][1]:.1f}%
Predicted: {metrics['predictions_range'][0]:.1f}% - {metrics['predictions_range'][1]:.1f}%
        """, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='center', fontfamily='monospace')
        plt.axis('off')
        plt.title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('neural_network_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Evaluation plots saved as 'neural_network_evaluation.png'")


def load_mfe_features():
    """
    Load MFE features and targets from previous analysis
    """
    try:
        # Try to load pre-processed features
        X = np.load('features_matrix.npy')
        y = np.load('target_vector.npy')
        print(f"✅ Loaded MFE features: {X.shape}")
        print(f"✅ Loaded targets: {y.shape}")
        return X, y
    except FileNotFoundError:
        print("❌ MFE features not found. Running data preprocessing first...")
        # Run MFE preprocessing
        exec(open('data_prepro.py').read())
        # Try loading again
        X = np.load('features_matrix.npy')
        y = np.load('target_vector.npy')
        return X, y


def main():
    """
    Main function to run the complete neural network pipeline
    """
    
    print("🧠 Neural Network Return Rate Prediction Pipeline")
    print("=" * 60)
    
    # Load MFE features
    X, y = load_mfe_features()
    
    print(f"\n📊 Dataset Information:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: {y.min():.3f} - {y.max():.3f}")
    print(f"Target mean: {y.mean():.3f} ± {y.std():.3f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")  
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Initialize predictor
    predictor = ReturnRatePredictor(input_dim=X.shape[1])
    
    # Hyperparameter tuning
    print(f"\n🔍 Hyperparameter Tuning Phase")
    best_params = predictor.hyperparameter_tuning(
        X_train, y_train, X_val, y_val,
        n_iter=30,  # Reduced for demo, increase for production
        cv_folds=3,
        verbose=1
    )
    
    # Train final model
    print(f"\n🚀 Training Final Model")
    model, history = predictor.train_model(
        X_train, y_train, X_val, y_val,
        params=best_params,
        epochs=150,
        verbose=1
    )
    
    # Evaluate model
    print(f"\n📊 Final Model Evaluation")
    metrics = predictor.evaluate_model(X_test, y_test, plot_results=True)
    
    # Save model and results
    model.save('return_rate_predictor.h5')
    np.save('best_hyperparameters.npy', best_params)
    
    print(f"\n💾 Results Saved:")
    print(f"  Model: return_rate_predictor.h5")
    print(f"  Best params: best_hyperparameters.npy")
    print(f"  Evaluation plot: neural_network_evaluation.png")
    
    return predictor, metrics, best_params


if __name__ == "__main__":
    predictor, metrics, best_params = main()
