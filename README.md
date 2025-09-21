# Meesho E-commerce Data Analysis Project

## 🎯 Overview

This project implements **Mahalanobis Feature Extraction (MFE)** for e-commerce return prediction analysis using a **synthetic dataset** generated specifically for this research. The dataset mimics real-world fashion retail data patterns similar to Meesho's e-commerce platform.

## 📊 **Synthetic Dataset Generation**

> **Important**: This project uses a completely **synthetic dataset** generated programmatically to simulate realistic e-commerce fashion retail scenarios. No real customer or business data is used.

### Dataset Characteristics:
- **1000 synthetic samples** with realistic statistical distributions
- **34 engineered features** across product, basket, and customer levels  
- **Target variable**: `return_probability` (continuous, 0.104-0.450 range)
- **Realistic return rate**: Calibrated to 25-33% industry standards
- **Multi-category data**: Women's ethnic/western, Men's, Kids' fashion items

## Project Structure

```
├── dataset.py                          # Dataset generation script
├── dataset_preprocess.py               # Data preprocessing with label encoding
├── data_prepro.py                     # Advanced MFE implementation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
└── README.md                         # This file
```

## Features

- **🏭 Synthetic Dataset Generation**: Creates realistic fashion e-commerce data with statistical accuracy
- **🧠 Mahalanobis Feature Extraction**: Advanced dimensionality reduction for categorical features  
- **📈 Return Probability Prediction**: Regression analysis for continuous return probability targets
- **📊 Data Visualization**: Comprehensive plots and statistical analysis charts
- **🔧 Complete Pipeline**: End-to-end data science workflow from generation to analysis

## Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv env_dice
   # Windows
   .\env_dice\Scripts\activate
   # Linux/Mac
   source env_dice/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generate Synthetic Dataset
```bash
python dataset.py
```
**What it does:**
- **Generates completely synthetic fashion e-commerce data**
- Creates `fashion_ecommerce_features.csv` with 1000 realistic samples  
- **34 engineered features** covering:
  - **Product Level**: Brand, category, pricing, ratings, materials
  - **Basket Level**: Order details, payment methods, device info
  - **Customer Level**: Purchase history, behavior patterns, demographics
- **Target**: `return_probability` (continuous regression target)
- **Calibrated return rate**: ~25-33% matching industry standards

### Preprocess Data
```bash
python dataset_preprocess.py
```
- Converts categorical columns to numerical using Label Encoding
- Creates visualization plots
- Outputs: `fashion_ecommerce_numerical_only.csv`

### Advanced MFE Analysis (Main Algorithm)
```bash
python data_prepro.py
```
**Core Implementation:**
- **Input**: Synthetic dataset with categorical + numerical features
- **Process**: Applies Mahalanobis Feature Extraction to categorical data  
- **Transformation**: 212 sparse categorical features → 20 dense MFE features
- **Target**: Uses `return_probability` as continuous regression target
- **Evaluation**: Compares feature sets using Random Forest Regression
- **Output**: Analysis plots, projection matrices, performance metrics

**Results**: MFE + Numerical features achieve **R² = 0.54** (54% variance explained)

## Key Algorithms

### Mahalanobis Feature Extraction (MFE)
- Transforms high-dimensional sparse categorical features into dense representations
- Preserves predictive relationships with target variable
- Reduces feature dimensionality while maintaining information content

### Data Processing Pipeline
1. **Categorical Features**: One-Hot Encoding → MFE Transformation
2. **Numerical Features**: Standard Scaling (mean=0, std=1)
3. **Feature Combination**: Concatenation of processed features

## Output Files

### Generated Datasets
- `fashion_ecommerce_features.csv`: **Original synthetic dataset** (1000×34)
- `fashion_ecommerce_numerical_only.csv`: Label-encoded version (all numerical)

### MFE Analysis Results  
- `mfe_projection_matrix.npy`: Trained MFE transformation matrix (212×20)
- `features_matrix.npy`: Final processed feature matrix (1000×33)
- `target_vector.npy`: Return probability targets (1000×1)

### Visualizations
- `pairplot.png`: Feature distribution analysis
- `mfe_analysis.png`: MFE visualization and feature importance
- `MFE_Mathematical_Flow.png`: Algorithm flow diagram

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **scipy**: Scientific computing

## Project Goals

- **🔬 Research Demonstration**: Showcase advanced feature engineering for e-commerce analytics
- **📚 Educational Resource**: Implement academic-standard dimensionality reduction techniques  
- **🏗️ Reusable Framework**: Create generalizable data science pipeline for fashion retail
- **📈 Synthetic Data Modeling**: Demonstrate realistic data generation for research purposes
- **🧮 Algorithm Implementation**: Provide production-ready MFE implementation

## Notes & Disclaimers

> **⚠️ Synthetic Data**: All data in this project is **artificially generated** using statistical models. No real customer, transaction, or business data is used or required.

- **Purpose**: Educational and research demonstration only
- **Return rates**: Calibrated to realistic e-commerce ranges (25-33%) based on industry research  
- **Statistical accuracy**: Synthetic data follows real-world fashion retail patterns
- **MFE implementation**: Based on academic research standards and best practices
- **Code optimization**: Designed for educational, research, and production use cases

## 🤝 Applications

This synthetic dataset and MFE implementation can be used for:
- **Academic research** in e-commerce analytics
- **Educational purposes** for learning advanced feature engineering
- **Proof-of-concept** development for retail analytics
- **Algorithm benchmarking** and comparison studies
- **Training data** for machine learning model development
