# Meesho E-commerce Data Analysis Project

This project implements Mahalanobis Feature Extraction (MFE) for e-commerce return prediction analysis, specifically tailored for fashion retail data similar to Meesho's use case.

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

- **Synthetic Dataset Generation**: Creates realistic fashion e-commerce data
- **Mahalanobis Feature Extraction**: Advanced dimensionality reduction for categorical features
- **Return Rate Analysis**: Predicts product return probability
- **Data Visualization**: Generates plots and analysis charts

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

### Generate Dataset
```bash
python dataset.py
```
- Creates `fashion_ecommerce_features.csv` with synthetic fashion data
- Generates 1000 samples with 34 features
- Target return rate: ~25-33%

### Preprocess Data
```bash
python dataset_preprocess.py
```
- Converts categorical columns to numerical using Label Encoding
- Creates visualization plots
- Outputs: `fashion_ecommerce_numerical_only.csv`

### Advanced MFE Analysis
```bash
python data_prepro.py
```
- Applies Mahalanobis Feature Extraction to categorical data
- Evaluates feature quality using Random Forest
- Generates analysis plots and projection matrices

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

- `fashion_ecommerce_features.csv`: Original synthetic dataset
- `fashion_ecommerce_numerical_only.csv`: Preprocessed numerical dataset
- `mfe_projection_matrix.npy`: Trained MFE transformation matrix
- `features_matrix.npy`: Processed feature matrix
- `target_vector.npy`: Target labels
- Various `.png` files: Visualization plots

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **scipy**: Scientific computing

## Project Goals

- Demonstrate advanced feature engineering for e-commerce data
- Implement state-of-the-art dimensionality reduction techniques
- Provide realistic return prediction modeling framework
- Create reusable data science pipeline

## Notes

- All data is synthetic and for demonstration purposes
- Return rates are calibrated to realistic e-commerce ranges (25-33%)
- MFE implementation follows academic research standards
- Code is optimized for educational and research use
