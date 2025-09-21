# Mahalanobis Feature Extraction (MFE) - Mathematical Formulation

## 📐 **Mathematical Foundation**

### **Problem Setup**
Given:
- **X** ∈ ℝ^(n×p): Sparse categorical feature matrix (n samples, p features after one-hot encoding)
- **y** ∈ ℝ^n: Target variable (continuous or binary)
- **d**: Desired output dimension (d << p)

**Goal**: Find projection matrix **W** ∈ ℝ^(p×d) that creates dense features **Z** = **XW** ∈ ℝ^(n×d)

---

## 🎯 **Core MFE Objective Function**

### **Mahalanobis Distance Formulation**

For a batch of data, MFE minimizes the **generalized Mahalanobis distance**:

```
𝔐(W) = (Z̄ - Ē)ᵀ V⁻¹ (Z̄ - Ē)
```

Where:
- **Z** = **XW** (projected features)
- **Z̄** = **Z**ᵀ**y** (correlation between projected features and target)
- **Ē** = 𝔼[**Z**ᵀ**y**] (expected correlation)
- **V** = Cov(**Z**, **y**) (covariance matrix)

---

## 🧮 **Detailed Mathematical Steps**

### **Step 1: Forward Projection**
```
Z = XW
```
- **X** ∈ ℝ^(b×p): Batch of sparse categorical features
- **W** ∈ ℝ^(p×d): Projection matrix to learn
- **Z** ∈ ℝ^(b×d): Dense projected features

### **Step 2: Statistics Computation**

#### **Correlation Vector**
```
Z̄ = Zᵀy = ∑ᵢ₌₁ᵇ zᵢyᵢ ∈ ℝᵈ
```

#### **Expected Correlation**
```
Ē = (∑ᵢ₌₁ᵇ zᵢ)(∑ᵢ₌₁ᵇ yᵢ) / b ∈ ℝᵈ
```

#### **Covariance Matrix**
```
V = (1/b) ∑ᵢ₌₁ᵇ (zᵢ - z̄)(zᵢ - z̄)ᵀ × σ²ᵧ + λI
```

Where:
- **z̄** = (1/b)∑zᵢ (mean of projected features)
- **σ²ᵧ** = Var(y) (target variance)
- **λI**: Ridge regularization term

### **Step 3: Mahalanobis Distance**
```
δ = Z̄ - Ē ∈ ℝᵈ
𝔐 = δᵀV⁻¹δ ∈ ℝ
```

### **Step 4: Gradient Computation**

#### **Gradient of δ with respect to W**
```
∂δ/∂W = ∂Z̄/∂W - ∂Ē/∂W
```

Where:
```
∂Z̄/∂W = Xᵀy ∈ ℝᵖ (broadcasted to ℝᵖˣᵈ)
∂Ē/∂W = (Xᵀ𝟙)(∑yᵢ)/b ∈ ℝᵖ (broadcasted to ℝᵖˣᵈ)
```

#### **Final Gradient**
```
∇W𝔐 = (∂δ/∂W) ⊗ (2V⁻¹δ)
```

Where ⊗ denotes outer product.

### **Step 5: Weight Update**
```
W ← W + η∇W𝔐
```
- **η**: Learning rate

---

## 💻 **Code-to-Math Mapping**

### **Your Code Implementation:**

```python
# Step 1: Forward projection
X_projected = X_batch.dot(W)  # Z = XW

# Step 2: Statistics
Z = X_projected.T.dot(y_batch)  # Z̄ = Zᵀy
sum_y = y_batch.sum()          # ∑yᵢ
sum_X = X_projected.sum(axis=0) # ∑zᵢ
E_Z = (sum_X * sum_y) / b      # Ē

# Step 3: Covariance
X_centered = X_projected - sum_X / b  # zᵢ - z̄
S_x = X_centered.T.dot(X_centered)    # ∑(zᵢ - z̄)(zᵢ - z̄)ᵀ
y_var = (sum_y2 / b - (sum_y / b) ** 2)  # σ²ᵧ
V = S_x * y_var + ridge * np.eye(d)      # V

# Step 4: Mahalanobis distance
delta = Z - E_Z                # δ
V_inv = np.linalg.inv(V)      # V⁻¹
M_val = delta.T.dot(V_inv).dot(delta)  # 𝔐

# Step 5: Gradient computation
grad_factor = 2.0 * V_inv.dot(delta)  # 2V⁻¹δ
dZ_dW = X_batch_T.dot(y_batch)        # ∂Z̄/∂W
dE_dW = X_batch_T.dot(np.ones(b)) * sum_y / b  # ∂Ē/∂W
grad_delta = (dZ_dW - dE_dW)          # ∂δ/∂W
grad_W = np.outer(grad_delta, grad_factor)  # ∇W𝔐

# Step 6: Update
W += lr * grad_W
```

---

## 🎯 **Intuitive Interpretation**

### **What MFE Optimizes:**

1. **Maximize Correlation**: Makes projected features **Z** highly correlated with target **y**
2. **Minimize Variance**: Reduces unwanted noise in the projected space
3. **Preserve Information**: Keeps the most predictive categorical relationships

### **Mathematical Intuition:**

```
Mahalanobis Distance = (Correlation - Expected)ᵀ × Precision × (Correlation - Expected)
```

This measures how "surprising" the correlation is, accounting for the natural variance in the data.

---

## 📊 **Dimensionality Transformation**

### **Before MFE:**
```
X_categorical ∈ ℝⁿˣᵖ (sparse, p ≈ 1000-10000)
```

### **After MFE:**
```
Z = XW ∈ ℝⁿˣᵈ (dense, d ≈ 10-50)
```

### **Information Preservation:**
The projection **W** is learned such that:
```
corr(Z, y) ≈ corr(X_optimal_subset, y)
```

---

## 🔬 **Advanced Mathematical Properties**

### **1. Optimization Landscape**
MFE optimizes a non-convex objective, but uses stochastic gradient descent with mini-batches for scalability.

### **2. Regularization**
Ridge term **λI** ensures numerical stability:
```
V_regularized = V + λI
```

### **3. Convergence**
The algorithm converges when:
```
||∇W𝔐|| < ε
```

---

## 🚀 **Why MFE Works**

1. **Statistical Efficiency**: Uses second-order statistics (covariance) vs first-order (correlation)
2. **Dimensionality Reduction**: p >> d compression while preserving predictive power
3. **Categorical Specialization**: Designed for sparse, high-dimensional categorical data
4. **Target Awareness**: Unlike PCA, it uses target information during feature extraction

This mathematical framework shows how MFE transforms sparse categorical embeddings into dense, predictive features through learned linear projections optimized via Mahalanobis distance minimization! 🎯
