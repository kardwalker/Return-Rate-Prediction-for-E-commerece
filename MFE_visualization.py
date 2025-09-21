import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (16, 12)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Create a comprehensive mathematical flow diagram
fig.suptitle('Mahalanobis Feature Extraction (MFE) - Mathematical Flow', 
             fontsize=20, fontweight='bold')

# Clear all axes first
for i in range(3):
    for j in range(3):
        axes[i,j].axis('off')

# ==================== TOP ROW: Problem Setup ====================
ax = fig.add_subplot(3, 1, 1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis('off')

# Input data visualization
ax.text(1, 1.5, 'INPUT DATA', fontsize=14, fontweight='bold', ha='center')
ax.add_patch(Rectangle((0.2, 0.8), 1.6, 0.8, facecolor='lightblue', edgecolor='black'))
ax.text(1, 1.2, 'X ∈ ℝⁿˣᵖ\n(sparse categorical)', fontsize=10, ha='center')

ax.text(3, 1.5, '→', fontsize=20, ha='center')

ax.text(4.5, 1.5, 'MFE TRANSFORMATION', fontsize=14, fontweight='bold', ha='center')
ax.add_patch(Rectangle((3.7, 0.8), 1.6, 0.8, facecolor='lightgreen', edgecolor='black'))
ax.text(4.5, 1.2, 'W ∈ ℝᵖˣᵈ\n(projection matrix)', fontsize=10, ha='center')

ax.text(6.5, 1.5, '→', fontsize=20, ha='center')

ax.text(8, 1.5, 'OUTPUT FEATURES', fontsize=14, fontweight='bold', ha='center')
ax.add_patch(Rectangle((7.2, 0.8), 1.6, 0.8, facecolor='lightcoral', edgecolor='black'))
ax.text(8, 1.2, 'Z ∈ ℝⁿˣᵈ\n(dense features)', fontsize=10, ha='center')

# ==================== MIDDLE ROW: Mathematical Steps ====================
ax2 = fig.add_subplot(3, 3, 4)
ax2.set_title('Step 1: Forward Projection', fontsize=12, fontweight='bold')
ax2.text(0.5, 0.8, r'Z = XW', fontsize=16, ha='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax2.text(0.5, 0.6, 'Project sparse features\ninto dense space', fontsize=10, ha='center')
ax2.text(0.5, 0.3, 'X: sparse (n×p)\nW: projection (p×d)\nZ: dense (n×d)', fontsize=9, ha='center')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

ax3 = fig.add_subplot(3, 3, 5)
ax3.set_title('Step 2: Statistics', fontsize=12, fontweight='bold')
ax3.text(0.5, 0.85, r'Z̄ = Z^T y', fontsize=14, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax3.text(0.5, 0.65, r'Ē = (Σz)(Σy)/b', fontsize=12, ha='center')
ax3.text(0.5, 0.45, r'V = Cov(Z,y) + λI', fontsize=12, ha='center')
ax3.text(0.5, 0.2, 'Correlation,\nExpectation,\nCovariance', fontsize=9, ha='center')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

ax4 = fig.add_subplot(3, 3, 6)
ax4.set_title('Step 3: Mahalanobis', fontsize=12, fontweight='bold')
ax4.text(0.5, 0.75, r'δ = Z̄ - Ē', fontsize=14, ha='center')
ax4.text(0.5, 0.55, r'M = δ^T V^(-1) δ', fontsize=16, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
ax4.text(0.5, 0.25, 'Measures correlation\n"surprise" accounting\nfor variance', fontsize=9, ha='center')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# ==================== BOTTOM ROW: Gradient and Update ====================
ax5 = fig.add_subplot(3, 1, 3)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 2)
ax5.axis('off')

# Gradient computation
ax5.text(2.5, 1.7, 'GRADIENT COMPUTATION', fontsize=14, fontweight='bold', ha='center')
ax5.text(2.5, 1.3, r'∂M/∂W = (∂δ/∂W) ⊗ (2V^(-1)δ)', 
         fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
ax5.text(2.5, 0.8, r'where: ∂δ/∂W = X^T y - (X^T 1)(Σy)/b', fontsize=10, ha='center')

# Update rule  
ax5.text(7.5, 1.7, 'WEIGHT UPDATE', fontsize=14, fontweight='bold', ha='center')
ax5.text(7.5, 1.3, r'W ← W + η (∂M/∂W)', 
         fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax5.text(7.5, 0.8, r'η: learning rate', fontsize=10, ha='center')

# Add arrow
ax5.annotate('', xy=(6, 1.3), xytext=(4, 1.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

plt.tight_layout()
plt.savefig('MFE_Mathematical_Flow.png', dpi=300, bbox_inches='tight')
plt.show()

print("Mathematical flow diagram saved as 'MFE_Mathematical_Flow.png'")
