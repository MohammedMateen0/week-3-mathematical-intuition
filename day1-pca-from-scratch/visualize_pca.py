import numpy as np
import matplotlib.pyplot as plt

from pca_from_scratch import pca_from_scratch

# ===== Generate correlated data =====
np.random.seed(42)
X = np.random.randn(500, 2)
X[:, 1] = 3 * X[:, 0] + np.random.randn(500) * 0.5

# ===== Run PCA =====
X_pca, explained_ratio, eigenvalues, components, mean = pca_from_scratch(X, 2)

# ===== 1D Projection (only PC1) =====
X_centered = X - mean
W1 = components[:, :1]
X_pca_1d = X_centered @ W1

# ===== Reconstruction from 1D =====
X_reconstructed = X_pca_1d @ W1.T + mean

# ===== Plot =====
plt.figure(figsize=(15, 5))

# -------- (1) Original Data + Principal Directions --------
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

# Draw principal components
for i in range(2):
    vec = components[:, i] * np.sqrt(eigenvalues[i])
    plt.quiver(mean[0], mean[1], vec[0], vec[1],
               angles='xy', scale_units='xy', scale=1,
               label=f'PC{i+1}')

plt.title("Original Data with Principal Components")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# -------- (2) Projection onto PC1 --------
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]),
            alpha=0.5)
plt.title("Projection onto Principal Component 1")
plt.xlabel("PC1")
plt.yticks([])
plt.grid(True)

# -------- (3) Reconstruction from 1D --------
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Original")
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
            alpha=0.6, label="Reconstructed (1D PCA)")
plt.title("Reconstruction from 1D PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()