import numpy as np
import matplotlib.pyplot as plt

def pca_from_scratch(X, n_components=2):
    # Step 1: Center
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Step 2: Covariance
    cov_matrix = np.cov(X_centered.T)

    # Step 3: Eigen decomposition (stable)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select components
    components = eigenvectors[:, :n_components]

    # Step 6: Project
    X_pca = X_centered @ components

    # Explained variance
    explained_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    return X_pca, explained_ratio, eigenvalues, components, mean


# ===== Demo =====
np.random.seed(42)
X = np.random.randn(1000, 2)
X[:, 1] = 3 * X[:, 0] + np.random.randn(1000) * 0.5

X_pca, explained_ratio, eigenvalues, components, mean = pca_from_scratch(X)

print("Eigenvalues:", eigenvalues)
print("Explained Ratio:", explained_ratio)