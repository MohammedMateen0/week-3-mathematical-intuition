from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(1000, 2)
X[:, 1] = 3 * X[:, 0] + np.random.randn(1000) * 0.5

pca = PCA(n_components=2)
X_sklearn = pca.fit_transform(X)

print("Explained variance (sklearn):", pca.explained_variance_ratio_)