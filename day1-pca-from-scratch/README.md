# Week 3 — Day 1: PCA From Scratch (Mathematical Intuition)

## 🎯 Objective

Implement Principal Component Analysis (PCA) from first principles and understand its geometric meaning.

---

## 📌 Problem Setup

We generate a 2D dataset with strong linear correlation:

* Feature 2 is approximately a linear function of Feature 1
* This creates a clear **principal direction of variance**

---

## 🧠 What PCA Does

PCA finds new axes such that:

* The first axis captures maximum variance
* Each subsequent axis captures remaining variance (orthogonal)

Mathematically:

Σv = λv

* v → principal direction (eigenvector)
* λ → variance along that direction (eigenvalue)

---

## ⚙️ Implementation Steps

1. **Center the data**
2. Compute **covariance matrix**
3. Perform **eigen decomposition**
4. Sort eigenvalues (descending)
5. Select top components
6. Project data

Projection formula:

Z = XW

---

## 📊 Key Results

* First principal component captures ~95%+ variance
* Data collapses into a near 1D structure
* PCA effectively rotates the coordinate system

---

## 📈 Visualisation

The plot shows:

* Original correlated data (ellipse shape)
* Principal axes (scaled by variance)
* Data after PCA projection

---

## ⚠️ Important Implementation Details

### 1. Centering is mandatory

Without centering, PCA is incorrect.

### 2. Use `np.linalg.eigh`

Covariance matrix is symmetric → more stable than `eig`.

### 3. Eigenvector sign ambiguity

Direction may flip but result remains valid.

---

## ✅ Validation with sklearn

We verify results using:

* sklearn PCA (SVD-based)
* Outputs match (up to sign)

---

## 🧠 Key Takeaways

* PCA is a **rotation of coordinate system**
* Eigenvectors = directions of maximum variance
* Eigenvalues = importance of those directions
* Dimensionality reduction = projection + information loss

---

## ▶️ Run Instructions

```bash
pip install -r requirements.txt
python pca_from_scratch.py
```

---



## 🚀 Extensions

* Add PCA reconstruction error
* Extend to higher dimensions
* Replace eigen decomposition with SVD

---
