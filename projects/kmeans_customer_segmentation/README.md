# KMeans Customer Segmentation

## Overview
Goal: segment customers into groups using K-Means clustering on **[your 3 numeric features]** to support **[business use-case: marketing, retention, pricing, etc.]**.

## Data
- Source: **[where it came from / synthetic / provided file]**
- File: `data/raw/Mall_Customers.xlsx`
- Features used: **[Feature A]**, **[Feature B]**, **[Feature C]**
- Target label (optional): `Gender` (used only for visualization / sanity-check, not for training)

> Note: Raw data is stored under `data/raw`. If the dataset is sensitive or large, replace it with a download step.

## Method
1. Preprocessing
   - Select numeric features
   - Standardize features (important for distance-based clustering)
2. Model selection
   - WCSS (elbow method) across k = 1..10
   - Multiple initializations (`n_init`) to reduce sensitivity to random centroid seeds
3. Training
   - Fit KMeans with chosen k
4. Visualization
   - Elbow plot with ΔWCSS annotations
   - 3D scatter of clusters + centroids
   - Optional: PCA 2D projection for interpretability

## Results
### Elbow / WCSS
![Elbow plot](reports/figures/elbow.png)

### Final clustering (3D)
![3D clusters](reports/figures/clusters_3d.png)

Key findings:
- **[Finding #1]**
- **[Finding #2]**
- **[Finding #3]**

## How to run
### Option 1: Notebook
Open: `notebooks/[your_notebook].ipynb`

### Option 2: Recreate environment
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt