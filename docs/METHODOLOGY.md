# Scan-to-HBIM Methodology

## Overview

This document describes the methodology for converting 3D point cloud scans of heritage structures into semantically enriched HBIM (Historic Building Information Models).

---

## Pipeline Phases

### Phase 1: Load & Validate

**Purpose:** Load raw point cloud and validate data quality.

**Input:** PLY, LAS, or E57 file
**Output:** Validated point cloud

**Checks:**
- Point count > 0
- Bounding box dimensions (expected 1-20m for heritage)
- Color availability
- Normal availability

---

### Phase 2: Preprocessing

**Purpose:** Reduce data size while preserving geometric features.

**Steps:**

1. **Voxel Downsampling (0.01m)**
   - Divides space into 1cm grid cells
   - Preserves fine heritage details (inscriptions, profiles)
   - Reduction: ~93% (9.2M → 620K points)

2. **Statistical Outlier Removal (SOR)**
   - Parameters: k=20, std=2.0
   - Removes scanner noise
   - Preserves edges

3. **Normal Estimation**
   - PCA-based normal computation
   - Radius: 0.03m
   - Consistent orientation

---

### Phase 3: Feature Extraction

**Reference:** Croce, V. et al. (2021) - "From Survey to Semantic Representation for Cultural Heritage"

**9 Geometric Features (per scale):**

| Feature | Formula | Description |
|---------|---------|-------------|
| Linearity | (λ₁-λ₂)/λ₁ | High for edges |
| Planarity | (λ₂-λ₃)/λ₁ | High for walls |
| Sphericity | λ₃/λ₁ | High for corners |
| Omnivariance | ∛(λ₁×λ₂×λ₃) | 3D spread |
| Eigenentropy | -Σ(λᵢ×ln(λᵢ)) | Disorder |
| Surface Variation | λ₃/(λ₁+λ₂+λ₃) | Edge detection |
| Sum Eigenvalues | λ₁+λ₂+λ₃ | Total variance |
| Anisotropy | (λ₁-λ₃)/λ₁ | Directional bias |
| Verticality | 1-|nz| | Vertical orientation |

**Multi-scale Radii:** 0.03m, 0.06m, 0.10m

**Position Features:**
- X_normalized
- **Y_normalized** (KEY for 99.86% accuracy!)
- Z_normalized

**Total:** 9 × 3 + 3 = 30 features

---

### Phase 4: Segmentation

**Algorithm:** DBSCAN (Density-Based Spatial Clustering)

**Parameters:**
- eps: 0.05m (5cm neighborhood)
- min_samples: 50

**Output:** Cluster labels per point

---

### Phase 5: Classification

**Algorithm:** Random Forest

**Parameters:**
- n_estimators: 100
- class_weight: balanced
- test_size: 15%

**Key Insight - Y_normalized:**

The Y coordinate (depth from facade) is the most discriminative feature:
- Kemer (arch): Y ≈ 0.3-0.5 (front)
- Ana Cephe (wall): Y ≈ 0.1-0.3 (back)
- Seki (platform): Y ≈ 0.6-0.8 (very front)

**Accuracy:** 99.86%

---

### Phase 6: Mesh Reconstruction

**Algorithm:** Poisson Surface Reconstruction

**Parameters:**
- Depth: 10
- Scale: 1.1

**Per-element Trim:**
| Element | Trim % | Reason |
|---------|--------|--------|
| zemin | 10% | Remove sparse edges |
| seki | 5% | Preserve detail |
| ana_cephe | 0% | High density |
| kemer | 0% | High density |
| sacak | 10% | Remove sparse edges |

---

### Phase 7: IFC Export

**Schema:** IFC4

**Geometry:** IfcTriangulatedFaceSet

**Property Sets:**

1. **Pset_YapiKimligi** (Building Identity)
   - YapiAdi, YapimTarihi, Donem, Bani

2. **Pset_ElemanBilgisi** (Element Information)
   - ElemanAdi_TR/EN, Malzeme, IFCSinifi

3. **Pset_KorumaDurumu** (Conservation Status)
   - TescilDurumu, SonRestorasyon, MevcutDurum

---

## Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 99.86% |
| Key Feature | Y_normalized |
| Final IFC Size | ~80 MB |
| Elements | 5 classes |
| Property Sets | 3 |

---

## References

1. Croce, V. et al. (2021). "From Survey to Semantic Representation for Cultural Heritage: Definition of a FeLiX-based Scan-to-BIM approach for HBIM workflows." ISPRS Int. J. Geo-Inf.

2. Poux, F. (2025). "3D Data Science with Python."

3. Gil, J. et al. (2024). Heritage ML comparison study.
