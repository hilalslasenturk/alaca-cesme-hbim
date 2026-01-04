"""
Configuration for Scan-to-HBIM Pipeline
=======================================

All parameters for the heritage point cloud processing pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import os

# =============================================================================
# HERITAGE CLASSES
# =============================================================================

@dataclass
class HeritageClass:
    """Heritage element classification."""
    id: int
    name: str
    name_tr: str
    color: Tuple[int, int, int]
    ifc_class: str

CLASSES = {
    0: HeritageClass(0, "zemin", "Zemin", (34, 139, 34), "IfcSlab"),
    1: HeritageClass(1, "seki", "Seki", (176, 196, 222), "IfcSlab"),
    2: HeritageClass(2, "ana_cephe", "Ana Cephe", (139, 90, 43), "IfcWall"),
    3: HeritageClass(3, "kemer", "Kemer", (178, 34, 34), "IfcBuildingElementProxy"),
    4: HeritageClass(4, "sacak", "Sacak", (210, 180, 140), "IfcRoof"),
}

CLASS_NAMES = [c.name for c in CLASSES.values()]
N_CLASSES = len(CLASSES)

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================

PREPROCESSING = {
    "voxel_size": 0.01,        # 1cm - fixed for heritage detail
    "sor_k_neighbors": 20,      # SOR k neighbors
    "sor_std_ratio": 2.0,       # SOR standard deviation ratio
    "normal_radius": 0.03,      # Normal estimation radius
    "normal_max_nn": 30,        # Normal max neighbors
}

# =============================================================================
# FEATURE EXTRACTION (Croce et al. 2021)
# =============================================================================

FEATURES = {
    "radii": [0.03, 0.06, 0.10],  # Multi-scale radii
    "feature_names": [
        "linearity", "planarity", "sphericity", "omnivariance",
        "eigenentropy", "surface_variation", "sum_eigenvalues",
        "anisotropy", "verticality"
    ],
    "n_features_per_scale": 9,
    "total_features": 30,  # 9 x 3 + 3 (x, y, z normalized)
}

# =============================================================================
# SEGMENTATION (DBSCAN)
# =============================================================================

SEGMENTATION = {
    "dbscan_eps": 0.05,         # 5cm neighborhood
    "dbscan_min_samples": 50,   # Min points per cluster
    "min_segment_points": 100,  # Remove tiny segments
}

# =============================================================================
# CLASSIFICATION (Random Forest)
# =============================================================================

CLASSIFICATION = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "test_size": 0.15,
    "random_state": 42,
}

# =============================================================================
# MESH RECONSTRUCTION (Poisson)
# =============================================================================

MESH_RECONSTRUCTION = {
    "poisson_depth": 10,
    "poisson_scale": 1.1,
    "trim_percentiles": {
        "zemin": 10,
        "seki": 5,
        "ana_cephe": 0,
        "kemer": 0,
        "sacak": 10,
    }
}

# =============================================================================
# IFC EXPORT
# =============================================================================

IFC_EXPORT = {
    "schema": "IFC4",
    "geometry_type": "IfcTriangulatedFaceSet",
    "property_sets": [
        "Pset_YapiKimligi",
        "Pset_ElemanBilgisi",
        "Pset_KorumaDurumu"
    ]
}

# =============================================================================
# GCS CONFIGURATION (from environment)
# =============================================================================

def get_gcs_config():
    """Get GCS configuration from environment variables."""
    return {
        "project_id": os.getenv("GCS_PROJECT_ID", ""),
        "bucket_name": os.getenv("GCS_BUCKET_NAME", "alaca-cesme-hbim-v6"),
        "credentials_path": os.getenv("GCS_CREDENTIALS_PATH", ""),
    }

# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================

OUTPUT_FOLDERS = [
    "01_raw",
    "02_preprocessed",
    "03_features",
    "04_segmentation",
    "05_classification",
    "06_mesh",
    "07_ifc",
]

def get_output_path(base_dir: Path, version: str, phase: str) -> Path:
    """Get output path for a specific phase."""
    return base_dir / "processed" / version / phase
