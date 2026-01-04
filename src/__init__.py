"""
Scan-to-HBIM Framework
======================

AI-Assisted Automation of the Scan-to-HBIM Process for Cultural Heritage

This package provides a complete pipeline for converting 3D point cloud scans
of heritage structures into semantically enriched HBIM models.

Modules:
    - config: Configuration parameters
    - phase1_load: Load and validate point cloud
    - phase2_preprocess: Preprocessing (voxel, SOR, normals)
    - phase3_features: Feature extraction (Croce et al. 2021)
    - phase4_segment: DBSCAN segmentation
    - phase5_classify: Random Forest classification
    - phase6_mesh: Poisson mesh reconstruction
    - phase7_ifc: IFC export with semantic properties

Usage:
    from src import run_pipeline
    run_pipeline("input.ply", output_dir="outputs/", version="v1")

Author: Hilal Sila Senturk
License: MIT
"""

__version__ = "6.0.0"
__author__ = "Hilal Sila Senturk"
__email__ = "hilalslasenturk@gmail.com"

from .config import *
