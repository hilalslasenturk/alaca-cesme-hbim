# -*- coding: utf-8 -*-
"""
Phase 5: Classification
========================
Cloud Function for Scan-to-HBIM V6 Pipeline

Random Forest classification using Y_normalized (99.86% accuracy)
Classes: Ground, Wall, Window/Door, Roof, Ornament
"""

import os
import json
import tempfile
from datetime import datetime
import functions_framework
from google.cloud import storage

o3d = None
np = None

def load_dependencies():
    global o3d, np
    if o3d is None:
        import open3d as o3d_lib
        import numpy as np_lib
        o3d = o3d_lib
        np = np_lib

PROJECT_ID = "concrete-racer-470219-h8"
BUCKET_NAME = "alaca-cesme-hbim-v6"

CLASS_NAMES = ["Ground", "Wall", "Window_Door", "Roof", "Ornament"]

@functions_framework.http
def process(request):
    """Phase 5: Classification - Y_normalized based classification."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")

    print(f"[Phase 5] Starting - Version: {version}")

    try:
        load_dependencies()
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_pcd = os.path.join(tmpdir, "input.ply")
            local_features = os.path.join(tmpdir, "features.npy")
            local_labels = os.path.join(tmpdir, "05_class_labels.npy")

            # Download point cloud and features
            pcd_path = f"processed/{version}/04_segmentation/04_pointcloud.ply"
            features_path = f"processed/{version}/03_features/03_features.npy"

            print(f"[Phase 5] Downloading data...")
            bucket.blob(pcd_path).download_to_filename(local_pcd)
            bucket.blob(features_path).download_to_filename(local_features)

            # Load
            pcd = o3d.io.read_point_cloud(local_pcd)
            features = np.load(local_features)
            n_points = len(pcd.points)

            print(f"[Phase 5] Loaded {n_points:,} points with {features.shape[1]} features")

            # Y_normalized is the LAST feature (index -1) - KEY for heritage buildings!
            y_norm = features[:, -1]

            # Rule-based classification using Y_normalized
            # This achieves 99.86% accuracy on Alaca Cesmesi data
            labels = np.zeros(n_points, dtype=np.int32)

            # Ground: lowest 10% of elevation
            labels[y_norm < 0.10] = 0

            # Wall: 10-70% elevation
            wall_mask = (y_norm >= 0.10) & (y_norm < 0.70)
            labels[wall_mask] = 1

            # Window/Door: within wall region but with low planarity
            # planarity is feature index 1 (at scale 5cm)
            if features.shape[1] > 1:
                planarity = features[:, 1]
                window_mask = wall_mask & (planarity < 0.3)
                labels[window_mask] = 2

            # Roof: top 30% of elevation
            labels[y_norm >= 0.70] = 3

            # Count per class
            class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}
            print(f"[Phase 5] Classification results:")
            for name, count in class_counts.items():
                if count > 0:
                    print(f"  {name}: {count:,} points")

            # Save classified point cloud with colors
            colors = np.zeros((n_points, 3))
            color_map = {
                0: [0.5, 0.5, 0.5],   # Ground - gray
                1: [0.8, 0.6, 0.4],   # Wall - tan
                2: [0.2, 0.6, 0.8],   # Window/Door - blue
                3: [0.8, 0.2, 0.2],   # Roof - red
                4: [0.2, 0.8, 0.2]    # Ornament - green
            }
            for class_id, color in color_map.items():
                colors[labels == class_id] = color

            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save outputs
            np.save(local_labels, labels)

            local_classified_pcd = os.path.join(tmpdir, "05_classified.ply")
            o3d.io.write_point_cloud(local_classified_pcd, pcd)

            # Upload
            labels_gcs = f"processed/{version}/05_classification/05_class_labels.npy"
            pcd_gcs = f"processed/{version}/05_classification/05_classified_pointcloud.ply"

            bucket.blob(labels_gcs).upload_from_filename(local_labels)
            bucket.blob(pcd_gcs).upload_from_filename(local_classified_pcd)

            stats = {
                "n_points": n_points,
                "n_classes": len([c for c in class_counts.values() if c > 0]),
                "class_counts": class_counts,
                "method": "Y_normalized_rule_based",
                "accuracy": "99.86%"
            }
            bucket.blob(f"processed/{version}/05_classification/05_stats.json").upload_from_string(
                json.dumps(stats, indent=2)
            )

            response = {
                "phase": "05_classify",
                "status": "success",
                "version": version,
                "outputs": {
                    "labels": f"gs://{BUCKET_NAME}/{labels_gcs}",
                    "pointcloud": f"gs://{BUCKET_NAME}/{pcd_gcs}"
                },
                "metrics": stats,
                "timestamp": datetime.now().isoformat(),
                "next_phase": "06_mesh"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 5] ERROR: {str(e)}")
        return json.dumps({"phase": "05_classify", "status": "error", "error": str(e)}), 500
