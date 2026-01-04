# -*- coding: utf-8 -*-
"""
Phase 3: Feature Extraction
============================
Cloud Function for Scan-to-HBIM V6 Pipeline

Computes 28 geometric features: 9 features x 3 scales + Y_normalized (KEY!)
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

@functions_framework.http
def process(request):
    """Phase 3: Feature Extraction - 28 geometric features."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    scales = request_json.get("scales", [0.05, 0.10, 0.20])

    print(f"[Phase 3] Starting - Version: {version}, Scales: {scales}")

    try:
        load_dependencies()
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "input.ply")
            local_features = os.path.join(tmpdir, "03_features.npy")

            # Download from Phase 2
            input_path = f"processed/{version}/02_preprocessed/02_preprocessed_pointcloud.ply"
            print(f"[Phase 3] Downloading: {input_path}")
            bucket.blob(input_path).download_to_filename(local_input)

            # Load
            pcd = o3d.io.read_point_cloud(local_input)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            n_points = len(points)

            print(f"[Phase 3] Computing features for {n_points:,} points")

            # Build KD-Tree
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            # 9 features x 3 scales + Y_normalized = 28 features
            n_features = 9 * len(scales) + 1
            features = np.zeros((n_points, n_features), dtype=np.float32)

            # Y_normalized - KEY FEATURE for heritage buildings!
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            y_range = y_max - y_min
            features[:, -1] = (points[:, 1] - y_min) / y_range if y_range > 0 else 0
            print(f"[Phase 3] Y_normalized computed (range: {y_range:.2f}m)")

            # Compute features at each scale
            for scale_idx, scale in enumerate(scales):
                print(f"[Phase 3] Scale {scale}m ({scale_idx+1}/{len(scales)})")
                feature_offset = scale_idx * 9

                for i in range(n_points):
                    if i % 50000 == 0:
                        print(f"[Phase 3]   Progress: {i:,}/{n_points:,}")

                    [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], scale)
                    if k < 3:
                        continue

                    neighbors = points[idx]
                    cov = np.cov(neighbors.T)

                    try:
                        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
                        e1, e2, e3 = eigenvalues
                        e_sum = e1 + e2 + e3 + 1e-10

                        features[i, feature_offset:feature_offset+9] = [
                            (e1 - e2) / (e1 + 1e-10),  # linearity
                            (e2 - e3) / (e1 + 1e-10),  # planarity
                            e3 / (e1 + 1e-10),          # sphericity
                            (e1 * e2 * e3) ** (1/3),    # omnivariance
                            (e1 - e3) / (e1 + 1e-10),  # anisotropy
                            -sum([(ev/e_sum) * np.log(ev/e_sum + 1e-10) for ev in eigenvalues]),  # eigenentropy
                            e3 / e_sum,                 # curvature
                            1 - abs(normals[i, 2]) if len(normals) > i else 0,  # verticality
                            neighbors[:, 2].max() - neighbors[:, 2].min()  # height_range
                        ]
                    except:
                        pass

            # Save features
            np.save(local_features, features)
            output_path = f"processed/{version}/03_features/03_features.npy"
            bucket.blob(output_path).upload_from_filename(local_features)

            # Also upload point cloud with features for reference
            pcd_path = f"processed/{version}/03_features/03_pointcloud.ply"
            bucket.blob(pcd_path).upload_from_filename(local_input)

            stats = {
                "n_points": n_points,
                "n_features": n_features,
                "scales": scales,
                "y_range_m": float(y_range)
            }
            bucket.blob(f"processed/{version}/03_features/03_stats.json").upload_from_string(
                json.dumps(stats, indent=2)
            )

            response = {
                "phase": "03_features",
                "status": "success",
                "version": version,
                "outputs": {
                    "features": f"gs://{BUCKET_NAME}/{output_path}",
                    "pointcloud": f"gs://{BUCKET_NAME}/{pcd_path}"
                },
                "metrics": stats,
                "timestamp": datetime.now().isoformat(),
                "next_phase": "04_segment"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 3] ERROR: {str(e)}")
        return json.dumps({"phase": "03_features", "status": "error", "error": str(e)}), 500
