# -*- coding: utf-8 -*-
"""
Phase 4: Segmentation
======================
Cloud Function for Scan-to-HBIM V6 Pipeline

DBSCAN clustering for point cloud segmentation
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
    """Phase 4: Segmentation - DBSCAN clustering."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    eps = request_json.get("eps", 0.05)
    min_samples = request_json.get("min_samples", 50)

    print(f"[Phase 4] Starting - Version: {version}, eps={eps}, min_samples={min_samples}")

    try:
        load_dependencies()
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "input.ply")
            local_labels = os.path.join(tmpdir, "04_labels.npy")

            # Download from Phase 3
            input_path = f"processed/{version}/03_features/03_pointcloud.ply"
            print(f"[Phase 4] Downloading: {input_path}")
            bucket.blob(input_path).download_to_filename(local_input)

            # Load
            pcd = o3d.io.read_point_cloud(local_input)
            n_points = len(pcd.points)
            print(f"[Phase 4] Loaded {n_points:,} points")

            # DBSCAN clustering
            print(f"[Phase 4] Running DBSCAN...")
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=True))

            n_clusters = int(labels.max() + 1)
            n_noise = int((labels == -1).sum())
            noise_pct = round(n_noise / n_points * 100, 2)

            print(f"[Phase 4] Found {n_clusters} clusters, {n_noise:,} noise points ({noise_pct}%)")

            # Save labels
            np.save(local_labels, labels)
            output_path = f"processed/{version}/04_segmentation/04_cluster_labels.npy"
            bucket.blob(output_path).upload_from_filename(local_labels)

            # Copy point cloud for next phase
            pcd_path = f"processed/{version}/04_segmentation/04_pointcloud.ply"
            bucket.blob(pcd_path).upload_from_filename(local_input)

            stats = {
                "n_points": n_points,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_percent": noise_pct,
                "eps": eps,
                "min_samples": min_samples
            }
            bucket.blob(f"processed/{version}/04_segmentation/04_stats.json").upload_from_string(
                json.dumps(stats, indent=2)
            )

            response = {
                "phase": "04_segment",
                "status": "success",
                "version": version,
                "outputs": {
                    "labels": f"gs://{BUCKET_NAME}/{output_path}",
                    "pointcloud": f"gs://{BUCKET_NAME}/{pcd_path}"
                },
                "metrics": stats,
                "timestamp": datetime.now().isoformat(),
                "next_phase": "05_classify"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 4] ERROR: {str(e)}")
        return json.dumps({"phase": "04_segment", "status": "error", "error": str(e)}), 500
