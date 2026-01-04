# -*- coding: utf-8 -*-
"""
Phase 2: Preprocess Point Cloud
================================
Cloud Function for Scan-to-HBIM V6 Pipeline

Operations: Voxel downsampling, SOR filtering, Normal estimation
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
    """Phase 2: Preprocess - Voxel downsample, SOR filter, Normal estimation."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    voxel_size = request_json.get("voxel_size", 0.01)
    nb_neighbors = request_json.get("nb_neighbors", 20)
    std_ratio = request_json.get("std_ratio", 2.0)

    print(f"[Phase 2] Starting - Version: {version}")

    try:
        load_dependencies()
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "input.ply")
            local_output = os.path.join(tmpdir, "02_preprocessed.ply")

            # Download from Phase 1 output
            input_path = f"processed/{version}/01_raw/01_raw_pointcloud.ply"
            print(f"[Phase 2] Downloading: {input_path}")
            bucket.blob(input_path).download_to_filename(local_input)

            # Load point cloud
            pcd = o3d.io.read_point_cloud(local_input)
            original_count = len(pcd.points)
            print(f"[Phase 2] Loaded {original_count:,} points")

            # Voxel downsampling
            print(f"[Phase 2] Voxel downsampling (size={voxel_size}m)")
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            after_voxel = len(pcd.points)

            # Statistical outlier removal
            print(f"[Phase 2] SOR filtering (k={nb_neighbors}, std={std_ratio})")
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            after_sor = len(pcd.points)

            # Normal estimation
            print("[Phase 2] Estimating normals")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            reduction_pct = round((1 - after_sor / original_count) * 100, 1)
            print(f"[Phase 2] Reduced to {after_sor:,} points ({reduction_pct}% reduction)")

            # Save and upload
            o3d.io.write_point_cloud(local_output, pcd)
            output_path = f"processed/{version}/02_preprocessed/02_preprocessed_pointcloud.ply"
            bucket.blob(output_path).upload_from_filename(local_output)

            # Save stats
            stats = {
                "original_points": original_count,
                "after_voxel": after_voxel,
                "after_sor": after_sor,
                "reduction_percent": reduction_pct,
                "voxel_size": voxel_size,
                "has_normals": True
            }
            stats_path = f"processed/{version}/02_preprocessed/02_stats.json"
            bucket.blob(stats_path).upload_from_string(json.dumps(stats, indent=2))

            response = {
                "phase": "02_preprocess",
                "status": "success",
                "version": version,
                "outputs": {
                    "pointcloud": f"gs://{BUCKET_NAME}/{output_path}",
                    "stats": f"gs://{BUCKET_NAME}/{stats_path}"
                },
                "metrics": stats,
                "timestamp": datetime.now().isoformat(),
                "next_phase": "03_features"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 2] ERROR: {str(e)}")
        return json.dumps({"phase": "02_preprocess", "status": "error", "error": str(e)}), 500
