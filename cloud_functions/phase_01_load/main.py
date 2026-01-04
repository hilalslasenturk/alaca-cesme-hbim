# -*- coding: utf-8 -*-
"""
Phase 1: Load & Validate Point Cloud
=====================================
Cloud Function for Scan-to-HBIM V6 Pipeline

Trigger: HTTP POST
Input: {"version": "v1", "input_file": "raw/pointcloud/alaca_cesme_full.ply"}
Output: {"status": "success", "output_path": "processed/v1/01_raw/..."}
"""

import os
import json
import tempfile
from datetime import datetime
import functions_framework
from google.cloud import storage

# Lazy imports for heavy libraries
o3d = None
np = None

def load_dependencies():
    """Lazy load heavy dependencies."""
    global o3d, np
    if o3d is None:
        import open3d as o3d_lib
        import numpy as np_lib
        o3d = o3d_lib
        np = np_lib

# Configuration
PROJECT_ID = "concrete-racer-470219-h8"
BUCKET_NAME = "alaca-cesme-hbim-v6"

@functions_framework.http
def process(request):
    """
    HTTP Cloud Function for Phase 1: Load & Validate.

    Args:
        request: HTTP request object

    Returns:
        JSON response with status and output paths
    """
    # Parse request
    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    input_file = request_json.get("input_file", "raw/pointcloud/alaca_cesme_full.ply")

    print(f"[Phase 1] Starting - Version: {version}, Input: {input_file}")

    try:
        # Load dependencies
        load_dependencies()

        # Initialize GCS client
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "input.ply")
            local_output_ply = os.path.join(tmpdir, "01_raw_pointcloud.ply")
            local_output_json = os.path.join(tmpdir, "01_raw_stats.json")

            # Download from GCS
            print(f"[Phase 1] Downloading: gs://{BUCKET_NAME}/{input_file}")
            blob = bucket.blob(input_file)
            blob.download_to_filename(local_input)
            file_size_mb = os.path.getsize(local_input) / (1024 * 1024)
            print(f"[Phase 1] Downloaded: {file_size_mb:.1f} MB")

            # Load point cloud
            print("[Phase 1] Loading point cloud...")
            pcd = o3d.io.read_point_cloud(local_input)
            points = np.asarray(pcd.points)

            if len(points) == 0:
                return json.dumps({"error": "Point cloud is empty!"}), 400

            # Compute statistics
            bbox = pcd.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            dimensions = max_bound - min_bound

            stats = {
                "file_name": os.path.basename(input_file),
                "point_count": int(len(points)),
                "bounding_box": {
                    "min": min_bound.tolist(),
                    "max": max_bound.tolist(),
                    "dimensions": dimensions.tolist()
                },
                "has_colors": pcd.has_colors(),
                "has_normals": pcd.has_normals(),
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "v6",
                "phase": "01_load"
            }

            print(f"[Phase 1] Loaded {len(points):,} points")
            print(f"[Phase 1] Dimensions: {dimensions[0]:.2f}m x {dimensions[1]:.2f}m x {dimensions[2]:.2f}m")

            # Save outputs locally
            o3d.io.write_point_cloud(local_output_ply, pcd)
            with open(local_output_json, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            # Upload to GCS
            output_base = f"processed/{version}/01_raw"

            ply_blob = bucket.blob(f"{output_base}/01_raw_pointcloud.ply")
            ply_blob.upload_from_filename(local_output_ply)

            json_blob = bucket.blob(f"{output_base}/01_raw_stats.json")
            json_blob.upload_from_filename(local_output_json)

            print(f"[Phase 1] Uploaded to: gs://{BUCKET_NAME}/{output_base}/")

            # Return success response
            response = {
                "phase": "01_load",
                "status": "success",
                "version": version,
                "outputs": {
                    "pointcloud": f"gs://{BUCKET_NAME}/{output_base}/01_raw_pointcloud.ply",
                    "stats": f"gs://{BUCKET_NAME}/{output_base}/01_raw_stats.json"
                },
                "metrics": {
                    "point_count": int(len(points)),
                    "has_colors": pcd.has_colors(),
                    "has_normals": pcd.has_normals(),
                    "dimensions_m": {
                        "x": float(dimensions[0]),
                        "y": float(dimensions[1]),
                        "z": float(dimensions[2])
                    }
                },
                "timestamp": datetime.now().isoformat(),
                "next_phase": "02_preprocess"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 1] ERROR: {str(e)}")
        return json.dumps({
            "phase": "01_load",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
