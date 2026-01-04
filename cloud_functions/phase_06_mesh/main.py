# -*- coding: utf-8 -*-
"""
Phase 6: Mesh Generation
=========================
Cloud Function for Scan-to-HBIM V6 Pipeline

Poisson surface reconstruction per semantic class
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
    """Phase 6: Mesh Generation - Poisson reconstruction per class."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    depth = request_json.get("depth", 9)
    density_threshold = request_json.get("density_threshold", 0.01)

    print(f"[Phase 6] Starting - Version: {version}, depth={depth}")

    try:
        load_dependencies()
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_pcd = os.path.join(tmpdir, "input.ply")
            local_labels = os.path.join(tmpdir, "labels.npy")

            # Download classified point cloud and labels
            pcd_path = f"processed/{version}/05_classification/05_classified_pointcloud.ply"
            labels_path = f"processed/{version}/05_classification/05_class_labels.npy"

            print(f"[Phase 6] Downloading data...")
            bucket.blob(pcd_path).download_to_filename(local_pcd)
            bucket.blob(labels_path).download_to_filename(local_labels)

            # Load
            pcd = o3d.io.read_point_cloud(local_pcd)
            labels = np.load(local_labels)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            print(f"[Phase 6] Loaded {len(points):,} points")

            # Generate mesh per class
            mesh_stats = {}
            output_base = f"processed/{version}/06_mesh"

            for class_id, class_name in enumerate(CLASS_NAMES):
                mask = labels == class_id
                n_class_points = mask.sum()

                if n_class_points < 100:
                    print(f"[Phase 6] Skipping {class_name} ({n_class_points} points)")
                    continue

                print(f"[Phase 6] Meshing {class_name} ({n_class_points:,} points)")

                # Create class point cloud
                class_pcd = o3d.geometry.PointCloud()
                class_pcd.points = o3d.utility.Vector3dVector(points[mask])

                # Ensure normals exist
                if len(normals) > 0:
                    class_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
                else:
                    class_pcd.estimate_normals()

                if colors is not None:
                    class_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

                try:
                    # Poisson reconstruction
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        class_pcd, depth=depth
                    )

                    # Remove low-density vertices (trim mesh)
                    densities = np.asarray(densities)
                    threshold = np.quantile(densities, density_threshold)
                    vertices_to_remove = densities < threshold
                    mesh.remove_vertices_by_mask(vertices_to_remove)

                    # Clean mesh
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_duplicated_vertices()

                    n_vertices = len(mesh.vertices)
                    n_triangles = len(mesh.triangles)

                    print(f"[Phase 6]   {class_name}: {n_vertices:,} vertices, {n_triangles:,} triangles")

                    # Save mesh
                    mesh_file = os.path.join(tmpdir, f"06_mesh_{class_name}.ply")
                    o3d.io.write_triangle_mesh(mesh_file, mesh)

                    # Upload
                    bucket.blob(f"{output_base}/06_mesh_{class_name}.ply").upload_from_filename(mesh_file)

                    mesh_stats[class_name] = {
                        "input_points": int(n_class_points),
                        "vertices": n_vertices,
                        "triangles": n_triangles
                    }

                except Exception as e:
                    print(f"[Phase 6]   WARNING: Failed to mesh {class_name}: {e}")
                    mesh_stats[class_name] = {"error": str(e)}

            # Save stats
            stats = {
                "classes_meshed": len([m for m in mesh_stats.values() if "vertices" in m]),
                "depth": depth,
                "mesh_stats": mesh_stats
            }
            bucket.blob(f"{output_base}/06_stats.json").upload_from_string(
                json.dumps(stats, indent=2)
            )

            response = {
                "phase": "06_mesh",
                "status": "success",
                "version": version,
                "outputs": {
                    "meshes": f"gs://{BUCKET_NAME}/{output_base}/",
                    "stats": f"gs://{BUCKET_NAME}/{output_base}/06_stats.json"
                },
                "metrics": stats,
                "timestamp": datetime.now().isoformat(),
                "next_phase": "07_ifc"
            }

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 6] ERROR: {str(e)}")
        return json.dumps({"phase": "06_mesh", "status": "error", "error": str(e)}), 500
