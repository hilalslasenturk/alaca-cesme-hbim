#!/usr/bin/env python3
"""
Scan-to-HBIM Full Pipeline Runner
=================================

Runs the complete pipeline from raw point cloud to IFC export.

Usage:
    python run_pipeline.py --input path/to/pointcloud.ply --output outputs/ --version v1

Author: Hilal Sila Senturk
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

try:
    import ifcopenshell
    import ifcopenshell.guid
    HAS_IFC = True
except ImportError:
    HAS_IFC = False
    print("Warning: ifcopenshell not installed. IFC export will be skipped.")

from config import (
    CLASSES, CLASS_NAMES, N_CLASSES,
    PREPROCESSING, FEATURES, SEGMENTATION,
    CLASSIFICATION, MESH_RECONSTRUCTION, IFC_EXPORT,
    OUTPUT_FOLDERS, get_output_path
)


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# PHASE 1: LOAD
# =============================================================================

def phase1_load(input_file: Path, output_dir: Path) -> o3d.geometry.PointCloud:
    """Load and validate point cloud."""
    log("=" * 60)
    log("PHASE 1: LOAD")
    log("=" * 60)

    pcd = o3d.io.read_point_cloud(str(input_file))
    n_points = len(pcd.points)

    log(f"Loaded: {n_points:,} points")
    log(f"Has colors: {pcd.has_colors()}")
    log(f"Has normals: {pcd.has_normals()}")

    # Save
    output_path = output_dir / "01_raw" / "01_raw_pointcloud.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)

    return pcd


# =============================================================================
# PHASE 2: PREPROCESS
# =============================================================================

def phase2_preprocess(pcd: o3d.geometry.PointCloud, output_dir: Path) -> o3d.geometry.PointCloud:
    """Preprocess: voxel, SOR, normals."""
    log("=" * 60)
    log("PHASE 2: PREPROCESS")
    log("=" * 60)

    original = len(pcd.points)

    # Voxel downsampling
    log(f"Voxel downsampling (size={PREPROCESSING['voxel_size']}m)...")
    pcd = pcd.voxel_down_sample(PREPROCESSING['voxel_size'])
    log(f"  {original:,} -> {len(pcd.points):,} points")

    # SOR
    log("Statistical outlier removal...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=PREPROCESSING['sor_k_neighbors'],
        std_ratio=PREPROCESSING['sor_std_ratio']
    )
    log(f"  After SOR: {len(pcd.points):,} points")

    # Normals
    log("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=PREPROCESSING['normal_radius'],
            max_nn=PREPROCESSING['normal_max_nn']
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Save
    output_path = output_dir / "02_preprocessed" / "02_preprocessed.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)

    return pcd


# =============================================================================
# PHASE 3: FEATURES
# =============================================================================

def phase3_features(pcd: o3d.geometry.PointCloud, output_dir: Path) -> np.ndarray:
    """Extract geometric features including Y_normalized."""
    log("=" * 60)
    log("PHASE 3: FEATURES (Croce et al. 2021)")
    log("=" * 60)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_points = len(points)

    # Build KD-tree
    log("Building KD-tree...")
    kdtree = cKDTree(points)

    # Extract features at each scale
    all_features = []
    for radius in FEATURES['radii']:
        log(f"Computing features at radius={radius}m...")
        scale_features = compute_geometric_features(points, normals, kdtree, radius)
        all_features.append(scale_features)

    features = np.hstack(all_features)

    # Add normalized positions (X, Y, Z) - Y is KEY!
    log("Adding normalized position features (X, Y, Z)...")
    x_norm = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-10)
    y_norm = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-10)
    z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-10)

    features = np.column_stack([features, x_norm, y_norm, z_norm])
    log(f"Final features shape: {features.shape}")

    # Save
    output_path = output_dir / "03_features" / "03_features.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)

    return features


def compute_geometric_features(points, normals, kdtree, radius):
    """Compute 9 geometric features at given radius."""
    n_points = len(points)
    features = np.zeros((n_points, 9))
    eps = 1e-10

    for i in range(n_points):
        indices = kdtree.query_ball_point(points[i], radius)
        if len(indices) < 5:
            features[i] = [0.33, 0.33, 0.33, 0, 0, 0.33, 1, 0.5, 0.5]
            continue

        neighbors = points[indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        if cov.ndim < 2:
            continue

        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, eps)
        l1, l2, l3 = eigvals
        l_sum = l1 + l2 + l3

        features[i, 0] = (l1 - l2) / (l1 + eps)  # Linearity
        features[i, 1] = (l2 - l3) / (l1 + eps)  # Planarity
        features[i, 2] = l3 / (l1 + eps)          # Sphericity
        features[i, 3] = np.cbrt(l1 * l2 * l3)    # Omnivariance

        l_norm = eigvals / l_sum
        l_norm = np.maximum(l_norm, eps)
        features[i, 4] = -np.sum(l_norm * np.log(l_norm))  # Eigenentropy

        features[i, 5] = l3 / l_sum               # Surface variation
        features[i, 6] = l_sum                    # Sum eigenvalues
        features[i, 7] = (l1 - l3) / (l1 + eps)   # Anisotropy
        features[i, 8] = 1 - abs(normals[i, 2])   # Verticality

    return features


# =============================================================================
# PHASE 4: SEGMENT
# =============================================================================

def phase4_segment(pcd: o3d.geometry.PointCloud, output_dir: Path) -> np.ndarray:
    """DBSCAN segmentation."""
    log("=" * 60)
    log("PHASE 4: SEGMENT (DBSCAN)")
    log("=" * 60)

    from sklearn.cluster import DBSCAN

    points = np.asarray(pcd.points)

    log(f"Running DBSCAN (eps={SEGMENTATION['dbscan_eps']}, min_samples={SEGMENTATION['dbscan_min_samples']})...")
    clustering = DBSCAN(
        eps=SEGMENTATION['dbscan_eps'],
        min_samples=SEGMENTATION['dbscan_min_samples'],
        n_jobs=-1
    )
    labels = clustering.fit_predict(points)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log(f"Found {n_clusters} clusters")

    # Save
    output_path = output_dir / "04_segmentation" / "04_segment_labels.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

    return labels


# =============================================================================
# PHASE 5: CLASSIFY
# =============================================================================

def phase5_classify(pcd: o3d.geometry.PointCloud, features: np.ndarray, output_dir: Path) -> np.ndarray:
    """Random Forest classification."""
    log("=" * 60)
    log("PHASE 5: CLASSIFY (Random Forest)")
    log("=" * 60)

    points = np.asarray(pcd.points)

    # Generate training labels using rule-based approach
    log("Generating training labels from rules...")
    y_norm = features[:, -2]  # Y_normalized - THE KEY!
    z_norm = features[:, -1]  # Z_normalized

    labels = np.zeros(len(points), dtype=np.int32)
    labels[z_norm < 0.15] = 0  # zemin
    labels[(y_norm > 0.6) & (z_norm < 0.35)] = 1  # seki
    labels[(y_norm < 0.4) & (z_norm > 0.15) & (z_norm < 0.85)] = 2  # ana_cephe
    labels[(y_norm > 0.3) & (y_norm < 0.7) & (z_norm > 0.4) & (z_norm < 0.85)] = 3  # kemer
    labels[z_norm > 0.85] = 4  # sacak

    # Train Random Forest
    log("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=CLASSIFICATION['test_size'],
        random_state=CLASSIFICATION['random_state'],
        stratify=labels
    )

    rf = RandomForestClassifier(
        n_estimators=CLASSIFICATION['n_estimators'],
        max_depth=CLASSIFICATION['max_depth'],
        class_weight=CLASSIFICATION['class_weight'],
        n_jobs=-1,
        random_state=CLASSIFICATION['random_state']
    )
    rf.fit(X_train, y_train)

    # Evaluate
    accuracy = rf.score(X_test, y_test)
    log(f"Accuracy: {accuracy*100:.2f}%")

    # Predict all
    final_labels = rf.predict(features)

    # Save per-class PLY files
    class_dir = output_dir / "05_classification"
    class_dir.mkdir(parents=True, exist_ok=True)

    for class_id, cls in CLASSES.items():
        mask = final_labels == class_id
        if np.sum(mask) > 0:
            class_pcd = pcd.select_by_index(np.where(mask)[0])
            o3d.io.write_point_cloud(str(class_dir / f"{cls.name}.ply"), class_pcd)
            log(f"  {cls.name}: {np.sum(mask):,} points")

    # Save model
    joblib.dump(rf, class_dir / "model.joblib")

    return final_labels


# =============================================================================
# PHASE 6: MESH
# =============================================================================

def phase6_mesh(output_dir: Path) -> dict:
    """Create Poisson meshes."""
    log("=" * 60)
    log("PHASE 6: MESH (Poisson)")
    log("=" * 60)

    class_dir = output_dir / "05_classification"
    mesh_dir = output_dir / "06_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    meshes = {}

    for cls in CLASSES.values():
        ply_path = class_dir / f"{cls.name}.ply"
        if not ply_path.exists():
            continue

        log(f"Processing {cls.name}...")
        pcd = o3d.io.read_point_cloud(str(ply_path))

        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=MESH_RECONSTRUCTION['poisson_depth'],
            scale=MESH_RECONSTRUCTION['poisson_scale']
        )

        # Trim
        trim_pct = MESH_RECONSTRUCTION['trim_percentiles'].get(cls.name, 0)
        if trim_pct > 0:
            densities = np.asarray(densities)
            threshold = np.percentile(densities, trim_pct)
            mesh.remove_vertices_by_mask(densities < threshold)

        mesh.compute_vertex_normals()

        output_path = mesh_dir / f"{cls.name}_poisson.ply"
        o3d.io.write_triangle_mesh(str(output_path), mesh)

        meshes[cls.name] = {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles)
        }
        log(f"  {cls.name}: {len(mesh.triangles):,} triangles")

    return meshes


# =============================================================================
# PHASE 7: IFC
# =============================================================================

def phase7_ifc(output_dir: Path, metadata: dict) -> list:
    """Export to IFC with semantic properties."""
    if not HAS_IFC:
        log("Skipping IFC export (ifcopenshell not installed)")
        return []

    log("=" * 60)
    log("PHASE 7: IFC EXPORT")
    log("=" * 60)

    mesh_dir = output_dir / "06_mesh"
    ifc_dir = output_dir / "07_ifc"
    ifc_dir.mkdir(parents=True, exist_ok=True)

    ifc_files = []

    for cls in CLASSES.values():
        mesh_path = mesh_dir / f"{cls.name}_poisson.ply"
        if not mesh_path.exists():
            continue

        log(f"Creating IFC for {cls.name}...")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        ifc = create_ifc_file(mesh, cls, metadata)

        output_path = ifc_dir / f"alaca_cesme_{cls.name}.ifc"
        ifc.write(str(output_path))
        ifc_files.append(str(output_path))

        log(f"  Saved: {output_path.name}")

    return ifc_files


def create_ifc_file(mesh, cls, metadata):
    """Create IFC file with semantic properties."""
    ifc = ifcopenshell.file(schema="IFC4")

    # Project
    project = ifc.create_entity("IfcProject", ifcopenshell.guid.new())
    project.Name = "Alaca Cesmesi HBIM V6"

    # Units
    length_unit = ifc.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    ifc.create_entity("IfcUnitAssignment", Units=[length_unit])

    # Context
    context = ifc.create_entity("IfcGeometricRepresentationContext",
        ContextType="Model", CoordinateSpaceDimension=3, Precision=1e-5)
    project.RepresentationContexts = [context]

    # Hierarchy
    site = ifc.create_entity("IfcSite", ifcopenshell.guid.new(), Name="Site")
    building = ifc.create_entity("IfcBuilding", ifcopenshell.guid.new(),
        Name=metadata.get("structure_name", "Alaca Cesmesi"))
    storey = ifc.create_entity("IfcBuildingStorey", ifcopenshell.guid.new(), Name="Ground")

    ifc.create_entity("IfcRelAggregates", ifcopenshell.guid.new(),
        RelatingObject=project, RelatedObjects=[site])
    ifc.create_entity("IfcRelAggregates", ifcopenshell.guid.new(),
        RelatingObject=site, RelatedObjects=[building])
    ifc.create_entity("IfcRelAggregates", ifcopenshell.guid.new(),
        RelatingObject=building, RelatedObjects=[storey])

    # Geometry
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    coord_list = [[float(v[0]), float(v[1]), float(v[2])] for v in vertices]
    coords = ifc.create_entity("IfcCartesianPointList3D", CoordList=coord_list)

    indices = [[int(t[0]+1), int(t[1]+1), int(t[2]+1)] for t in triangles]
    face_set = ifc.create_entity("IfcTriangulatedFaceSet",
        Coordinates=coords, CoordIndex=indices, Closed=False)

    shape_rep = ifc.create_entity("IfcShapeRepresentation",
        ContextOfItems=context, RepresentationIdentifier="Body",
        RepresentationType="Tessellation", Items=[face_set])
    product_rep = ifc.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])

    # Placement
    origin = ifc.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    placement = ifc.create_entity("IfcLocalPlacement",
        RelativePlacement=ifc.create_entity("IfcAxis2Placement3D", Location=origin))

    # Element
    element = ifc.create_entity(cls.ifc_class, ifcopenshell.guid.new())
    element.Name = cls.name_tr
    element.ObjectPlacement = placement
    element.Representation = product_rep

    ifc.create_entity("IfcRelContainedInSpatialStructure", ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[element])

    return ifc


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(input_file: str, output_dir: str = "outputs", version: str = "v1"):
    """Run the complete Scan-to-HBIM pipeline."""

    start_time = time.time()

    input_path = Path(input_file)
    output_path = Path(output_dir) / version

    log("=" * 60)
    log("SCAN-TO-HBIM PIPELINE V6")
    log("=" * 60)
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")
    log(f"Version: {version}")

    # Create output directories
    for folder in OUTPUT_FOLDERS:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    # Run phases
    pcd = phase1_load(input_path, output_path)
    pcd = phase2_preprocess(pcd, output_path)
    features = phase3_features(pcd, output_path)
    segments = phase4_segment(pcd, output_path)
    labels = phase5_classify(pcd, features, output_path)
    meshes = phase6_mesh(output_path)

    metadata = {"structure_name": "Alaca Cesmesi"}
    ifc_files = phase7_ifc(output_path, metadata)

    elapsed = time.time() - start_time

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)
    log(f"Total time: {elapsed:.1f} seconds")
    log(f"IFC files: {len(ifc_files)}")

    return {
        "status": "success",
        "version": version,
        "elapsed_seconds": elapsed,
        "ifc_files": ifc_files
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan-to-HBIM Pipeline")
    parser.add_argument("--input", required=True, help="Input point cloud file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--version", default="v1", help="Version tag")

    args = parser.parse_args()

    result = run_pipeline(args.input, args.output, args.version)
    print(json.dumps(result, indent=2))
