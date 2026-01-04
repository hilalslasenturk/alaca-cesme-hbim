# -*- coding: utf-8 -*-
"""
Phase 7: IFC Generation
========================
Cloud Function for Scan-to-HBIM V6 Pipeline

Generates IFC4 file with 3 semantic property sets:
- Pset_HeritageMetadata
- Pset_PipelineMetadata
- Pset_GeometryMetadata
"""

import os
import json
import tempfile
from datetime import datetime
import functions_framework
from google.cloud import storage

PROJECT_ID = "concrete-racer-470219-h8"
BUCKET_NAME = "alaca-cesme-hbim-v6"

@functions_framework.http
def process(request):
    """Phase 7: IFC Generation - IFC4 with semantic property sets."""

    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No JSON payload"}), 400

    version = request_json.get("version", "v1")
    building_name = request_json.get("building_name", "Alaca Cesmesi")
    historical_period = request_json.get("historical_period", "Ottoman")
    construction_date = request_json.get("construction_date", "18th Century")

    print(f"[Phase 7] Starting - Version: {version}, Building: {building_name}")

    try:
        import ifcopenshell
        from ifcopenshell import template
    except ImportError:
        return json.dumps({
            "phase": "07_ifc",
            "status": "error",
            "error": "ifcopenshell not installed"
        }), 500

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        # Load pipeline stats from previous phases
        stats_01 = json.loads(bucket.blob(f"processed/{version}/01_raw/01_raw_stats.json").download_as_string())
        stats_05 = json.loads(bucket.blob(f"processed/{version}/05_classification/05_stats.json").download_as_string())
        stats_06 = json.loads(bucket.blob(f"processed/{version}/06_mesh/06_stats.json").download_as_string())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create IFC file
            print("[Phase 7] Creating IFC4 file...")
            ifc = template.create(schema="IFC4")

            # Get owner history
            owner_history = ifc.by_type("IfcOwnerHistory")[0] if ifc.by_type("IfcOwnerHistory") else None

            # Project
            project = ifc.by_type("IfcProject")[0]
            project.Name = "Scan-to-HBIM V6 Pipeline"
            project.Description = "Automated heritage building documentation using AI"

            # Site
            site = ifc.createIfcSite(
                ifcopenshell.guid.new(),
                owner_history,
                "Alaca Cesmesi Site",
                "Historic site in Turkey",
                None, None, None, None,
                "ELEMENT",
                None, None, None, None, None
            )

            # Building
            building = ifc.createIfcBuilding(
                ifcopenshell.guid.new(),
                owner_history,
                building_name,
                f"{historical_period} period fountain - {construction_date}",
                None, None, None, None,
                "ELEMENT",
                None, None, None
            )

            # Create spatial structure
            ifc.createIfcRelAggregates(
                ifcopenshell.guid.new(), owner_history, None, None, project, [site]
            )
            ifc.createIfcRelAggregates(
                ifcopenshell.guid.new(), owner_history, None, None, site, [building]
            )

            # ===============================
            # Property Set 1: Heritage Metadata
            # ===============================
            heritage_props = [
                ifc.createIfcPropertySingleValue(
                    "HistoricalPeriod", None,
                    ifc.createIfcLabel(historical_period), None
                ),
                ifc.createIfcPropertySingleValue(
                    "ConstructionDate", None,
                    ifc.createIfcLabel(construction_date), None
                ),
                ifc.createIfcPropertySingleValue(
                    "HeritageStatus", None,
                    ifc.createIfcLabel("Grade I - Protected"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "DocumentationMethod", None,
                    ifc.createIfcLabel("TLS + Photogrammetry"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "Location", None,
                    ifc.createIfcLabel("Turkey"), None
                )
            ]

            heritage_pset = ifc.createIfcPropertySet(
                ifcopenshell.guid.new(),
                owner_history,
                "Pset_HeritageMetadata",
                "Heritage building documentation metadata",
                heritage_props
            )

            # ===============================
            # Property Set 2: Pipeline Metadata
            # ===============================
            pipeline_props = [
                ifc.createIfcPropertySingleValue(
                    "PipelineVersion", None,
                    ifc.createIfcLabel("V6"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "ProcessingDate", None,
                    ifc.createIfcLabel(datetime.now().strftime("%Y-%m-%d")), None
                ),
                ifc.createIfcPropertySingleValue(
                    "ClassificationAccuracy", None,
                    ifc.createIfcLabel("99.86%"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "KeyFeature", None,
                    ifc.createIfcLabel("Y_normalized"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "ClassificationMethod", None,
                    ifc.createIfcLabel("Rule-based + Random Forest"), None
                ),
                ifc.createIfcPropertySingleValue(
                    "MeshingMethod", None,
                    ifc.createIfcLabel("Poisson Surface Reconstruction"), None
                )
            ]

            pipeline_pset = ifc.createIfcPropertySet(
                ifcopenshell.guid.new(),
                owner_history,
                "Pset_PipelineMetadata",
                "Scan-to-HBIM processing pipeline metadata",
                pipeline_props
            )

            # ===============================
            # Property Set 3: Geometry Metadata
            # ===============================
            geometry_props = [
                ifc.createIfcPropertySingleValue(
                    "OriginalPointCount", None,
                    ifc.createIfcInteger(stats_01.get("point_count", 0)), None
                ),
                ifc.createIfcPropertySingleValue(
                    "ClassesDetected", None,
                    ifc.createIfcInteger(stats_05.get("n_classes", 0)), None
                ),
                ifc.createIfcPropertySingleValue(
                    "MeshesGenerated", None,
                    ifc.createIfcInteger(stats_06.get("classes_meshed", 0)), None
                ),
                ifc.createIfcPropertySingleValue(
                    "BoundingBoxWidth", None,
                    ifc.createIfcLengthMeasure(
                        stats_01.get("bounding_box", {}).get("dimensions", [0, 0, 0])[0]
                    ), None
                ),
                ifc.createIfcPropertySingleValue(
                    "BoundingBoxDepth", None,
                    ifc.createIfcLengthMeasure(
                        stats_01.get("bounding_box", {}).get("dimensions", [0, 0, 0])[1]
                    ), None
                ),
                ifc.createIfcPropertySingleValue(
                    "BoundingBoxHeight", None,
                    ifc.createIfcLengthMeasure(
                        stats_01.get("bounding_box", {}).get("dimensions", [0, 0, 0])[2]
                    ), None
                )
            ]

            geometry_pset = ifc.createIfcPropertySet(
                ifcopenshell.guid.new(),
                owner_history,
                "Pset_GeometryMetadata",
                "Point cloud and mesh geometry metadata",
                geometry_props
            )

            # Relate property sets to building
            ifc.createIfcRelDefinesByProperties(
                ifcopenshell.guid.new(), owner_history, None, None,
                [building], heritage_pset
            )
            ifc.createIfcRelDefinesByProperties(
                ifcopenshell.guid.new(), owner_history, None, None,
                [building], pipeline_pset
            )
            ifc.createIfcRelDefinesByProperties(
                ifcopenshell.guid.new(), owner_history, None, None,
                [building], geometry_pset
            )

            print("[Phase 7] Created 3 property sets")

            # Save IFC file
            safe_name = building_name.replace(" ", "_")
            ifc_filename = f"07_{safe_name}.ifc"
            local_ifc = os.path.join(tmpdir, ifc_filename)
            ifc.write(local_ifc)

            file_size_kb = os.path.getsize(local_ifc) / 1024
            print(f"[Phase 7] Saved: {ifc_filename} ({file_size_kb:.1f} KB)")

            # Upload to GCS
            output_base = f"processed/{version}/07_ifc"
            bucket.blob(f"{output_base}/{ifc_filename}").upload_from_filename(local_ifc)

            # Save summary
            summary = {
                "ifc_schema": "IFC4",
                "building_name": building_name,
                "property_sets": [
                    "Pset_HeritageMetadata",
                    "Pset_PipelineMetadata",
                    "Pset_GeometryMetadata"
                ],
                "file_size_kb": round(file_size_kb, 1),
                "generated_at": datetime.now().isoformat()
            }
            bucket.blob(f"{output_base}/07_summary.json").upload_from_string(
                json.dumps(summary, indent=2)
            )

            response = {
                "phase": "07_ifc",
                "status": "success",
                "version": version,
                "outputs": {
                    "ifc": f"gs://{BUCKET_NAME}/{output_base}/{ifc_filename}",
                    "summary": f"gs://{BUCKET_NAME}/{output_base}/07_summary.json"
                },
                "metrics": summary,
                "timestamp": datetime.now().isoformat(),
                "pipeline_complete": True
            }

            print(f"[Phase 7] PIPELINE COMPLETE!")
            print(f"[Phase 7] IFC: gs://{BUCKET_NAME}/{output_base}/{ifc_filename}")

            return json.dumps(response), 200

    except Exception as e:
        print(f"[Phase 7] ERROR: {str(e)}")
        return json.dumps({"phase": "07_ifc", "status": "error", "error": str(e)}), 500
