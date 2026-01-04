#!/bin/bash
# Deploy all Scan-to-HBIM V6 Cloud Functions
# Usage: ./deploy_all.sh

set -e

PROJECT_ID="concrete-racer-470219-h8"
REGION="europe-west1"
RUNTIME="python311"
MEMORY="2048MB"
TIMEOUT="540s"  # 9 minutes

echo "=============================================="
echo "Deploying Scan-to-HBIM V6 Cloud Functions"
echo "=============================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=============================================="

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "[1/8] Enabling required APIs..."
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com

# Deploy Phase 1: Load
echo "[2/8] Deploying phase-01-load..."
cd phase_01_load
gcloud functions deploy phase-01-load \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 2: Preprocess
echo "[3/8] Deploying phase-02-preprocess..."
cd phase_02_preprocess
gcloud functions deploy phase-02-preprocess \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 3: Features
echo "[4/8] Deploying phase-03-features..."
cd phase_03_features
gcloud functions deploy phase-03-features \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=4096MB \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 4: Segment
echo "[5/8] Deploying phase-04-segment..."
cd phase_04_segment
gcloud functions deploy phase-04-segment \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 5: Classify
echo "[6/8] Deploying phase-05-classify..."
cd phase_05_classify
gcloud functions deploy phase-05-classify \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 6: Mesh
echo "[7/8] Deploying phase-06-mesh..."
cd phase_06_mesh
gcloud functions deploy phase-06-mesh \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=4096MB \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

# Deploy Phase 7: IFC
echo "[8/8] Deploying phase-07-ifc..."
cd phase_07_ifc
gcloud functions deploy phase-07-ifc \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=. \
    --entry-point=process \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --set-env-vars="BUCKET_NAME=alaca-cesme-hbim-v6"
cd ..

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "Function URLs:"
gcloud functions list --filter="name~phase" --format="table(name,httpsTrigger.url)"
echo ""
echo "Test with:"
echo 'curl -X POST https://$REGION-$PROJECT_ID.cloudfunctions.net/phase-01-load -H "Content-Type: application/json" -d "{\"version\":\"v1\",\"input_file\":\"raw/pointcloud/alaca_cesme_full.ply\"}"'
