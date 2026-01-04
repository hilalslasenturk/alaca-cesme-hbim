# Google Cloud Storage Setup Guide

## Overview

This guide explains how to set up Google Cloud Storage (GCS) for the Scan-to-HBIM pipeline.

---

## 1. Create GCS Bucket

### Using Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Cloud Storage** > **Buckets**
3. Click **Create Bucket**
4. Configure:
   - Name: `alaca-cesme-hbim-v6` (or your choice)
   - Region: `europe-west1` (recommended)
   - Storage class: Standard
   - Access control: Uniform

### Using gcloud CLI

```bash
# Create bucket
gcloud storage buckets create gs://alaca-cesme-hbim-v6 \
    --location=europe-west1 \
    --storage-class=STANDARD

# Verify
gcloud storage buckets list
```

---

## 2. Create Folder Structure

```bash
# Create folders
gsutil mkdir gs://alaca-cesme-hbim-v6/raw/
gsutil mkdir gs://alaca-cesme-hbim-v6/raw/pointcloud/
gsutil mkdir gs://alaca-cesme-hbim-v6/raw/metadata/
gsutil mkdir gs://alaca-cesme-hbim-v6/processed/
gsutil mkdir gs://alaca-cesme-hbim-v6/exports/
gsutil mkdir gs://alaca-cesme-hbim-v6/models/
gsutil mkdir gs://alaca-cesme-hbim-v6/logs/
```

### Expected Structure

```
gs://alaca-cesme-hbim-v6/
├── raw/
│   ├── pointcloud/
│   │   └── alaca_cesme_raw.ply
│   └── metadata/
│       └── heritage_metadata.json
├── processed/
│   └── v1/
│       ├── 01_raw/
│       ├── 02_preprocessed/
│       ├── 03_features/
│       ├── 04_segmentation/
│       ├── 05_classification/
│       ├── 06_mesh/
│       └── 07_ifc/
├── exports/
├── models/
│   └── rf_classifier.joblib
└── logs/
```

---

## 3. Create Service Account

### Using Console

1. Go to **IAM & Admin** > **Service Accounts**
2. Click **Create Service Account**
3. Name: `scan-to-hbim-pipeline`
4. Grant roles:
   - Storage Object Admin
   - Storage Object Viewer
5. Create key (JSON)
6. Download and save securely

### Using gcloud CLI

```bash
# Create service account
gcloud iam service-accounts create scan-to-hbim-pipeline \
    --display-name="Scan-to-HBIM Pipeline"

# Get email
SA_EMAIL=$(gcloud iam service-accounts list \
    --filter="name:scan-to-hbim-pipeline" \
    --format="value(email)")

# Grant bucket access
gsutil iam ch serviceAccount:$SA_EMAIL:objectAdmin \
    gs://alaca-cesme-hbim-v6

# Create key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=$SA_EMAIL
```

---

## 4. Upload Initial Data

```bash
# Upload raw point cloud
gsutil cp alaca_cesme_raw.ply \
    gs://alaca-cesme-hbim-v6/raw/pointcloud/

# Upload metadata
gsutil cp heritage_metadata.json \
    gs://alaca-cesme-hbim-v6/raw/metadata/

# Verify
gsutil ls -r gs://alaca-cesme-hbim-v6/raw/
```

---

## 5. Configure Python Access

### Install Library

```bash
pip install google-cloud-storage
```

### Authentication

**Option 1: Environment Variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Option 2: In Code**
```python
from google.cloud import storage

client = storage.Client.from_service_account_json('credentials.json')
bucket = client.bucket('alaca-cesme-hbim-v6')
```

**Option 3: Google Colab**
```python
from google.colab import auth
auth.authenticate_user()

from google.cloud import storage
client = storage.Client(project='your-project-id')
```

---

## 6. Download/Upload Functions

```python
from google.cloud import storage
import os

BUCKET_NAME = "alaca-cesme-hbim-v6"
PROJECT_ID = "your-project-id"

def download_from_gcs(blob_name, local_path):
    """Download file from GCS."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded: {blob_name}")

def upload_to_gcs(local_path, blob_name):
    """Upload file to GCS."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded: {blob_name}")

# Usage
download_from_gcs("raw/pointcloud/alaca_cesme_raw.ply", "/tmp/input.ply")
upload_to_gcs("/tmp/output.ifc", "processed/v1/07_ifc/result.ifc")
```

---

## 7. Security Best Practices

| Rule | Description |
|------|-------------|
| Never commit credentials | Use .gitignore |
| Use service accounts | Not personal accounts |
| Minimum permissions | Only what's needed |
| Rotate keys regularly | Every 90 days |
| Use environment variables | Not hardcoded paths |

---

## 8. Cost Optimization

| Tip | Details |
|-----|---------|
| Use Standard class | For active data |
| Set lifecycle rules | Auto-delete old versions |
| Use Nearline for archive | For historical data |
| Monitor usage | Set budget alerts |

```bash
# Set lifecycle rule (delete after 365 days)
cat > lifecycle.json << EOF
{
  "rule": [{
    "action": {"type": "Delete"},
    "condition": {"age": 365}
  }]
}
EOF

gsutil lifecycle set lifecycle.json gs://alaca-cesme-hbim-v6
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | Check IAM roles |
| Bucket not found | Verify bucket name |
| Slow upload | Use parallel composite |
| Large file timeout | Use resumable upload |

```bash
# Parallel composite upload for large files
gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
    cp large_file.ply gs://bucket/
```
