# n8n Workflow Setup Guide

## Overview

This guide explains how to set up the n8n workflow for automating the Scan-to-HBIM pipeline.

---

## 1. n8n Instance

### Option 1: n8n Cloud (Recommended)

1. Go to [n8n.io](https://n8n.io)
2. Sign up for n8n Cloud
3. Your instance URL: `https://your-name.app.n8n.cloud`

### Option 2: Self-hosted

```bash
# Docker
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# Access at http://localhost:5678
```

---

## 2. Import Workflow

1. Open n8n
2. Go to **Workflows** > **Import from File**
3. Select `n8n/scan_to_hbim_v6_workflow.json`
4. Click **Import**

---

## 3. Configure Credentials

### Google Cloud Storage

1. Go to **Credentials** > **Add Credential**
2. Select **Google Cloud Storage**
3. Upload service account JSON
4. Test connection

### Email (Optional)

1. Add **SMTP** credential for notifications
2. Configure:
   - Host: smtp.gmail.com
   - Port: 587
   - User: your email
   - Password: app password

---

## 4. Workflow Nodes

| Node | Type | Purpose |
|------|------|---------|
| Webhook Trigger | Trigger | Start pipeline on file upload |
| Phase 1-7 | HTTP Request | Call Colab notebooks |
| Save Log | GCS | Save execution log |
| Notify | Email | Send completion notification |

### Webhook URL

```
https://your-name.app.n8n.cloud/webhook/scan-to-hbim-v6
```

### Trigger Payload Example

```json
{
  "version": "v1",
  "input_file": "gs://bucket/raw/pointcloud/scan.ply",
  "email": "user@example.com"
}
```

---

## 5. Colab Integration

Each phase node calls a Colab notebook via HTTP:

```yaml
Node: Phase 1 - Load
Method: POST
URL: https://colab.research.google.com/...
Body:
  version: "{{ $json.version }}"
  input_file: "{{ $json.input_file }}"
Timeout: 5 minutes
```

### Colab Execution Options

**Option A: Manual Trigger**
- Open notebook in Colab
- Run all cells
- n8n waits for webhook response

**Option B: Colab API (Advanced)**
- Use Colab Enterprise or
- Google Cloud Vertex AI Workbench

---

## 6. Webhook Configuration

### GCS Webhook (Cloud Functions)

Create a Cloud Function to trigger n8n on file upload:

```python
# cloud_function/main.py
import requests

def trigger_n8n(event, context):
    """Trigger n8n when file uploaded to GCS."""

    file_name = event['name']
    bucket = event['bucket']

    # Only trigger for raw point clouds
    if not file_name.startswith('raw/pointcloud/'):
        return

    webhook_url = "https://your-name.app.n8n.cloud/webhook/scan-to-hbim-v6"

    payload = {
        "version": "v1",
        "input_file": f"gs://{bucket}/{file_name}",
        "bucket": bucket
    }

    response = requests.post(webhook_url, json=payload)
    print(f"Triggered n8n: {response.status_code}")
```

Deploy:
```bash
gcloud functions deploy trigger_n8n \
    --runtime python39 \
    --trigger-resource your-bucket \
    --trigger-event google.storage.object.finalize
```

---

## 7. Error Handling

### Retry Configuration

```yaml
Settings:
  Retry On Fail: true
  Max Tries: 3
  Wait Between Tries: 5000ms
```

### Error Workflow

Create error handling workflow:

1. **Catch Error** node after each phase
2. **Send Email** with error details
3. **Save to GCS** error log

---

## 8. Monitoring

### Execution History

- View in n8n: **Executions** tab
- Filter by status: Success, Error, Running

### Logging

Each phase saves status to GCS:
```
gs://bucket/logs/pipeline_2026-01-04_14-30.json
```

---

## 9. Workflow Diagram

```
┌─────────────┐
│   Webhook   │
│   Trigger   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 1    │ → Colab: 01_Load.ipynb
│    Load     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 2    │ → Colab: 02_Preprocess.ipynb
│ Preprocess  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 3    │ → Colab: 03_Features.ipynb
│  Features   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 4    │ → Colab: 04_Segment.ipynb
│   Segment   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 5    │ → Colab: 05_Classify.ipynb
│  Classify   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 6    │ → Colab: 06_Mesh.ipynb
│    Mesh     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Phase 7    │ → Colab: 07_IFC.ipynb
│    IFC      │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Save Log   │────>│   Notify    │
└─────────────┘     └─────────────┘
```

---

## 10. Testing

### Manual Test

```bash
curl -X POST \
  https://your-name.app.n8n.cloud/webhook/scan-to-hbim-v6 \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v1",
    "input_file": "gs://bucket/raw/pointcloud/test.ply"
  }'
```

### Expected Response

```json
{
  "status": "triggered",
  "execution_id": "abc123"
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Webhook not triggering | Check URL, verify credentials |
| Colab timeout | Increase timeout in HTTP node |
| GCS permission denied | Verify service account roles |
| Email not sending | Check SMTP credentials |
