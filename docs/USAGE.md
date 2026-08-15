# AutoLabeler Usage Guide

## Quick Start

### 1. Start the Backend

Open a terminal and run:

```cmd
cd C:\auto-labeler\backend
.\venv\Scripts\activate
python -m app.main
```

The backend will start at `http://localhost:8000`

### 2. Start the Frontend

Open another terminal and run:

```cmd
cd C:\auto-labeler\frontend
npm run dev
```

The frontend will start at `http://localhost:5173`

### 3. Open Browser

Navigate to `http://localhost:5173` to access the application.

## Workflow

### Step 1: Upload Dataset

1. Go to **Upload Dataset** page
2. Choose upload method:
   - **ZIP Upload**: Select or drag a ZIP file containing images
   - **Folder Upload**: Select multiple image files
3. Optionally provide a dataset name
4. Wait for validation to complete
5. Note the Dataset ID for later use

**Supported Formats:** JPG, JPEG, PNG, BMP, TIFF, WebP

**Maximum Dataset Size:** Limited by available disk space

### Step 2: Define Classes

1. Go to **Labeling Job** page
2. In the Class Manager section:
   - Click "Add Class" to create a new class
   - Enter class name and description
   - Select a color for visualization
   - For hierarchical classes, use "Add Child" to create parent-child relationships
3. Define all classes you want to detect/segment

**Example Class Hierarchy:**
```
Vehicle (parent)
├── Car
├── Truck
├── Motorcycle
└── Bicycle

Person (parent)
├── Adult
└── Child
```

### Step 3: Configure Labeling Job

1. Select your uploaded dataset
2. Choose **Task Type**:
   - **Object Detection**: Bounding boxes around objects
   - **Instance Segmentation**: Bounding boxes + pixel masks
   - **Semantic Segmentation**: Pixel-level classification
3. Select **Labeling Strategy**:
   - **AI-Assisted**: Uses pretrained models (recommended)
   - **Rule-Based**: Uses custom rules (future feature)
   - **Hybrid**: Combines both approaches (future feature)
4. Set **Confidence Threshold** (0.1 - 0.95):
   - Lower values: More labels, potentially less accurate
   - Higher values: Fewer labels, more confident
   - Recommended: 0.5 for balanced results

### Step 4: Start Labeling

1. Click "Create Labeling Job"
2. Review job configuration
3. Click "Start Job" to begin processing
4. Monitor progress in real-time:
   - Images processed
   - Processing speed
   - Current image being processed
   - Errors (if any)

**Processing Time Estimates:**
- 1,000 images: ~5-10 minutes (GPU), ~30-60 minutes (CPU)
- 10,000 images: ~1-2 hours (GPU), ~5-10 hours (CPU)
- 50,000 images: ~5-10 hours (GPU), ~1-2 days (CPU)

### Step 5: Export Results

1. Go to **Export** page
2. Select the completed job
3. Choose **Export Format**:
   - **COCO JSON**: Standard format, widely supported
   - **Pascal VOC**: XML format, good for legacy systems
   - **YOLO**: Text format, optimized for YOLO training
4. Configure **Dataset Split**:
   - Training: 70% (default)
   - Validation: 15% (default)
   - Test: 15% (default)
5. Set advanced options:
   - Include unlabeled images
   - Minimum confidence filter
6. Click "Export Dataset"
7. Download the ZIP file when ready

## Advanced Usage

### Using the API Directly

The backend provides a REST API at `http://localhost:8000/api/v1`

**API Documentation:** `http://localhost:8000/docs`

**Example: Create a labeling job via curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "task_type": "object_detection",
    "class_hierarchy": {
      "classes": [
        {"id": 1, "name": "person", "color": "#3b82f6"},
        {"id": 2, "name": "car", "color": "#ef4444"}
      ]
    },
    "confidence_threshold": 0.5
  }'
```

### Custom Model Configuration

Edit `backend/app/core/config.py` to customize:

```python
# Model settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS_PER_IMAGE = 300

# YOLO model variant
YOLOV8_MODEL = "yolov8x.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# SAM model variant
SAM_MODEL = "sam_vit_h.pth"  # Options: vit_h, vit_l, vit_b
```

### Performance Tuning

For large datasets (50K+ images), adjust in `config.py`:

```python
# Reduce batch size if running out of memory
BATCH_SIZE = 8  # Default: 16

# Reduce parallel workers
MAX_WORKERS = 2  # Default: 4

# Limit image size
MAX_IMAGE_SIZE = 1024  # Default: 2048
```

## Troubleshooting

### Backend Won't Start

**Problem:** `ModuleNotFoundError` or import errors

**Solution:**
```cmd
cd backend
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Out of Memory

**Problem:** CUDA out of memory error

**Solutions:**
1. Reduce batch size in config.py
2. Use smaller model variant (yolov8s instead of yolov8x)
3. Process images in smaller chunks
4. Use CPU instead of GPU (slower but more memory)

### Slow Processing

**Problem:** Labeling is very slow

**Solutions:**
1. Ensure GPU is being used (check Dashboard)
2. Close other applications using GPU
3. Reduce image size in config.py
4. Use smaller model variant

### Model Download Fails

**Problem:** Models fail to download

**Solution:**
```cmd
cd backend
.\venv\Scripts\activate
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

### Frontend Build Errors

**Problem:** npm install fails

**Solution:**
```cmd
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Best Practices

### Dataset Preparation

1. **Organize Images**: Group images by category if possible
2. **Remove Duplicates**: Use the duplicate detection feature
3. **Consistent Naming**: Use descriptive filenames
4. **Quality Check**: Remove corrupted or very low-quality images

### Class Definition

1. **Be Specific**: Use clear, specific class names
2. **Avoid Overlap**: Classes should be mutually exclusive
3. **Hierarchy**: Use parent-child relationships for related classes
4. **Colors**: Choose distinct colors for visualization

### Labeling Strategy

1. **Start Conservative**: Use higher confidence threshold (0.7+)
2. **Review Results**: Check a sample of labeled images
3. **Iterate**: Adjust threshold based on results
4. **Export Multiple**: Try different formats for your use case

### Export Configuration

1. **Balanced Split**: Keep train/val/test ratio around 70/15/15
2. **Confidence Filter**: Remove low-confidence predictions
3. **Format Selection**:
   - Use COCO for general-purpose datasets
   - Use YOLO for YOLO model training
   - Use Pascal VOC for legacy compatibility

## Tips for Large Datasets (50K+ images)

1. **Pre-validate**: Test with a small subset first
2. **Monitor Resources**: Watch GPU memory and CPU usage
3. **Batch Export**: Export in chunks if needed
4. **Backup**: Keep original dataset backed up
5. **Resume**: Jobs can be stopped and restarted

## Keyboard Shortcuts

- **Ctrl+1**: Dashboard
- **Ctrl+2**: Upload Dataset
- **Ctrl+3**: Labeling Job
- **Ctrl+4**: Export

## Getting Help

1. Check API documentation: `http://localhost:8000/docs`
2. Review logs: `backend/logs/autolabeler.log`
3. Check system status on Dashboard
4. Verify GPU availability in System Status card
