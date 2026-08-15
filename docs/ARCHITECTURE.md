# AutoLabeler Architecture

## System Overview

AutoLabeler is a complete AI-powered image dataset labeling application with a modular architecture designed for scalability and extensibility.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Dashboard     │  │ Dataset Upload│  │ Class Manager │  │ Job Monitor      │ │
│  │ - System stats│  │ - ZIP/Folder  │  │ - Hierarchical│  │ - Real-time      │ │
│  │ - GPU status  │  │ - Validation  │  │   class mgmt  │  │   progress       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐                                          │
│  │ Labeling Job  │  │ Export        │                                          │
│  │ - Task config │  │ - COCO/VOC    │                                          │
│  │ - Model select│  │   YOLO        │                                          │
│  └──────────────┘  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ /health       │  │ /datasets     │  │ /jobs         │  │ /export          │ │
│  │ /models       │  │ /upload       │  │ /start        │  │ /download        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Services Layer                                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐ ││
│  │  │Preprocessor │  │ ModelManager│  │LabelService │  │ ExportService  │ ││
│  │  │ - Validate  │  │ - YOLOv8    │  │ - Pipeline  │  │ - COCO JSON    │ ││
│  │  │ - Normalize │  │ - SAM       │  │ - Progress  │  │ - Pascal VOC   │ ││
│  │  │ - Deduplicate│ │ - Detectron2│  │ - WebSocket │  │ - YOLO format  │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI MODELS                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │ YOLOv8 (Detect) │  │ SAM (Segment)   │  │ Detectron2 (Instance Seg)   │ │
│  │ - 80 COCO classes│ │ - Vit-H/L/B     │  │ - Mask R-CNN                │ │
│  │ - Bounding boxes│  │ - Point/box     │  │ - Panoptic FPN              │ │
│  │ - Masks         │  │   prompts       │  │ - Keypoint detection        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Backend Architecture

### Core Components

#### 1. Configuration (`app/core/config.py`)
- Centralized settings management using Pydantic
- Environment variable support
- Path management for uploads, outputs, and models
- GPU/CPU configuration

#### 2. Logging (`app/core/logging.py`)
- Rich console output with colored logs
- File-based logging for persistence
- Structured logging with JSON support

#### 3. Data Models (`app/models/schemas.py`)
- Pydantic models for request/response validation
- Type-safe data structures
- Hierarchical class definitions

### Services Layer

#### 1. Preprocessor Service (`app/services/preprocessor.py`)

**Responsibilities:**
- Image validation (format, corruption, dimensions)
- Dataset integrity checks
- Format normalization
- Duplicate detection using perceptual hashing
- Parallel processing with ThreadPoolExecutor

**Key Methods:**
```python
validate_image(image_path) -> (bool, ImageInfo, error)
process_dataset(dataset_path) -> DatasetInfo
normalize_image(image_path, target_size) -> np.ndarray
detect_duplicates(image_paths) -> List[Set[str]]
```

#### 2. Model Manager Service (`app/services/model_manager.py`)

**Responsibilities:**
- Model loading and lifecycle management
- Inference pipeline orchestration
- Confidence calibration
- Uncertainty estimation
- GPU memory management

**Supported Models:**
- **YOLOv8**: Object detection, instance segmentation
- **SAM**: Semantic segmentation, mask generation
- **Detectron2**: Instance segmentation (optional)

**Key Methods:**
```python
load_yolo(model_name="yolov8x.pt") -> YOLO
detect_objects_yolo(image, conf_threshold) -> List[LabelAnnotation]
segment_with_sam(image, bounding_boxes) -> List[SegmentationMask]
calculate_confidence_scores(annotations) -> List[float]
estimate_uncertainty(annotation, all_annotations) -> Dict[str, float]
```

#### 3. Labeling Service (`app/services/labeler.py`)

**Responsibilities:**
- Job creation and management
- Pipeline orchestration
- Progress tracking
- WebSocket updates
- Result persistence

**Labeling Pipeline:**
1. Load dataset metadata
2. Load required models
3. Process images in batches
4. Run detection/segmentation
5. Map to user-defined classes
6. Calculate uncertainty
7. Save results

**Key Methods:**
```python
create_job(dataset_info, task_type, class_hierarchy) -> LabelingJob
run_job(job_id, progress_callback) -> List[ImageLabels]
process_image(image_info, task_type, class_hierarchy) -> ImageLabels
stream_progress(job_id) -> AsyncGenerator[LabelingProgress]
```

#### 4. Export Service (`app/services/exporter.py`)

**Responsibilities:**
- Format conversion
- Train/val/test splitting
- Statistics generation
- File organization

**Supported Formats:**
- **COCO JSON**: Standard format with images, annotations, categories
- **Pascal VOC**: XML format per image
- **YOLO**: TXT files with normalized coordinates

**Key Methods:**
```python
export_dataset(job_id, results, class_hierarchy, config) -> ExportResult
export_coco(splits, class_hierarchy, output_dir) -> Dict[str, str]
export_pascal_voc(splits, class_hierarchy, output_dir) -> Dict[str, str]
export_yolo(splits, class_hierarchy, output_dir) -> Dict[str, str]
```

## Frontend Architecture

### Component Structure

```
src/
├── components/ui/          # shadcn/ui components (pre-built)
├── pages/
│   ├── Dashboard.tsx       # System status and overview
│   ├── DatasetUpload.tsx   # File upload with drag-and-drop
│   ├── ClassManager.tsx    # Hierarchical class editor
│   ├── LabelingJob.tsx     # Job configuration
│   ├── JobMonitor.tsx      # Real-time progress tracking
│   └── Export.tsx          # Export configuration and download
├── services/
│   └── api.ts              # API client functions
├── hooks/
│   └── useApi.ts           # Custom React hooks
├── types/
│   └── index.ts            # TypeScript type definitions
├── App.tsx                 # Main application component
└── App.css                 # Application styles
```

### State Management

- **Local State**: React useState for component-level state
- **Server State**: React Query pattern with custom hooks
- **Real-time Updates**: WebSocket for job progress

### Key Features

1. **Dashboard**
   - System health monitoring
   - GPU/CPU resource display
   - Active jobs counter

2. **Dataset Upload**
   - Drag-and-drop ZIP upload
   - Folder upload support
   - Progress tracking
   - Validation results

3. **Class Manager**
   - Hierarchical class structure
   - Color assignment
   - Parent-child relationships

4. **Labeling Job**
   - Task type selection
   - Strategy configuration
   - Confidence threshold

5. **Job Monitor**
   - Real-time progress
   - Statistics display
   - Error tracking

6. **Export**
   - Format selection
   - Train/val/test split
   - Download management

## Data Flow

### Upload Flow
```
User -> Drop files -> Frontend -> POST /datasets/upload
                                      |
                                      v
Backend -> Save ZIP -> Extract -> Validate images
                                      |
                                      v
                              Save metadata.json
                                      |
                                      v
                              Return DatasetInfo
```

### Labeling Flow
```
User -> Create Job -> POST /jobs
                         |
                         v
              Start Job -> POST /jobs/{id}/start
                         |
                         v
              Load Models -> Process Images (batch)
                         |
                         v
              WebSocket Updates -> Save Results
                         |
                         v
              Return Completion
```

### Export Flow
```
User -> Configure Export -> POST /export
                               |
                               v
                    Load Results -> Split Dataset
                               |
                               v
                    Convert Format -> Save Files
                               |
                               v
                    Return ExportResult -> Download ZIP
```

## Performance Optimizations

### Backend
1. **Batch Processing**: Process images in configurable batches
2. **Parallel Loading**: Multi-threaded image validation
3. **GPU Utilization**: Automatic GPU detection and usage
4. **Model Caching**: Keep models loaded between requests
5. **Memory Management**: Unload models when not needed

### Frontend
1. **Lazy Loading**: Components loaded on demand
2. **Progressive Updates**: WebSocket for real-time progress
3. **Optimistic UI**: Immediate feedback on user actions
4. **Debounced Requests**: Reduce API calls for search/filter

## Security Considerations

1. **File Validation**: Check file types and sizes
2. **Path Sanitization**: Prevent directory traversal
3. **Rate Limiting**: Prevent abuse of API endpoints
4. **CORS**: Configured for local development

## Extensibility

### Adding New Models
1. Create model wrapper in `app/services/model_manager.py`
2. Add model info to `/models` endpoint
3. Update frontend model selection

### Adding New Export Formats
1. Implement export method in `app/services/exporter.py`
2. Add format to ExportFormat enum
3. Update frontend format selection

### Adding New Task Types
1. Add task type to TaskType enum
2. Implement processing logic in labeling service
3. Update frontend task selection
