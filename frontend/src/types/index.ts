/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

export interface DatasetInfo {
  id: string;
  name: string;
  path: string;
  total_images: number;
  valid_images: number;
  corrupted_images: number;
  total_size_mb: number;
  formats: Record<string, number>;
  created_at: string;
  status: string;
}

export interface ClassDefinition {
  id: number;
  name: string;
  description?: string;
  parent_id?: number;
  color?: string;
  attributes?: Record<string, any>;
}

export interface ClassHierarchy {
  classes: ClassDefinition[];
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LabelAnnotation {
  id: number;
  image_id: string;
  class_id: number;
  class_name?: string;
  confidence: number;
  bbox?: BoundingBox;
  segmentation?: {
    polygon?: number[][][];
    mask_path?: string;
  };
  area?: number;
  iscrowd: boolean;
  attributes?: Record<string, any>;
}

export interface ImageLabels {
  image_id: string;
  image_url?: string;
  annotations: LabelAnnotation[];
  status: string;
  processed_at?: string;
  processing_time_ms?: number;
}

export type TaskType = 'object_detection' | 'semantic_segmentation' | 'instance_segmentation';
export type LabelingStrategy = 'rule_based' | 'ai_assisted' | 'hybrid';
export type ExportFormat = 'coco' | 'pascal_voc' | 'yolo';

export interface LabelingJob {
  id: string;
  dataset_id: string;
  task_type: TaskType;
  strategy: LabelingStrategy;
  class_hierarchy: ClassHierarchy;
  confidence_threshold: number;
  models_config: Record<string, any>;
  status: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: Record<string, any>;
  total_images: number;
  processed_images: number;
  failed_images: number;
}

export interface LabelingProgress {
  job_id: string;
  total_images: number;
  processed_images: number;
  failed_images: number;
  current_image?: string;
  current_model?: string;
  estimated_time_remaining?: number;
  status: string;
  errors: string[];
}

export interface SystemStatus {
  status: string;
  version: string;
  gpu_available: boolean;
  gpu_info?: {
    name: string;
    index: number;
    memory_total_gb: number;
    memory_allocated_gb: number;
    memory_cached_gb: number;
  };
  cpu_count: number;
  cpu_usage: number;
  memory_gb: number;
  memory_usage_percent: number;
  disk_space_gb: number;
  active_jobs: number;
  models_loaded: string[];
  device_preference: string;
}
export interface ModelDownloadStatus {
  status: 'not_downloaded' | 'downloading' | 'ready' | 'error';
  progress: number;
  error?: string | null;
}

export type ModelsStatus = Record<string, ModelDownloadStatus>;

export interface ModelCatalogProfile {
  id: string;
  name: string;
  provider: string;
  year: number;
  tasks: string[];
  runtime_status: string;
  availability: string;
  recommended_for: string[];
  strengths: string[];
  constraints: string[];
  research_basis: string;
  score?: number;
  why?: string[];
}

export interface ExportConfig {
  job_id: string;
  format: ExportFormat;
  split_ratios?: Record<string, number>;
  include_validation_report?: boolean;
}

export interface ExportResult {
  export_id: string;
  job_id: string;
  format: ExportFormat;
  output_path: string;
  file_paths: Record<string, string>;
  statistics: Record<string, any>;
  created_at: string;
}
