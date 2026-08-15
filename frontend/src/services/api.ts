/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

/// <reference types="vite/client" />
import axios, { AxiosProgressEvent } from 'axios';
import {
  DatasetInfo, SystemStatus, ClassHierarchy, TaskType,
  LabelingStrategy, LabelingJob, ExportConfig, ExportResult,
  LabelingProgress, ModelsStatus, ModelCatalogProfile
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async (): Promise<SystemStatus> => {
  const response = await api.get('/health');
  return response.data;
};

export const uploadDataset = async (
  file: File,
  datasetName?: string,
  onProgress?: (progress: number) => void
): Promise<DatasetInfo> => {
  const formData = new FormData();
  formData.append('file', file);
  if (datasetName) {
    formData.append('dataset_name', datasetName);
  }

  const response = await api.post('/datasets/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent: AxiosProgressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  });
  return response.data;
};

export const getDatasets = async (): Promise<DatasetInfo[]> => {
  const response = await api.get('/datasets');
  return response.data;
};

export const getDataset = async (datasetId: string): Promise<DatasetInfo> => {
  const response = await api.get(`/datasets/${datasetId}`);
  return response.data;
};

export const createLabelingJob = async (
  datasetId: string,
  taskType: TaskType,
  classHierarchy: ClassHierarchy,
  strategy: LabelingStrategy = 'ai_assisted',
  confidenceThreshold: number = 0.5,
  device: string = 'auto',
  customClasses?: string[],
  modelsConfig?: Record<string, any>
): Promise<LabelingJob> => {
  const response = await api.post('/jobs', {
    dataset_id: datasetId,
    task_type: taskType,
    class_hierarchy: classHierarchy,
    strategy,
    confidence_threshold: confidenceThreshold,
    device: device,
    custom_classes: customClasses,
    models_config: modelsConfig
  });
  return response.data;
};

export const startLabelingJob = async (jobId: string): Promise<{ message: string; job_id: string }> => {
  const response = await api.post(`/jobs/${jobId}/start`);
  return response.data;
};

export const stopLabelingJob = async (jobId: string): Promise<{ message: string; job_id: string }> => {
  const response = await api.post(`/jobs/${jobId}/stop`);
  return response.data;
};

export const deleteLabelingJob = async (jobId: string): Promise<{ message: string; job_id: string }> => {
  const response = await api.delete(`/jobs/${jobId}`);
  return response.data;
};

export const getLabelingJobs = async (): Promise<LabelingJob[]> => {
  const response = await api.get('/jobs');
  return response.data;
};

export const getJob = async (jobId: string): Promise<LabelingJob> => {
  const response = await api.get(`/jobs/${jobId}`);
  return response.data;
};

export const getJobProgress = async (jobId: string): Promise<LabelingProgress> => {
  const response = await api.get(`/jobs/${jobId}/progress`);
  return response.data;
};

export const exportLabels = async (config: ExportConfig): Promise<ExportResult> => {
  const response = await api.post('/export', config);
  return response.data;
};

export const getModelsStatus = async (): Promise<ModelsStatus> => {
  const response = await api.get('/models/status');
  return response.data;
};

export const triggerModelDownload = async (modelName: string): Promise<{ message: string }> => {
  const response = await api.post(`/models/download?model_name=${modelName}`);
  return response.data;
};

export const getJobResults = async (jobId: string): Promise<{ job_id: string, total_images: number, results: any[] }> => {
  const response = await api.get(`/jobs/${jobId}/results`);
  return response.data;
};

export const updateJobResults = async (jobId: string, results: any[]): Promise<{ status: string }> => {
  const response = await api.put(`/jobs/${jobId}/results`, results);
  return response.data;
};

export const setSystemDevice = async (device: string): Promise<{ status: string, device: string }> => {
  const response = await api.post(`/system/device?device=${device}`);
  return response.data;
};

export const refineJobResult = async (jobId: string, imageId: string, prompt: string): Promise<any> => {
  const response = await api.post(`/jobs/${jobId}/refine`, { image_id: imageId, prompt });
  return response.data;
};

export const createWorkspace = async (name: string, description?: string): Promise<any> => {
  const response = await api.post('/workspaces', { name, description });
  return response.data;
};

export const getWorkspaces = async (): Promise<any[]> => {
  const response = await api.get('/workspaces');
  return response.data;
};

export const createProject = async (workspaceId: string, name: string, description?: string): Promise<any> => {
  const response = await api.post('/projects', { workspace_id: workspaceId, name, description });
  return response.data;
};

export const getProjects = async (): Promise<any[]> => {
  const response = await api.get('/projects');
  return response.data;
};

export const getDatasetVersions = async (datasetId: string): Promise<any[]> => {
  const response = await api.get(`/datasets/${datasetId}/versions`);
  return response.data;
};

export const getDatasetLineage = async (datasetId: string): Promise<any[]> => {
  const response = await api.get(`/datasets/${datasetId}/lineage`);
  return response.data;
};

export const getReviewQueue = async (jobId: string): Promise<any> => {
  const response = await api.get(`/review/${jobId}/queue`);
  return response.data;
};

export const selectActiveLearning = async (jobId: string, nSamples = 25): Promise<any> => {
  const response = await api.post(`/review/${jobId}/active-learning/select`, {
    strategy: 'uncertainty',
    n_samples: nSamples,
    min_uncertainty: 0.3
  });
  return response.data;
};

export const validateExport = async (jobId: string): Promise<any> => {
  const response = await api.post(`/exports/${jobId}/validate`);
  return response.data;
};

export const getExportValidations = async (): Promise<any[]> => {
  const response = await api.get('/exports/validations');
  return response.data;
};

export const getAuditEvents = async (): Promise<any[]> => {
  const response = await api.get('/audit/events');
  return response.data;
};

export const getWorkerQueue = async (): Promise<any[]> => {
  const response = await api.get('/workers/queue');
  return response.data;
};

export const getModelRuns = async (): Promise<any[]> => {
  const response = await api.get('/model-gateway/runs');
  return response.data;
};

export const getModelCatalog = async (): Promise<ModelCatalogProfile[]> => {
  const response = await api.get('/model-gateway/catalog');
  return response.data.models;
};

export const getModelIntegrationStatus = async (): Promise<Record<string, any>> => {
  const response = await api.get('/model-gateway/integrations/status');
  return response.data;
};

export const prepareModelIntegration = async (modelKey: string, allowDownload = false): Promise<any> => {
  const response = await api.post(`/model-gateway/integrations/${modelKey}/prepare`, {
    allow_download: allowDownload
  });
  return response.data;
};

export const recommendModels = async (
  taskType: TaskType,
  classNames: string[],
  device = 'auto',
  limit = 5
): Promise<ModelCatalogProfile[]> => {
  const response = await api.post('/model-gateway/recommend', {
    task_type: taskType,
    class_names: classNames,
    device,
    limit
  });
  return response.data.recommendations;
};

export const enqueueQualityEvaluation = async (jobId: string): Promise<any> => {
  const response = await api.post(`/jobs/${jobId}/quality/enqueue`);
  return response.data;
};

export const evaluateJobQuality = async (jobId: string): Promise<any> => {
  const response = await api.post(`/jobs/${jobId}/quality/evaluate`);
  return response.data;
};

export const getQualityReports = async (jobId: string): Promise<any[]> => {
  const response = await api.get(`/jobs/${jobId}/quality/reports`);
  return response.data;
};

export const getQualityScores = async (jobId: string): Promise<any[]> => {
  const response = await api.get(`/jobs/${jobId}/quality/scores`);
  return response.data;
};

export const getDatasetHealth = async (datasetId: string): Promise<any> => {
  const response = await api.get(`/datasets/${datasetId}/health`);
  return response.data;
};

export const runNextWorkerJob = async (workerId = 'ui-worker'): Promise<any> => {
  const response = await api.post(`/workers/run-next?worker_id=${encodeURIComponent(workerId)}`);
  return response.data;
};

export const getObservabilityMetrics = async (): Promise<any[]> => {
  const response = await api.get('/observability/metrics');
  return response.data;
};

export const createPreferenceItem = async (payload: any): Promise<any> => {
  const response = await api.post('/preferences/items', payload);
  return response.data;
};

export const getPreferenceItems = async (): Promise<any[]> => {
  const response = await api.get('/preferences/items');
  return response.data;
};

export const recordPreferenceVote = async (itemId: string, selectedCandidateId: string, rationale?: string): Promise<any> => {
  const response = await api.post(`/preferences/items/${itemId}/votes`, {
    selected_candidate_id: selectedCandidateId,
    rationale
  });
  return response.data;
};

export default api;
