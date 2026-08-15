/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useCallback, useEffect } from 'react';
import { getHealth, getDataset, getJob, getJobProgress, getModelsStatus } from '@/services/api';
import { SystemStatus, DatasetInfo, LabelingJob, LabelingProgress, ModelsStatus } from '@/types';

export const useSystemStatus = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getHealth();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return { status, loading, error, refetch: fetchStatus };
};

export const useDataset = (datasetId: string | null) => {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDataset = useCallback(async () => {
    if (!datasetId) return;
    try {
      setLoading(true);
      const data = await getDataset(datasetId);
      setDataset(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch dataset');
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    fetchDataset();
  }, [fetchDataset]);

  return { dataset, loading, error, refetch: fetchDataset };
};

export const useLabelingJob = (jobId: string | null) => {
  const [job, setJob] = useState<LabelingJob | null>(null);
  const [progress, setProgress] = useState<LabelingProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;
    try {
      setLoading(true);
      const data = await getJob(jobId);
      setJob(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch job');
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const fetchProgress = useCallback(async () => {
    if (!jobId) return;
    try {
      const data = await getJobProgress(jobId);
      setProgress(data);
    } catch (err) {
      console.error('Failed to fetch progress:', err);
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;

    const poll = setInterval(() => {
      fetchJob();
      fetchProgress();
    }, 2000);

    return () => clearInterval(poll);
  }, [jobId, fetchJob, fetchProgress]);

  return { job, progress, loading, error, refetch: fetchJob };
};

export const useModelsStatus = () => {
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await getModelsStatus();
      setModelsStatus(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch models status:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch models status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return { modelsStatus, loading, error, refetch: fetchStatus };
};
