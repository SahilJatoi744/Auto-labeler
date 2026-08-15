/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ExportFormat, LabelingJob } from '@/types';
import { getLabelingJobs, exportLabels } from '@/services/api';

export function Export() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<LabelingJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>('');
  const [format, setFormat] = useState<ExportFormat>('coco');
  const [trainRatio, setTrainRatio] = useState(70);
  const [valRatio, setValRatio] = useState(15);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [exportId, setExportId] = useState<string | null>(null);

  useEffect(() => {
    let interval: number;

    const fetchJobs = async () => {
      try {
        const data = await getLabelingJobs();
        setJobs(data);

        // If no job selected yet, pick the first one (prefer completed or running)
        if (!selectedJobId && data.length > 0) {
          const firstSelectable = data.find(j => j.status === 'completed' || j.status === 'running') || data[0];
          setSelectedJobId(firstSelectable.id);
        }
      } catch (err) {
        console.error('Failed to fetch jobs:', err);
      }
    };

    fetchJobs();
    interval = window.setInterval(fetchJobs, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [selectedJobId]);

  const selectedJob = jobs.find(j => j.id === selectedJobId);
  const isCompleted = selectedJob?.status === 'completed';
  const isRunning = selectedJob?.status === 'running';
  const progress = selectedJob ? Math.round((selectedJob.processed_images / selectedJob.total_images) * 100) : 0;

  const handleExport = async () => {
    if (!selectedJobId) {
      setError('Please select a completed labeling job');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      const result = await exportLabels({
        job_id: selectedJobId,
        format,
        split_ratios: {
          train: trainRatio / 100,
          val: valRatio / 100,
          test: testRatio / 100
        }
      });
      setExportId(result.export_id);
      setSuccess(`Dataset successfully exported!`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setLoading(false);
    }
  };


  const testRatio = 100 - trainRatio - valRatio;

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Export Labels</h1>
        <p className="text-gray-600">Export your labeled dataset in various formats</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center justify-between">
          <p className="text-green-700 font-medium">{success}</p>
          {exportId && (
            <a
              href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/export/${exportId}/download`}
              download
              className="px-4 py-2 bg-green-600 text-white text-sm font-semibold rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download ZIP
            </a>
          )}
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Select Labeling Job</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Job Name / ID</label>
            <select
              value={selectedJobId}
              onChange={(e) => setSelectedJobId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {jobs.length === 0 && <option value="">No jobs found</option>}
              {jobs.map(j => (
                <option key={j.id} value={j.id}>
                  {j.task_type.toUpperCase()} - {j.id.slice(0, 8)} ({j.status})
                </option>
              ))}
            </select>
          </div>

          {selectedJob && (
            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-600">Status: <span className="text-blue-600 uppercase">{selectedJob.status}</span></span>
                <span className="text-sm font-medium text-gray-600">{selectedJob.processed_images} / {selectedJob.total_images} images</span>
              </div>

              {(isRunning || isCompleted) && (
                <div className="mt-4 flex gap-3">
                  <div className="flex-1">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div
                        className={`h-2.5 rounded-full transition-all duration-500 ${isCompleted ? 'bg-green-600' : 'bg-blue-600'}`}
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  </div>
                  <button
                    onClick={() => navigate(`/gallery/${selectedJobId}`)}
                    className="px-3 py-1 bg-white border border-blue-300 text-blue-600 text-xs font-semibold rounded hover:bg-blue-50 transition-colors whitespace-nowrap"
                  >
                    View Results
                  </button>
                </div>
              )}

              {isRunning && (
                <p className="text-xs text-blue-600 mt-2 animate-pulse font-medium">
                  Labeling in progress... Please wait to export.
                </p>
              )}

              {isCompleted && (
                <p className="text-xs text-green-600 mt-2 font-medium">
                  Labeling complete! Ready for export.
                </p>
              )}
            </div>
          )}

          {jobs.length === 0 && (
            <p className="text-sm text-amber-600 mt-2">
              You need to start a labeling job from the "Labeling Job" page first.
            </p>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Export Format</h3>
        <div className="grid grid-cols-3 gap-4">
          {[
            { value: 'coco', label: 'COCO JSON' },
            { value: 'pascal_voc', label: 'Pascal VOC XML' },
            { value: 'yolo', label: 'YOLO Format' },
          ].map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setFormat(value as ExportFormat)}
              className={`
                p-4 rounded-lg border-2 text-center transition-all
                ${format === value
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300'}
              `}
            >
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Dataset Split</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Training: {trainRatio}%
            </label>
            <input
              type="range"
              min="50"
              max="90"
              step="5"
              value={trainRatio}
              onChange={(e) => {
                const newTrain = parseInt(e.target.value);
                const remaining = 100 - newTrain;
                const newVal = Math.min(valRatio, remaining - 5);
                setTrainRatio(newTrain);
                setValRatio(newVal);
              }}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Validation: {valRatio}%
            </label>
            <input
              type="range"
              min="5"
              max="30"
              step="5"
              value={valRatio}
              onChange={(e) => {
                const newVal = parseInt(e.target.value);
                const maxVal = 100 - trainRatio - 5;
                if (newVal <= maxVal) setValRatio(newVal);
              }}
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-4 mt-6">
          <div className="flex-1 p-3 bg-blue-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-blue-600">{trainRatio}%</p>
            <p className="text-sm text-blue-600">Train</p>
          </div>
          <div className="flex-1 p-3 bg-green-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-600">{valRatio}%</p>
            <p className="text-sm text-green-600">Validation</p>
          </div>
          <div className="flex-1 p-3 bg-purple-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-purple-600">{testRatio}%</p>
            <p className="text-sm text-purple-600">Test</p>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleExport}
          disabled={loading || !isCompleted}
          className={`px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 flex items-center gap-2 ${loading || !isCompleted ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
          {loading ? 'Exporting...' : 'Export Dataset'}
        </button>
      </div>
    </div>
  );
}
