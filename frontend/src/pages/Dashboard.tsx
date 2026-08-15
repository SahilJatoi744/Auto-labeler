/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSystemStatus, useModelsStatus } from '@/hooks/useApi';
import { triggerModelDownload, getLabelingJobs, startLabelingJob, stopLabelingJob, deleteLabelingJob, setSystemDevice } from '@/services/api';
import LoadingSpinner from '@/components/LoadingSpinner';

export function Dashboard() {
  const { status, loading, error, refetch: refetchStatus } = useSystemStatus();
  const { modelsStatus } = useModelsStatus();
  const [showDeviceModal, setShowDeviceModal] = useState(false);
  const [isUpdatingDevice, setIsUpdatingDevice] = useState(false);

  useEffect(() => {
    // Show modal if device preference not set in local storage
    const deviceSet = localStorage.getItem('device_preference_set');
    if (!deviceSet) {
      setShowDeviceModal(true);
    }
  }, []);

  const handleDeviceSelection = async (device: string) => {
    try {
      setIsUpdatingDevice(true);
      await setSystemDevice(device);
      localStorage.setItem('device_preference_set', 'true');
      setShowDeviceModal(false);
      await refetchStatus();
    } catch (err) {
      console.error('Failed to set device:', err);
      alert('Failed to update system device preference');
    } finally {
      setIsUpdatingDevice(false);
    }
  };

  const handleDownload = async (modelName: string) => {
    try {
      await triggerModelDownload(modelName);
    } catch (err) {
      console.error('Download trigger failed:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner label="Connecting to system..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h2 className="text-red-700 font-semibold mb-2">Connection Error</h2>
          <p className="text-red-600">{error}</p>
          <p className="text-gray-600 mt-2 text-sm">
            Make sure the backend server is running on http://localhost:8000
          </p>
        </div>
      </div>
    );
  }

  if (!status) return null;

  return (
    <div className="p-6 space-y-6 relative">
      {/* Device Selection Modal */}
      {showDeviceModal && (
        <div className="fixed inset-0 bg-black/60 z-[100] flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 border border-gray-100 animate-in fade-in zoom-in duration-300">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to AutoLabeler</h2>
            <p className="text-gray-600 mb-6">Select your preferred computation device to get started. You can change this later.</p>

            <div className="grid grid-cols-1 gap-3">
              <button
                onClick={() => handleDeviceSelection('auto')}
                disabled={isUpdatingDevice}
                className="flex items-center justify-between p-4 border-2 border-blue-100 rounded-xl hover:border-blue-500 hover:bg-blue-50 transition-all group"
              >
                <div className="text-left">
                  <div className="font-bold text-gray-900 group-hover:text-blue-700">Auto-Detect</div>
                  <div className="text-xs text-gray-500">Uses GPU if available, else CPU</div>
                </div>
                <div className="w-6 h-6 rounded-full border-2 border-blue-200 flex items-center justify-center group-hover:border-blue-500">
                  <div className="w-2.5 h-2.5 bg-blue-500 rounded-full opacity-0 group-hover:opacity-100"></div>
                </div>
              </button>

              <button
                onClick={() => handleDeviceSelection('gpu')}
                disabled={isUpdatingDevice || !status.gpu_available}
                className={`flex items-center justify-between p-4 border-2 rounded-xl transition-all group ${!status.gpu_available ? 'opacity-50 cursor-not-allowed bg-gray-50 border-gray-200' : 'border-green-100 hover:border-green-500 hover:bg-green-50'}`}
              >
                <div className="text-left">
                  <div className={`font-bold ${!status.gpu_available ? 'text-gray-400' : 'text-gray-900 group-hover:text-green-700'}`}>NVIDIA GPU (CUDA)</div>
                  <div className="text-xs text-gray-500">{status.gpu_available ? 'Maximum performance' : 'No compatible GPU detected'}</div>
                </div>
                {status.gpu_available && (
                  <div className="w-6 h-6 rounded-full border-2 border-green-200 flex items-center justify-center group-hover:border-green-500">
                    <div className="w-2.5 h-2.5 bg-green-500 rounded-full opacity-0 group-hover:opacity-100"></div>
                  </div>
                )}
              </button>

              <button
                onClick={() => handleDeviceSelection('cpu')}
                disabled={isUpdatingDevice}
                className="flex items-center justify-between p-4 border-2 border-gray-100 rounded-xl hover:border-gray-500 hover:bg-gray-50 transition-all group"
              >
                <div className="text-left">
                  <div className="font-bold text-gray-900 group-hover:text-gray-700">CPU Only</div>
                  <div className="text-xs text-gray-500">Stable but slower for large models</div>
                </div>
                <div className="w-6 h-6 rounded-full border-2 border-gray-200 flex items-center justify-center group-hover:border-gray-500">
                  <div className="w-2.5 h-2.5 bg-gray-500 rounded-full opacity-0 group-hover:opacity-100"></div>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">System overview and resource usage</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">System Status</h3>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${status.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-2xl font-bold">{status.status === 'healthy' ? 'Healthy' : 'Error'}</span>
          </div>
          <p className="text-xs text-gray-500 mt-1">Version {status.version}</p>
        </div>

        <div className="bg-white rounded-lg shadow p-4 border border-gray-200 hover:border-blue-400 transition-colors group cursor-pointer relative overflow-hidden" onClick={() => setShowDeviceModal(true)}>
          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <svg className="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </div>
          <h3 className="text-sm font-medium text-gray-500 mb-2">GPU Status (Mode: {status.device_preference})</h3>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${status.gpu_available && status.device_preference !== 'cpu' ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span className="text-2xl font-bold">
              {status.device_preference === 'cpu' ? 'CPU Only' : status.gpu_available ? 'Available' : 'CPU Only'}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {status.device_preference === 'cpu' ? 'Forced CPU Mode' : (status.gpu_info?.name || 'No GPU detected')}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">CPU Usage</h3>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">{status.cpu_usage.toFixed(0)}%</span>
            <span className="text-xs text-gray-400">({status.cpu_count} Cores)</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-1.5 mt-2">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${status.cpu_usage > 80 ? 'bg-red-500' : status.cpu_usage > 50 ? 'bg-yellow-500' : 'bg-blue-600'}`}
              style={{ width: `${status.cpu_usage}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Memory Usage</h3>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">{status.memory_usage_percent.toFixed(0)}%</span>
            <span className="text-xs text-gray-400">({status.memory_gb.toFixed(1)} GB Total)</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-1.5 mt-2">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${status.memory_usage_percent > 85 ? 'bg-red-500' : status.memory_usage_percent > 65 ? 'bg-yellow-500' : 'bg-green-600'}`}
              style={{ width: `${status.memory_usage_percent}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Active Jobs</h3>
          <div className="text-2xl font-bold">{status.active_jobs}</div>
          <p className="text-xs text-gray-500 mt-1">Currently running</p>
        </div>
      </div>

      {status.gpu_available && status.gpu_info && (
        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">GPU Information</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Memory Usage</span>
              <span>{status.gpu_info.memory_allocated_gb.toFixed(2)} / {status.gpu_info.memory_total_gb.toFixed(2)} GB</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${(status.gpu_info.memory_allocated_gb / status.gpu_info.memory_total_gb) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">AI Model Readiness</h3>
        <div className="space-y-6">
          {modelsStatus && Object.entries(modelsStatus).map(([name, status]) => (
            <div key={name} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium uppercase">{name}</span>
                <span className={`text-xs font-semibold px-2 py-1 rounded ${status.status === 'ready' ? 'bg-green-100 text-green-700' :
                  status.status === 'downloading' ? 'bg-blue-100 text-blue-700' :
                    status.status === 'error' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                  }`}>
                  {status.status.replace('_', ' ')}
                </span>
              </div>

              {status.status === 'downloading' && (
                <div className="w-full bg-gray-100 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${status.progress}%` }}
                  ></div>
                </div>
              )}

              {status.status === 'not_downloaded' && (
                <button
                  onClick={() => handleDownload(name)}
                  className="w-full py-1.5 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
                >
                  Download {name.toUpperCase()}
                </button>
              )}

              {status.error && (
                <p className="text-xs text-red-500 mt-1">{status.error}</p>
              )}
            </div>
          ))}
          {!modelsStatus && <p className="text-gray-500">Loading model information...</p>}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 bg-gray-50/50 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900">Recent Labeling Jobs</h3>
          <a href="/label" className="text-sm font-medium text-blue-600 hover:text-blue-700">New Job +</a>
        </div>
        <JobsList />
      </div>

      <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Available Disk Space</h3>
        <div className="text-2xl font-bold">{status.disk_space_gb.toFixed(1)} GB</div>
        <p className="text-xs text-gray-500 mt-1">Free space for datasets and outputs</p>
      </div>
    </div>
  );
}

// Reusable JobsTable component
interface JobsTableProps {
  jobs: any[];
  handleStart: (jobId: string) => void;
  handleStop: (jobId: string) => void;
  handleDelete: (jobId: string) => void;
  navigate: (path: string) => void;
}

function JobsTable({ jobs, handleStart, handleStop, handleDelete, navigate }: JobsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="bg-gray-50 text-gray-600 uppercase text-[10px] font-bold tracking-wider">
          <tr>
            <th className="px-6 py-3">Job ID</th>
            <th className="px-6 py-3">Task</th>
            <th className="px-6 py-3">Progress</th>
            <th className="px-6 py-3">Status</th>
            <th className="px-6 py-3">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {jobs.map(job => (
            <tr key={job.id} className="hover:bg-gray-50/80 transition-colors">
              <td className="px-6 py-4 font-mono text-[11px] text-gray-500">{job.id}</td>
              <td className="px-6 py-4">
                <div className="font-medium text-gray-900">{job.task_type.replace('_', ' ')}</div>
                <div className="text-[10px] text-gray-400 capitalize">{job.strategy} strategy</div>
              </td>
              <td className="px-6 py-4 w-48">
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-blue-600 h-full transition-all duration-500"
                      style={{ width: `${(job.processed_images / job.total_images) * 100}%` }}
                    />
                  </div>
                  <span className="text-[11px] font-medium text-gray-600">
                    {job.processed_images}/{job.total_images}
                  </span>
                </div>
              </td>
              <td className="px-6 py-4">
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${job.status === 'completed' ? 'bg-green-100 text-green-700' :
                  job.status === 'running' ? 'bg-blue-100 text-blue-700 animate-pulse' :
                    job.status === 'failed' ? 'bg-red-100 text-red-700' :
                      job.status === 'stopped' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-gray-100 text-gray-600'
                  }`}>
                  {job.status}
                </span>
              </td>
              <td className="px-6 py-4">
                <div className="flex gap-2">
                  {job.status === 'created' && (
                    <button
                      onClick={() => handleStart(job.id)}
                      className="text-blue-600 hover:text-blue-800 font-semibold"
                    >
                      Start
                    </button>
                  )}
                  {job.status === 'running' && (
                    <button
                      onClick={() => handleStop(job.id)}
                      className="text-orange-600 hover:text-orange-800 font-semibold"
                    >
                      Cancel
                    </button>
                  )}
                  {job.status === 'completed' && (
                    <button
                      onClick={() => navigate(`/gallery/${job.id}`)}
                      className="text-indigo-600 hover:text-indigo-800 font-semibold"
                    >
                      View
                    </button>
                  )}
                  {(job.status === 'completed' || job.status === 'running') && (
                    <button
                      onClick={() => navigate(`/export`)}
                      className="text-gray-600 hover:text-gray-800 font-semibold"
                    >
                      Export
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(job.id)}
                    className="text-red-500 hover:text-red-700 font-semibold"
                    title="Delete job"
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JobsList() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const fetchJobs = async () => {
    try {
      const data = await getLabelingJobs();
      // Sort by created_at desc
      setJobs(data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async (jobId: string) => {
    try {
      await startLabelingJob(jobId);
      fetchJobs();
    } catch (err) {
      alert('Failed to start job: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleStop = async (jobId: string) => {
    try {
      await stopLabelingJob(jobId);
      fetchJobs();
    } catch (err) {
      alert('Failed to stop job: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleDelete = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
      return;
    }
    try {
      await deleteLabelingJob(jobId);
      fetchJobs();
    } catch (err) {
      alert('Failed to delete job: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  // Find any running job
  const runningJob = jobs.find(job => job.status === 'running');

  if (loading && jobs.length === 0) return <div className="p-8 text-center text-gray-500">Loading jobs...</div>;
  if (jobs.length === 0) return <div className="p-12 text-center text-gray-400 italic">No labeling jobs created yet.</div>;

  // Show prominent progress modal when a job is running
  if (runningJob) {
    const progress = runningJob.total_images > 0
      ? Math.round((runningJob.processed_images / runningJob.total_images) * 100)
      : 0;

    return (
      <div className="p-6">
        {/* Running Job Progress Modal */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-2xl p-8 shadow-lg">
          <div className="flex flex-col items-center text-center space-y-6">
            {/* Animated Spinner */}
            <div className="relative">
              <div className="w-20 h-20 border-4 border-blue-200 rounded-full"></div>
              <div className="absolute top-0 left-0 w-20 h-20 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <span className="text-xl font-bold text-blue-600">{progress}%</span>
              </div>
            </div>

            {/* Status Text */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Processing Images...</h2>
              <p className="text-gray-600">
                {runningJob.task_type.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
              </p>
            </div>

            {/* Progress Bar */}
            <div className="w-full max-w-md">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{runningJob.processed_images} / {runningJob.total_images} images</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Current Image */}
            {runningJob.progress?.current_image && (
              <p className="text-sm text-gray-500">
                Current: <span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded">{runningJob.progress.current_image}</span>
              </p>
            )}

            {/* Elapsed Time */}
            {runningJob.progress?.elapsed_seconds && (
              <p className="text-sm text-gray-500">
                Elapsed: {Math.floor(runningJob.progress.elapsed_seconds / 60)}m {runningJob.progress.elapsed_seconds % 60}s
              </p>
            )}

            {/* Cancel Button */}
            <button
              onClick={() => handleStop(runningJob.id)}
              className="px-6 py-2 bg-orange-100 text-orange-700 font-semibold rounded-lg hover:bg-orange-200 transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Cancel Job
            </button>

            <p className="text-xs text-gray-400">
              Job ID: <span className="font-mono">{runningJob.id}</span>
            </p>
          </div>
        </div>

        {/* Other Jobs Table (collapsed) */}
        {jobs.filter(j => j.status !== 'running').length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-500 mb-2">Other Jobs</h3>
            <JobsTable jobs={jobs.filter(j => j.status !== 'running')} handleStart={handleStart} handleStop={handleStop} handleDelete={handleDelete} navigate={navigate} />
          </div>
        )}
      </div>
    );
  }

  return (
    <JobsTable
      jobs={jobs}
      handleStart={handleStart}
      handleStop={handleStop}
      handleDelete={handleDelete}
      navigate={navigate}
    />
  );
}
