/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadDataset } from '@/services/api';
import { DatasetInfo } from '@/types';

interface DatasetUploadProps {
  onUploadComplete?: (dataset: DatasetInfo) => void;
}

export function DatasetUpload({ onUploadComplete }: DatasetUploadProps) {
  const [datasetName, setDatasetName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploadedDataset, setUploadedDataset] = useState<DatasetInfo | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    if (!file.name.endsWith('.zip')) {
      setError('Please upload a ZIP file');
      return;
    }

    try {
      setUploading(true);
      setProgress(0);
      setError(null);

      const dataset = await uploadDataset(
        file,
        datasetName || undefined,
        (p) => setProgress(p)
      );

      setUploadedDataset(dataset);
      onUploadComplete?.(dataset);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [datasetName, onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/zip': ['.zip'],
    },
    multiple: false,
    disabled: uploading,
  });

  const reset = () => {
    setUploadedDataset(null);
    setProgress(0);
    setError(null);
    setDatasetName('');
  };

  if (uploadedDataset) {
    return (
      <div className="p-6">
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h2 className="text-green-800 text-xl font-semibold mb-4">Upload Complete!</h2>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <p className="text-sm text-gray-600">Dataset Name</p>
              <p className="font-medium">{uploadedDataset.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Dataset ID</p>
              <p className="font-medium font-mono text-sm">{uploadedDataset.id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Valid Images</p>
              <p className="font-medium text-green-600">{uploadedDataset.valid_images}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Corrupted</p>
              <p className="font-medium text-red-600">{uploadedDataset.corrupted_images}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Size</p>
              <p className="font-medium">{uploadedDataset.total_size_mb.toFixed(2)} MB</p>
            </div>
          </div>
          <div className="flex gap-4">
            <button
              onClick={reset}
              className="px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Upload Another
            </button>
            <button
              onClick={() => window.location.href = '/label'}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
            >
              Start Labeling Job
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Upload Dataset</h1>
        <p className="text-gray-600">Upload your images for automatic labeling</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Dataset Information</h3>
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Dataset Name (optional)</label>
          <input
            type="text"
            placeholder="My Dataset"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            disabled={uploading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Upload ZIP File</h3>
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
            transition-colors
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
            ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          {isDragActive ? (
            <p className="text-lg font-medium text-blue-600">Drop the ZIP file here</p>
          ) : (
            <>
              <p className="text-lg font-medium text-gray-700">Drag & drop a ZIP file here</p>
              <p className="text-sm text-gray-500 mt-2">or click to select a file</p>
            </>
          )}
        </div>

        {uploading && (
          <div className="mt-6 space-y-2">
            <div className="flex justify-between text-sm">
              <span>Uploading...</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-blue-800 text-sm">
          <strong>Tip:</strong> Large datasets (50K+ images) may take several minutes to process.
          Supported formats: JPG, PNG, BMP, TIFF, WebP
        </p>
      </div>
    </div>
  );
}
