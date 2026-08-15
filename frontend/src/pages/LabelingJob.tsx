/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ClassHierarchy, TaskType, LabelingStrategy, DatasetInfo, ModelCatalogProfile } from '@/types';
import { getDatasets, createLabelingJob, recommendModels } from '@/services/api';
import LoadingSpinner from '@/components/LoadingSpinner';

interface ClassDefinition {
  id: number;
  name: string;
  color: string;
  aliases: string[];
}

// ── Class Template System ─────────────────────────────────────
// These are OPTIONAL presets the user can load. The form always starts EMPTY.
interface ClassTemplate {
  label: string;
  description: string;
  taskTypes: TaskType[];
  classes: Omit<ClassDefinition, 'id'>[];
}

const CLASS_TEMPLATES: ClassTemplate[] = [
  {
    label: 'General Object Detection',
    description: 'Common objects: vehicles, people, furniture, etc.',
    taskTypes: ['object_detection', 'instance_segmentation'],
    classes: [
      { name: 'Car', color: '#ef4444', aliases: ['vehicle', 'automobile'] },
      { name: 'Person', color: '#8b5cf6', aliases: ['pedestrian', 'human', 'man', 'woman'] },
      { name: 'Chair', color: '#22c55e', aliases: ['seat'] },
      { name: 'Table', color: '#6366f1', aliases: ['desk'] },
      { name: 'Door', color: '#f59e0b', aliases: ['entrance', 'gate'] },
      { name: 'Tree', color: '#10b981', aliases: ['plant'] },
    ],
  },
  {
    label: 'Autonomous Driving',
    description: 'Vehicles, pedestrians, traffic elements, road infrastructure',
    taskTypes: ['object_detection', 'instance_segmentation'],
    classes: [
      { name: 'Car', color: '#ef4444', aliases: ['vehicle', 'automobile', 'sedan'] },
      { name: 'Bus', color: '#f97316', aliases: ['transit'] },
      { name: 'Truck', color: '#a855f7', aliases: ['lorry'] },
      { name: 'Motorcycle', color: '#ec4899', aliases: ['motorbike', 'bike'] },
      { name: 'Bicycle', color: '#14b8a6', aliases: ['cycle'] },
      { name: 'Person', color: '#8b5cf6', aliases: ['pedestrian', 'human'] },
      { name: 'Traffic Light', color: '#eab308', aliases: ['signal light'] },
      { name: 'Traffic Sign', color: '#06b6d4', aliases: ['road sign', 'stop sign'] },
      { name: 'Bollard', color: '#3b82f6', aliases: ['post'] },
      { name: 'Pole', color: '#64748b', aliases: ['lamppost', 'utility pole'] },
    ],
  },
  {
    label: 'Urban Scene Segmentation',
    description: 'Roads, sidewalks, buildings, vegetation for scene understanding',
    taskTypes: ['semantic_segmentation'],
    classes: [
      { name: 'Drivable Area', color: '#3b82f6', aliases: ['road', 'driveway', 'pavement'] },
      { name: 'Sidewalk', color: '#22c55e', aliases: ['walkway', 'footpath', 'pedestrian path'] },
      { name: 'Building', color: '#a855f7', aliases: ['structure'] },
      { name: 'Vegetation', color: '#10b981', aliases: ['grass', 'tree', 'plant', 'bush'] },
      { name: 'Sky', color: '#0ea5e9', aliases: [] },
    ],
  },
  {
    label: 'Indoor Scene',
    description: 'Furniture, appliances, and indoor objects',
    taskTypes: ['object_detection', 'instance_segmentation'],
    classes: [
      { name: 'Chair', color: '#22c55e', aliases: ['seat', 'stool'] },
      { name: 'Table', color: '#6366f1', aliases: ['desk', 'counter'] },
      { name: 'Sofa', color: '#f59e0b', aliases: ['couch'] },
      { name: 'Bed', color: '#ec4899', aliases: [] },
      { name: 'Monitor', color: '#3b82f6', aliases: ['screen', 'display', 'tv'] },
      { name: 'Lamp', color: '#eab308', aliases: ['light'] },
      { name: 'Door', color: '#a855f7', aliases: ['entrance'] },
      { name: 'Window', color: '#0ea5e9', aliases: [] },
    ],
  },
  {
    label: 'Medical Imaging',
    description: 'Common medical structures for radiology / pathology',
    taskTypes: ['semantic_segmentation', 'instance_segmentation'],
    classes: [
      { name: 'Lesion', color: '#ef4444', aliases: ['tumor', 'mass', 'nodule'] },
      { name: 'Organ', color: '#22c55e', aliases: ['tissue'] },
      { name: 'Bone', color: '#f59e0b', aliases: ['skeletal'] },
      { name: 'Background', color: '#64748b', aliases: [] },
    ],
  },
  {
    label: 'Architecture & Urban Scene',
    description: 'Specialized for stairs, doors, pillars, and urban elements',
    taskTypes: ['object_detection', 'instance_segmentation', 'semantic_segmentation'],
    classes: [
      { name: 'Stairs', color: '#ef4444', aliases: ['staircase', 'flight of stairs', 'staris'] },
      { name: 'Door', color: '#f59e0b', aliases: ['entrance', 'exit', 'dor'] },
      { name: 'Chairs', color: '#22c55e', aliases: ['chair', 'seat', 'armchair'] },
      { name: 'Tables', color: '#6366f1', aliases: ['table', 'desk', 'dining table'] },
      { name: 'Pillars', color: '#8b5cf6', aliases: ['pillar', 'column', 'structural post'] },
      { name: 'Cars', color: '#ec4899', aliases: ['car', 'vehicle', 'automobile'] },
      { name: 'People', color: '#14b8a6', aliases: ['person', 'human', 'pedestrian'] },
      { name: 'Pole', color: '#3b82f6', aliases: ['streetlight', 'utility pole', 'lamp post'] },
      { name: 'Trees', color: '#10b981', aliases: ['tree', 'vegetation', 'forest'] },
      { name: 'Bollards', color: '#f97316', aliases: ['bollard', 'security post'] },
    ],
  },
];

const CLASS_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#10b981',
  '#0ea5e9', '#a855f7', '#eab308', '#06b6d4', '#64748b',
  '#dc2626', '#059669', '#d946ef', '#84cc16', '#fb923c',
];

export function LabelingJob() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [taskType, setTaskType] = useState<TaskType>('instance_segmentation');
  const [classes, setClasses] = useState<ClassDefinition[]>([]); // Starts EMPTY
  const [newClassName, setNewClassName] = useState('');
  const [newClassAliases, setNewClassAliases] = useState('');
  const [strategy, setStrategy] = useState<LabelingStrategy>('ai_assisted');

  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [device, setDevice] = useState<string>('auto');
  const [modelRecommendations, setModelRecommendations] = useState<ModelCatalogProfile[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingClassId, setEditingClassId] = useState<number | null>(null);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setDatasetLoading(true);
        const data = await getDatasets();
        setDatasets(data);
        if (data.length > 0) setSelectedDatasetId(data[0].id);
      } catch (err) {
        console.error('Failed to fetch datasets:', err);
        setError('Failed to load datasets. Please check if the backend is running.');
      } finally {
        setDatasetLoading(false);
      }
    };
    fetchDatasets();
  }, []);

  useEffect(() => {
    const classNames = classes.map((c) => c.name);
    const timer = window.setTimeout(() => {
      recommendModels(taskType, classNames, device, 5)
        .then((recommendations) => {
          setModelRecommendations(recommendations);
          if (recommendations.length && (!selectedModelId || !recommendations.some((item) => item.id === selectedModelId))) {
            setSelectedModelId(recommendations[0].id);
          }
        })
        .catch(() => setModelRecommendations([]));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [taskType, device, classes, selectedModelId]);

  // Get templates that are relevant to the current task type
  const availableTemplates = CLASS_TEMPLATES.filter(t => t.taskTypes.includes(taskType));

  const loadTemplate = (template: ClassTemplate) => {
    const templatedClasses: ClassDefinition[] = template.classes.map((c, i) => ({
      ...c,
      id: i + 1,
    }));
    setClasses(templatedClasses);
  };

  const addClass = () => {
    if (!newClassName.trim()) return;
    const newId = Math.max(...classes.map(c => c.id), 0) + 1;
    const aliases = newClassAliases.split(',').map(a => a.trim().toLowerCase()).filter(a => a);
    setClasses([...classes, {
      id: newId,
      name: newClassName.trim(),
      color: CLASS_COLORS[newId % CLASS_COLORS.length],
      aliases
    }]);
    setNewClassName('');
    setNewClassAliases('');
  };

  const removeClass = (id: number) => {
    setClasses(classes.filter(c => c.id !== id));
  };

  const updateClassAliases = (id: number, aliasesStr: string) => {
    setClasses(classes.map(c => {
      if (c.id === id) {
        return { ...c, aliases: aliasesStr.split(',').map(a => a.trim().toLowerCase()).filter(a => a) };
      }
      return c;
    }));
  };

  const handleCreateJob = async () => {
    if (!selectedDatasetId) {
      setError('Please select a dataset');
      return;
    }
    if (classes.length === 0) {
      setError('Please define at least one class');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const hierarchy: ClassHierarchy = {
        classes: classes.map(c => ({
          id: c.id,
          name: c.name,
          color: c.color,
          attributes: c.aliases.length > 0 ? { aliases: c.aliases } : undefined
        }))
      };

      await createLabelingJob(
        selectedDatasetId,
        taskType,
        hierarchy,
        strategy,
        confidenceThreshold,
        device,
        classes.map(c => c.name),
        selectedModelId ? { selected_model_id: selectedModelId } : undefined
      );

      navigate(`/`);
    } catch (err) {
      console.error('Job creation failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to create job');
    } finally {
      setLoading(false);
    }
  };

  if (datasetLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <LoadingSpinner label="Loading available datasets..." />
      </div>
    );
  }

  const selectedRecommendation = modelRecommendations.find((item) => item.id === selectedModelId);

  return (
    <div className="p-6 space-y-6 relative">
      {loading && (
        <div className="absolute inset-0 bg-white/50 z-10 flex items-center justify-center backdrop-blur-sm rounded-lg">
          <LoadingSpinner label="Initializing your labeling job..." />
        </div>
      )}

      <div>
        <h1 className="text-3xl font-bold text-gray-900">Create Labeling Job</h1>
        <p className="text-gray-600">Configure your classes and start automatic labeling</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Dataset Selection */}
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">1. Select Dataset</h3>
        {datasets.length === 0 ? (
          <div className="text-center py-6">
            <p className="text-gray-500 mb-3">No datasets uploaded yet.</p>
            <button
              onClick={() => navigate('/upload')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Upload a Dataset
            </button>
          </div>
        ) : (
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {datasets.map(d => (
              <option key={d.id} value={d.id}>{d.name} ({d.valid_images} images)</option>
            ))}
          </select>
        )}
      </div>

      {/* Class Definitions */}
      <div className="bg-white rounded-lg shadow p-6 border-2 border-blue-200">
        <h3 className="text-lg font-semibold mb-2">2. Define Your Classes</h3>
        <p className="text-sm text-gray-500 mb-4">
          <strong>Important:</strong> Only objects matching these classes will be labeled.
          Add aliases to catch different names for the same class (e.g., "car, truck, bus" for "Vehicle").
          <br className="mb-2" />
          <span className="inline-block mt-1 text-blue-700 bg-blue-50 px-2 py-1 rounded text-xs">
            💡 <strong>Tip:</strong> Use descriptive English names as the AI reads these labels.
          </span>
        </p>

        {/* Template Loading */}
        {availableTemplates.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-blue-800">Quick Start — Load a Template (optional)</span>
              {classes.length > 0 && (
                <button
                  onClick={() => setClasses([])}
                  className="text-xs text-red-600 hover:text-red-800"
                >
                  Clear All
                </button>
              )}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {availableTemplates.map((template) => (
                <button
                  key={template.label}
                  onClick={() => loadTemplate(template)}
                  className="text-left p-3 bg-white border border-blue-200 rounded-lg hover:bg-blue-100 hover:border-blue-400 transition-colors"
                >
                  <div className="font-medium text-sm text-blue-900">{template.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{template.description}</div>
                  <div className="text-[10px] text-gray-400 mt-1">{template.classes.length} classes</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Add new class */}
        <div className="bg-gray-50 p-4 rounded-lg mb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Class Name</label>
              <input
                type="text"
                placeholder="e.g., Vehicle"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addClass()}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Aliases (comma-separated)</label>
              <input
                type="text"
                placeholder="e.g., car, truck, bus"
                value={newClassAliases}
                onChange={(e) => setNewClassAliases(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addClass()}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <button
            onClick={addClass}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            + Add Class
          </button>
        </div>

        {/* Class list */}
        <div className="space-y-3">
          {classes.length === 0 ? (
            <div className="text-center py-8 border-2 border-dashed border-gray-200 rounded-lg">
              <p className="text-gray-400">No classes defined yet.</p>
              <p className="text-sm text-gray-400 mt-1">Add classes manually above, or load a template to get started.</p>
            </div>
          ) : (
            classes.map((cls) => (
              <div
                key={cls.id}
                className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                <div
                  className="w-4 h-4 rounded-full mt-1 flex-shrink-0"
                  style={{ backgroundColor: cls.color }}
                ></div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-gray-900">{cls.name}</div>
                  {editingClassId === cls.id ? (
                    <input
                      type="text"
                      defaultValue={cls.aliases.join(', ')}
                      onBlur={(e) => {
                        updateClassAliases(cls.id, e.target.value);
                        setEditingClassId(null);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          updateClassAliases(cls.id, (e.target as HTMLInputElement).value);
                          setEditingClassId(null);
                        }
                      }}
                      className="w-full mt-1 px-2 py-1 text-sm border border-blue-300 rounded focus:ring-2 focus:ring-blue-500"
                      autoFocus
                    />
                  ) : (
                    <div
                      className="text-sm text-gray-500 cursor-pointer hover:text-blue-600"
                      onClick={() => setEditingClassId(cls.id)}
                    >
                      {cls.aliases.length > 0
                        ? `Aliases: ${cls.aliases.join(', ')}`
                        : 'Click to add aliases...'}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => removeClass(cls.id)}
                  className="text-red-500 hover:text-red-700 text-lg font-bold"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Task Configuration */}
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">3. Task Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Task Type</label>
            <select
              value={taskType}
              onChange={(e) => {
                setTaskType(e.target.value as TaskType);
                // Don't auto-replace classes — let user decide via templates
              }}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="object_detection">Object Detection (Bounding Boxes)</option>
              <option value="instance_segmentation">Instance Segmentation (Masks + Boxes)</option>
              <option value="semantic_segmentation">Semantic Segmentation (Pixel-level)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Labeling Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as LabelingStrategy)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="ai_assisted">AI-Assisted (Recommended)</option>
              <option value="rule_based">Rule-Based</option>
              <option value="hybrid">Hybrid (AI + Rules)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Device (GPU/CPU)</label>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="auto">Auto-Detect (Best Available)</option>
              <option value="gpu">GPU (NVIDIA CUDA)</option>
              <option value="cpu">CPU (Standard)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold: {confidenceThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Minimum confidence for automatic label acceptance
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">4. Model Profile</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Recommended Model</label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {modelRecommendations.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} {typeof model.score === 'number' ? `(${model.score})` : ''}
                </option>
              ))}
              {modelRecommendations.length === 0 && <option value="">Loading recommendations...</option>}
            </select>
          </div>

          {selectedRecommendation && (
            <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                <div>
                  <div className="font-semibold text-gray-900">{selectedRecommendation.name}</div>
                  <div className="text-xs text-gray-500">
                    {selectedRecommendation.provider} | {selectedRecommendation.runtime_status} | {selectedRecommendation.availability}
                  </div>
                </div>
                {typeof selectedRecommendation.score === 'number' && (
                  <div className="text-sm font-semibold text-blue-700">Fit score {selectedRecommendation.score}</div>
                )}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
                <div>
                  <div className="text-xs font-semibold uppercase text-gray-500 mb-1">Why recommended</div>
                  <ul className="text-sm text-gray-700 space-y-1 list-disc pl-5">
                    {(selectedRecommendation.why || []).map((reason) => <li key={reason}>{reason}</li>)}
                  </ul>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase text-gray-500 mb-1">Runtime note</div>
                  <p className="text-sm text-gray-700">
                    {selectedRecommendation.constraints?.[0] || selectedRecommendation.research_basis}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleCreateJob}
          disabled={loading || !selectedDatasetId || classes.length === 0}
          className={`px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 flex items-center gap-2 ${loading || classes.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
          {loading ? 'Starting...' : 'Create Labeling Job'}
        </button>
      </div>
    </div>
  );
}
