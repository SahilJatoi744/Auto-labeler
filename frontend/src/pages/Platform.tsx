/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useEffect, useState } from 'react';
import {
  createPreferenceItem,
  createProject,
  createWorkspace,
  enqueueQualityEvaluation,
  evaluateJobQuality,
  getAuditEvents,
  getDatasets,
  getDatasetHealth,
  getDatasetLineage,
  getDatasetVersions,
  getExportValidations,
  getLabelingJobs,
  getModelCatalog,
  getModelIntegrationStatus,
  getModelRuns,
  getObservabilityMetrics,
  getPreferenceItems,
  getProjects,
  getQualityReports,
  getQualityScores,
  getReviewQueue,
  getWorkerQueue,
  getWorkspaces,
  prepareModelIntegration,
  recordPreferenceVote,
  runNextWorkerJob,
  selectActiveLearning,
  validateExport,
} from '@/services/api';

type Tab = 'projects' | 'review' | 'rlhf' | 'intelligence' | 'ops';

export function Platform() {
  const [tab, setTab] = useState<Tab>('projects');
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [reviewQueue, setReviewQueue] = useState<any | null>(null);
  const [activeSelection, setActiveSelection] = useState<any | null>(null);
  const [versions, setVersions] = useState<any[]>([]);
  const [lineage, setLineage] = useState<any[]>([]);
  const [preferences, setPreferences] = useState<any[]>([]);
  const [audit, setAudit] = useState<any[]>([]);
  const [queue, setQueue] = useState<any[]>([]);
  const [modelRuns, setModelRuns] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [validations, setValidations] = useState<any[]>([]);
  const [modelCatalog, setModelCatalog] = useState<any[]>([]);
  const [qualityReport, setQualityReport] = useState<any | null>(null);
  const [qualityReports, setQualityReports] = useState<any[]>([]);
  const [qualityScores, setQualityScores] = useState<any[]>([]);
  const [datasetHealth, setDatasetHealth] = useState<any | null>(null);
  const [workerResult, setWorkerResult] = useState<any | null>(null);
  const [integrationStatus, setIntegrationStatus] = useState<Record<string, any>>({});
  const [prepareResult, setPrepareResult] = useState<any | null>(null);
  const [workspaceName, setWorkspaceName] = useState('Default Workspace');
  const [projectName, setProjectName] = useState('Image Annotation Project');
  const [preferencePrompt, setPreferencePrompt] = useState('Which annotation is better for training?');
  const [preferenceImageId, setPreferenceImageId] = useState('');
  const [message, setMessage] = useState('');

  const refresh = async () => {
    const [ws, prj, ds, jb, aud, q, runs, mets, vals, prefs, catalog, integrations] = await Promise.all([
      getWorkspaces(),
      getProjects(),
      getDatasets(),
      getLabelingJobs(),
      getAuditEvents(),
      getWorkerQueue(),
      getModelRuns(),
      getObservabilityMetrics(),
      getExportValidations(),
      getPreferenceItems(),
      getModelCatalog(),
      getModelIntegrationStatus(),
    ]);
    setWorkspaces(ws);
    setProjects(prj);
    setDatasets(ds);
    setJobs(jb);
    setAudit(aud);
    setQueue(q);
    setModelRuns(runs);
    setMetrics(mets);
    setValidations(vals);
    setPreferences(prefs);
    setModelCatalog(catalog);
    setIntegrationStatus(integrations);
    if (!selectedJobId && jb.length) setSelectedJobId(jb[0].id);
    if (!selectedDatasetId && ds.length) setSelectedDatasetId(ds[0].id);
  };

  useEffect(() => {
    refresh().catch((err) => setMessage(err instanceof Error ? err.message : 'Failed to load platform data'));
  }, []);

  useEffect(() => {
    if (!selectedDatasetId) return;
    Promise.all([getDatasetVersions(selectedDatasetId), getDatasetLineage(selectedDatasetId)])
      .then(([nextVersions, nextLineage]) => {
        setVersions(nextVersions);
        setLineage(nextLineage);
      })
      .catch(() => undefined);
  }, [selectedDatasetId]);

  const handleCreateProject = async () => {
    const workspace = workspaces[0] || await createWorkspace(workspaceName);
    await createProject(workspace.id, projectName);
    setMessage('Project created');
    await refresh();
  };

  const handleReviewLoad = async () => {
    if (!selectedJobId) return;
    const [queueData, selection] = await Promise.all([
      getReviewQueue(selectedJobId),
      selectActiveLearning(selectedJobId, 25),
    ]);
    setReviewQueue(queueData);
    setActiveSelection(selection);
    setMessage('Review queue and active learning selection refreshed');
  };

  const handleValidateExport = async () => {
    if (!selectedJobId) return;
    await validateExport(selectedJobId);
    setValidations(await getExportValidations());
    setMessage('Export validation recorded');
  };

  const handleEvaluateQuality = async () => {
    if (!selectedJobId) return;
    const report = await evaluateJobQuality(selectedJobId);
    const [reports, scores] = await Promise.all([
      getQualityReports(selectedJobId),
      getQualityScores(selectedJobId),
    ]);
    setQualityReport(report);
    setQualityReports(reports);
    setQualityScores(scores);
    setMessage('Quality agent evaluation recorded');
    await refresh();
  };

  const handleEnqueueQuality = async () => {
    if (!selectedJobId) return;
    await enqueueQualityEvaluation(selectedJobId);
    setQueue(await getWorkerQueue());
    setMessage('Quality evaluation queued');
  };

  const handleDatasetHealth = async () => {
    if (!selectedDatasetId) return;
    setDatasetHealth(await getDatasetHealth(selectedDatasetId));
    setMessage('Dataset health refreshed');
  };

  const handleRunWorker = async () => {
    const result = await runNextWorkerJob('platform-ui-worker');
    setWorkerResult(result);
    setQueue(await getWorkerQueue());
    setMessage(result.status === 'empty' ? 'Worker queue is empty' : `Worker finished ${result.task_type || 'task'}`);
    await refresh();
  };

  const handlePrepareIntegration = async (modelKey: string) => {
    const result = await prepareModelIntegration(modelKey, modelKey === 'sam3');
    setPrepareResult(result);
    setIntegrationStatus(await getModelIntegrationStatus());
    setMessage(result.message || `Prepared ${modelKey}`);
  };

  const handleCreatePreference = async () => {
    if (!preferenceImageId.trim()) {
      setMessage('Enter an image id for the preference item');
      return;
    }
    await createPreferenceItem({
      project_id: projects[0]?.id,
      image_id: preferenceImageId,
      prompt: preferencePrompt,
      candidates: [
        { id: 'candidate_a', label: 'Candidate A' },
        { id: 'candidate_b', label: 'Candidate B' },
      ],
    });
    setPreferences(await getPreferenceItems());
    setMessage('Preference item created');
  };

  const handleVote = async (itemId: string, selectedCandidateId: string) => {
    await recordPreferenceVote(itemId, selectedCandidateId, 'Selected from platform review UI');
    setPreferences(await getPreferenceItems());
    setMessage('Preference vote recorded');
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold text-gray-900">Platform</h1>
        <p className="text-gray-600">Projects, review queues, RLHF preferences, lineage, audit, and operations</p>
      </div>

      {message && <div className="bg-blue-50 border border-blue-200 text-blue-800 rounded-lg p-3 text-sm">{message}</div>}

      <div className="flex flex-wrap gap-2">
        {(['projects', 'review', 'rlhf', 'intelligence', 'ops'] as Tab[]).map((nextTab) => (
          <button
            key={nextTab}
            onClick={() => setTab(nextTab)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold ${tab === nextTab ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200 text-gray-700'}`}
          >
            {nextTab.toUpperCase()}
          </button>
        ))}
      </div>

      {tab === 'projects' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Create Workspace / Project</h2>
            <div className="space-y-3">
              <input value={workspaceName} onChange={(e) => setWorkspaceName(e.target.value)} className="w-full border rounded-lg px-3 py-2" />
              <input value={projectName} onChange={(e) => setProjectName(e.target.value)} className="w-full border rounded-lg px-3 py-2" />
              <button onClick={handleCreateProject} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">Create Project</button>
            </div>
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Projects</h2>
            <List items={projects} empty="No projects yet" />
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Dataset Versioning</h2>
            <Select value={selectedDatasetId} onChange={setSelectedDatasetId} items={datasets} labelKey="name" />
            <List items={versions} empty="No versions recorded" />
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Lineage</h2>
            <List items={lineage} empty="No lineage events recorded" />
          </section>
        </div>
      )}

      {tab === 'review' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Review Queue</h2>
            <Select value={selectedJobId} onChange={setSelectedJobId} items={jobs} labelKey="id" />
            <div className="flex gap-2 mt-3">
              <button onClick={handleReviewLoad} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">Load Review Queue</button>
              <button onClick={handleValidateExport} className="px-4 py-2 bg-white border rounded-lg font-semibold">Validate Export</button>
            </div>
            <List items={reviewQueue?.items || []} empty="No flagged annotations loaded" />
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Active Learning Selection</h2>
            <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto min-h-32">{JSON.stringify(activeSelection || {}, null, 2)}</pre>
          </section>
        </div>
      )}

      {tab === 'rlhf' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Create Image Preference Item</h2>
            <input value={preferenceImageId} onChange={(e) => setPreferenceImageId(e.target.value)} placeholder="Image ID" className="w-full border rounded-lg px-3 py-2 mb-3" />
            <textarea value={preferencePrompt} onChange={(e) => setPreferencePrompt(e.target.value)} className="w-full border rounded-lg px-3 py-2 mb-3" />
            <button onClick={handleCreatePreference} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">Create Preference</button>
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Preference Queue</h2>
            <div className="space-y-3">
              {preferences.map((item) => (
                <div key={item.id} className="border rounded-lg p-3 bg-gray-50">
                  <div className="font-semibold">{item.prompt}</div>
                  <div className="text-xs text-gray-500">Image: {item.image_id} | Status: {item.status}</div>
                  <div className="flex gap-2 mt-2">
                    <button onClick={() => handleVote(item.id, 'candidate_a')} className="px-3 py-1 bg-white border rounded text-sm">Candidate A</button>
                    <button onClick={() => handleVote(item.id, 'candidate_b')} className="px-3 py-1 bg-white border rounded text-sm">Candidate B</button>
                  </div>
                </div>
              ))}
              {preferences.length === 0 && <p className="text-sm text-gray-500">No preference items yet.</p>}
            </div>
          </section>
        </div>
      )}

      {tab === 'intelligence' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Model Catalog</h2>
            <List items={modelCatalog} empty="No model profiles loaded" />
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Local Integration Status</h2>
            <div className="space-y-3">
              {Object.entries(integrationStatus).map(([key, value]) => (
                <div key={key} className="border rounded-lg p-3 bg-gray-50">
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-semibold text-gray-900">{key.replace('_', ' ').toUpperCase()}</div>
                    <span className={`text-xs font-semibold px-2 py-1 rounded ${value.ready ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-800'}`}>
                      {value.ready ? 'READY' : 'NOT READY'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">{value.package || 'model'} {value.package_version || ''}</div>
                  <div className="text-xs text-gray-600 mt-2">{value.install_hint}</div>
                  <button
                    onClick={() => handlePrepareIntegration(key)}
                    className="mt-3 px-3 py-1.5 bg-white border rounded-lg text-sm font-semibold hover:bg-gray-100"
                  >
                    Prepare
                  </button>
                </div>
              ))}
              {Object.keys(integrationStatus).length === 0 && <p className="text-sm text-gray-500">No integration status loaded.</p>}
            </div>
            {prepareResult && (
              <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto mt-3">{JSON.stringify(prepareResult, null, 2)}</pre>
            )}
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Quality Agent</h2>
            <Select value={selectedJobId} onChange={setSelectedJobId} items={jobs} labelKey="id" />
            <div className="flex flex-wrap gap-2 mt-3">
              <button onClick={handleEvaluateQuality} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">Evaluate Quality</button>
              <button onClick={handleEnqueueQuality} className="px-4 py-2 bg-white border rounded-lg font-semibold">Queue Quality Job</button>
            </div>
            <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto min-h-32 mt-3">{JSON.stringify(qualityReport || {}, null, 2)}</pre>
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Dataset Health</h2>
            <Select value={selectedDatasetId} onChange={setSelectedDatasetId} items={datasets} labelKey="name" />
            <button onClick={handleDatasetHealth} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold mt-3">Check Dataset Health</button>
            <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto min-h-32 mt-3">{JSON.stringify(datasetHealth || {}, null, 2)}</pre>
          </section>
          <section className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
            <h2 className="font-bold text-lg mb-4">Worker Execution</h2>
            <button onClick={handleRunWorker} className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold">Run Next Worker Job</button>
            <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto min-h-32 mt-3">{JSON.stringify(workerResult || {}, null, 2)}</pre>
          </section>
          <Panel title="Quality Reports" items={qualityReports} />
          <Panel title="Quality Scores" items={qualityScores} />
        </div>
      )}

      {tab === 'ops' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <Panel title="Durable Worker Queue" items={queue} />
          <Panel title="Model Gateway Runs" items={modelRuns} />
          <Panel title="Observability Metrics" items={metrics} />
          <Panel title="Export Validations" items={validations} />
          <Panel title="Audit Events" items={audit} wide />
        </div>
      )}
    </div>
  );
}

function Select({ value, onChange, items, labelKey }: { value: string; onChange: (value: string) => void; items: any[]; labelKey: string }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full border rounded-lg px-3 py-2">
      {items.length === 0 && <option value="">None available</option>}
      {items.map((item) => <option key={item.id} value={item.id}>{item[labelKey] || item.id}</option>)}
    </select>
  );
}

function Panel({ title, items, wide = false }: { title: string; items: any[]; wide?: boolean }) {
  return (
    <section className={`bg-white border border-gray-200 rounded-lg p-5 shadow-sm ${wide ? 'xl:col-span-2' : ''}`}>
      <h2 className="font-bold text-lg mb-4">{title}</h2>
      <List items={items} empty="No records yet" />
    </section>
  );
}

function List({ items, empty }: { items: any[]; empty: string }) {
  if (!items.length) return <p className="text-sm text-gray-500">{empty}</p>;
  return (
    <div className="space-y-2 mt-3 max-h-96 overflow-auto">
      {items.map((item, index) => (
        <pre key={item.id || index} className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto">{JSON.stringify(item, null, 2)}</pre>
      ))}
    </div>
  );
}
