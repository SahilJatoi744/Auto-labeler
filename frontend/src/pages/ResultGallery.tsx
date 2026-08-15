/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    getJobResults,
    updateJobResults,
    getJob,
    refineJobResult
} from '@/services/api';
import { ImageLabels, LabelAnnotation, LabelingJob } from '@/types';
import LabeledCanvas from '@/components/LabeledCanvas';
import type { AnnotationTool } from '@/components/LabeledCanvas';
import LoadingSpinner from '@/components/LoadingSpinner';
import {
    ChevronLeft,
    ChevronRight,
    Save,
    Trash2,

    Download,
    LayoutGrid,
    Info,
    CheckCircle2,
    AlertCircle,
    Plus,
    Undo2,
    Redo2,
    Maximize,
    ZoomIn,
    ZoomOut,
    MousePointer,
    Square,
    Pentagon,
} from 'lucide-react';

// ── Undo / Redo stack ───────────────────────────────────────────
interface HistoryEntry {
    annotations: LabelAnnotation[];
}

export default function ResultGallery() {
    const { jobId } = useParams<{ jobId: string }>();
    const navigate = useNavigate();
    const canvasContainerRef = useRef<HTMLDivElement>(null);

    // State
    const [results, setResults] = useState<ImageLabels[]>([]);
    const [currentJob, setCurrentJob] = useState<LabelingJob | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [selectedAnnId, setSelectedAnnId] = useState<number | null>(null);
    const [hasChanges, setHasChanges] = useState(false);
    const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
    const [refinePrompt, setRefinePrompt] = useState('');
    const [activeTool, setActiveTool] = useState<AnnotationTool>('select');
    const [zoom, setZoom] = useState(1);

    // Class selector popup state (shown after drawing a new annotation)
    const [pendingAnnotation, setPendingAnnotation] = useState<Partial<LabelAnnotation> | null>(null);
    const [showClassSelector, setShowClassSelector] = useState(false);
    const [newClassInput, setNewClassInput] = useState('');

    // Undo / redo
    const [undoStack, setUndoStack] = useState<HistoryEntry[]>([]);
    const [redoStack, setRedoStack] = useState<HistoryEntry[]>([]);

    // Save current state to undo stack
    const pushUndo = useCallback(() => {
        if (!results.length) return;
        const current = results[currentIndex];
        setUndoStack(prev => [...prev.slice(-49), { annotations: JSON.parse(JSON.stringify(current.annotations)) }]);
        setRedoStack([]);
    }, [results, currentIndex]);

    const undo = useCallback(() => {
        if (undoStack.length === 0 || !results.length) return;
        const prev = undoStack[undoStack.length - 1];
        const current = results[currentIndex];
        // Save current to redo
        setRedoStack(r => [...r, { annotations: JSON.parse(JSON.stringify(current.annotations)) }]);
        setUndoStack(u => u.slice(0, -1));
        // Apply previous state
        const newResults = [...results];
        newResults[currentIndex] = { ...newResults[currentIndex], annotations: prev.annotations };
        setResults(newResults);
        setHasChanges(true);
        setSelectedAnnId(null);
    }, [undoStack, results, currentIndex]);

    const redo = useCallback(() => {
        if (redoStack.length === 0 || !results.length) return;
        const next = redoStack[redoStack.length - 1];
        const current = results[currentIndex];
        setUndoStack(u => [...u, { annotations: JSON.parse(JSON.stringify(current.annotations)) }]);
        setRedoStack(r => r.slice(0, -1));
        const newResults = [...results];
        newResults[currentIndex] = { ...newResults[currentIndex], annotations: next.annotations };
        setResults(newResults);
        setHasChanges(true);
        setSelectedAnnId(null);
    }, [redoStack, results, currentIndex]);

    // Reset undo/redo when navigating between images
    useEffect(() => {
        setUndoStack([]);
        setRedoStack([]);
        setSelectedAnnId(null);
    }, [currentIndex]);

    // Handle new annotation creation — show class selector
    const handleAnnotationCreated = (partialAnn: Partial<LabelAnnotation>) => {
        if (!currentJob || !results.length) return;
        setPendingAnnotation(partialAnn);
        setShowClassSelector(true);
    };

    // Commit annotation with selected class
    const commitAnnotation = (classId: number, className: string) => {
        if (!pendingAnnotation || !results.length) return;
        pushUndo();

        const newId = Math.max(0, ...results.flatMap(r => r.annotations.map(a => a.id))) + 1;
        const newAnn: LabelAnnotation = {
            id: newId,
            image_id: results[currentIndex].image_id,
            class_id: classId,
            class_name: className,
            confidence: 1.0,
            iscrowd: false,
            ...pendingAnnotation,
        };

        const newResults = [...results];
        const currentItem = { ...newResults[currentIndex] };
        currentItem.annotations = [...currentItem.annotations, newAnn];
        newResults[currentIndex] = currentItem;
        setResults(newResults);
        setHasChanges(true);
        setSelectedAnnId(newId);
        setActiveTool('select');
        setShowClassSelector(false);
        setPendingAnnotation(null);
        setToast({ message: `Created ${className} annotation`, type: 'success' });
    };

    const cancelPendingAnnotation = () => {
        setShowClassSelector(false);
        setPendingAnnotation(null);
    };

    const handleAddClass = (name: string) => {
        if (!currentJob || !name.trim()) return;
        const newId = Math.max(0, ...currentJob.class_hierarchy.classes.map(c => c.id)) + 1;
        const newClass = { id: newId, name: name.trim() };
        const updatedJob = { ...currentJob };
        updatedJob.class_hierarchy = {
            ...updatedJob.class_hierarchy,
            classes: [...updatedJob.class_hierarchy.classes, newClass]
        };
        setCurrentJob(updatedJob);
        setToast({ message: `Added class: ${name}`, type: 'success' });
        return newClass;
    };

    // Handle annotation updates from canvas (resize, vertex drag)
    const handleAnnotationUpdated = (annId: number, updates: Partial<LabelAnnotation>) => {
        const newResults = [...results];
        const currentItem = { ...newResults[currentIndex] };
        currentItem.annotations = currentItem.annotations.map(ann =>
            ann.id === annId ? { ...ann, ...updates } : ann
        );
        newResults[currentIndex] = currentItem;
        setResults(newResults);
        setHasChanges(true);
    };

    // Fetch initial data
    useEffect(() => {
        const fetchData = async () => {
            if (!jobId) return;
            try {
                setLoading(true);
                const [resultsData, jobData] = await Promise.all([
                    getJobResults(jobId),
                    getJob(jobId)
                ]);
                setResults(resultsData.results || []);
                setCurrentJob(jobData);
            } catch (err) {
                setError('Failed to load audit data');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    // Toast auto-dismiss
    useEffect(() => {
        if (toast) {
            const timer = setTimeout(() => setToast(null), 3000);
            return () => clearTimeout(timer);
        }
    }, [toast]);

    const handleSave = useCallback(async () => {
        if (!jobId || !results.length) return;
        try {
            setSaving(true);
            await updateJobResults(jobId, results);
            setHasChanges(false);
            setToast({ message: 'Changes saved successfully', type: 'success' });
        } catch {
            setToast({ message: 'Failed to save changes', type: 'error' });
        } finally {
            setSaving(false);
        }
    }, [jobId, results]);

    const handleRefine = async () => {
        if (!refinePrompt || !jobId || !results[currentIndex]) return;
        try {
            setSaving(true);
            setToast({ message: 'Generating label from prompt...', type: 'success' });
            const newAnn = await refineJobResult(jobId, results[currentIndex].image_id, refinePrompt);
            if (newAnn) {
                pushUndo();
                const newResults = [...results];
                const currentItem = { ...newResults[currentIndex] };
                currentItem.annotations = [...currentItem.annotations, newAnn];
                newResults[currentIndex] = currentItem;
                setResults(newResults);
                setHasChanges(true);
                setSelectedAnnId(newAnn.id);
                setToast({ message: `Added ${newAnn.class_name}`, type: 'success' });
            }
        } catch (err) {
            setToast({ message: 'Refinement failed: ' + ((err as any).response?.data?.detail || 'Unknown error'), type: 'error' });
        } finally {
            setSaving(false);
            setRefinePrompt('');
        }
    };

    const updateAnnotation = (annId: number, updates: Partial<LabelAnnotation>) => {
        pushUndo();

        // If class_name is changed, also sync the class_id from job hierarchy
        if (updates.class_name && !updates.class_id && currentJob) {
            const cls = currentJob.class_hierarchy.classes.find(c => c.name === updates.class_name);
            if (cls) {
                updates.class_id = cls.id;
            }
        } else if (updates.class_id && !updates.class_name && currentJob) {
            const cls = currentJob.class_hierarchy.classes.find(c => c.id === updates.class_id);
            if (cls) {
                updates.class_name = cls.name;
            }
        }

        handleAnnotationUpdated(annId, updates);
    };

    const deleteAnnotation = useCallback((annId: number) => {
        pushUndo();
        const newResults = [...results];
        const currentItem = { ...newResults[currentIndex] };
        currentItem.annotations = currentItem.annotations.filter(ann => ann.id !== annId);
        newResults[currentIndex] = currentItem;
        setResults(newResults);
        setHasChanges(true);
        if (selectedAnnId === annId) setSelectedAnnId(null);
    }, [results, currentIndex, selectedAnnId, pushUndo]);

    const fitToScreen = useCallback(() => {
        const container = canvasContainerRef.current;
        if (container) {
            const fn = (container as any).__fitToScreen;
            if (fn) fn();
        }
    }, []);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

            // Tool selection (Strictly character keys to avoid conflict with class index keys)
            if (e.key.toLowerCase() === 'v') setActiveTool('select');
            if (e.key.toLowerCase() === 'b' || e.key.toLowerCase() === 'r') setActiveTool('bbox');
            if (e.key.toLowerCase() === 'p') setActiveTool('polygon');

            // Navigation
            if (e.key === 'ArrowLeft') setCurrentIndex(i => Math.max(0, i - 1));
            if (e.key === 'ArrowRight') setCurrentIndex(i => Math.min(results.length - 1, i + 1));

            // Delete
            if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnId != null) {
                deleteAnnotation(selectedAnnId);
            }

            // Escape — cancel current drawing or deselect
            if (e.key === 'Escape') {
                setSelectedAnnId(null);
                cancelPendingAnnotation();
                setActiveTool('select');
            }

            // Undo/Redo
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                undo();
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                redo();
            }

            // Save
            if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                handleSave();
            }

            // Fit to screen
            if (e.key === 'f' || e.key === '0') {
                fitToScreen();
            }

            // Quick class assignment via number keys (4-9 for classes)
            if (selectedAnnId != null && currentJob) {
                const num = parseInt(e.key);
                if (num >= 4 && num <= 9) {
                    const classIdx = num - 4;
                    const cls = currentJob.class_hierarchy.classes[classIdx];
                    if (cls) {
                        updateAnnotation(selectedAnnId, { class_id: cls.id, class_name: cls.name });
                        setToast({ message: `Changed class to ${cls.name}`, type: 'success' });
                    }
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [results.length, handleSave, selectedAnnId, deleteAnnotation, undo, redo, fitToScreen, currentJob]);

    if (loading) return (
        <div className="flex flex-col items-center justify-center h-screen bg-slate-950 space-y-4">
            <LoadingSpinner label="Preparing audit suite..." />
        </div>
    );

    if (error || !results.length) return (
        <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
            <div className="max-w-md w-full bg-slate-900 border border-slate-800 p-8 rounded-2xl text-center">
                <AlertCircle className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                <h2 className="text-xl font-bold text-white mb-2">No results available</h2>
                <p className="text-slate-400 mb-6">{error || "Labeling results for this job haven't been generated yet."}</p>
                <div className="flex gap-4">
                    <button onClick={() => navigate('/')} className="flex-1 py-3 bg-slate-800 text-white rounded-xl hover:bg-slate-700 transition-colors font-semibold">Dashboard</button>
                    <button onClick={() => window.location.reload()} className="flex-1 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-semibold">Retry</button>
                </div>
            </div>
        </div>
    );

    const currentItem = results[currentIndex];
    const classDistribution: Record<string, number> = {};
    currentItem.annotations.forEach(a => {
        const name = a.class_name || 'Unknown';
        classDistribution[name] = (classDistribution[name] || 0) + 1;
    });

    return (
        <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
            {/* Header */}
            <header className="h-14 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md px-4 flex items-center justify-between z-10">
                <div className="flex items-center gap-3">
                    <button onClick={() => navigate('/')} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white" title="Back to Dashboard">
                        <ChevronLeft className="w-5 h-5" />
                    </button>
                    <div className="h-5 w-px bg-slate-800 hidden sm:block" />
                    <div className="flex items-center gap-3">
                        <div className="p-1.5 bg-blue-600/20 rounded-lg">
                            <LayoutGrid className="w-4 h-4 text-blue-500" />
                        </div>
                        <div>
                            <h1 className="text-sm font-bold">Annotation Editor</h1>
                            <p className="text-[10px] text-slate-500 flex items-center gap-1">
                                Job: <span className="font-mono">{jobId?.slice(0, 8)}...</span>
                                <span className="mx-1">•</span>
                                {currentJob?.task_type.replace('_', ' ')}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button onClick={() => navigate('/export')} className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg transition-all text-xs font-medium">
                        <Download className="w-3.5 h-3.5" /> Export
                    </button>
                    <button
                        disabled={!hasChanges || saving}
                        onClick={handleSave}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all text-xs font-bold shadow-lg
                            ${hasChanges ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/20' : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'}`}
                    >
                        {saving ? <LoadingSpinner size="sm" /> : <Save className="w-3.5 h-3.5" />}
                        {saving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </header>

            <main className="flex-1 flex overflow-hidden">
                {/* Canvas area */}
                <div className="flex-1 flex flex-col min-w-0 bg-slate-950 relative">
                    <div ref={canvasContainerRef} className="flex-1 relative">
                        <LabeledCanvas
                            imageLabels={currentItem}
                            selectedId={selectedAnnId}
                            onSelect={setSelectedAnnId}
                            onAnnotationCreated={handleAnnotationCreated}
                            onAnnotationUpdated={handleAnnotationUpdated}
                            tool={activeTool}
                            className="w-full h-full"
                            zoom={zoom}
                            onZoomChange={setZoom}
                        />
                    </div>

                    {/* Toolbar — left side */}
                    <div className="absolute top-3 left-3 flex flex-col gap-1 p-1 bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-700 shadow-2xl z-20">
                        <button onClick={() => setActiveTool('select')}
                            className={`p-2 rounded-lg transition-all ${activeTool === 'select' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            title="Select (V)">
                            <MousePointer className="w-4 h-4" />
                        </button>
                        <button onClick={() => setActiveTool('bbox')}
                            className={`p-2 rounded-lg transition-all ${activeTool === 'bbox' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            title="Bounding Box (B)">
                            <Square className="w-4 h-4" />
                        </button>
                        <button onClick={() => setActiveTool('polygon')}
                            className={`p-2 rounded-lg transition-all ${activeTool === 'polygon' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            title="Polygon (P)">
                            <Pentagon className="w-4 h-4" />
                        </button>

                        <div className="h-px bg-slate-700 my-0.5" />

                        <button onClick={undo} disabled={undoStack.length === 0}
                            className={`p-2 rounded-lg transition-all ${undoStack.length > 0 ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-600 cursor-not-allowed'}`}
                            title="Undo (Ctrl+Z)">
                            <Undo2 className="w-4 h-4" />
                        </button>
                        <button onClick={redo} disabled={redoStack.length === 0}
                            className={`p-2 rounded-lg transition-all ${redoStack.length > 0 ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-600 cursor-not-allowed'}`}
                            title="Redo (Ctrl+Y)">
                            <Redo2 className="w-4 h-4" />
                        </button>

                        <div className="h-px bg-slate-700 my-0.5" />

                        <button onClick={() => setZoom(z => Math.min(MAX_ZOOM, z * 1.2))}
                            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
                            title="Zoom In">
                            <ZoomIn className="w-4 h-4" />
                        </button>
                        <button onClick={() => setZoom(z => Math.max(MIN_ZOOM, z * 0.8))}
                            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
                            title="Zoom Out">
                            <ZoomOut className="w-4 h-4" />
                        </button>
                        <button onClick={fitToScreen}
                            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
                            title="Fit to Screen (F)">
                            <Maximize className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Navigation bar */}
                    <div className="absolute inset-x-0 bottom-6 flex justify-center items-center gap-4 z-20">
                        <button onClick={() => setCurrentIndex(i => Math.max(0, i - 1))} disabled={currentIndex === 0}
                            className="p-2.5 bg-slate-900/80 hover:bg-blue-600 rounded-full border border-slate-700 disabled:opacity-30 disabled:hover:bg-slate-900/80 transition-all shadow-xl">
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <div className="px-4 py-1.5 bg-slate-900/90 rounded-full border border-slate-700 text-xs font-mono shadow-xl">
                            {currentIndex + 1} <span className="text-slate-500">/</span> {results.length}
                        </div>
                        <button onClick={() => setCurrentIndex(i => Math.min(results.length - 1, i + 1))} disabled={currentIndex === results.length - 1}
                            className="p-2.5 bg-slate-900/80 hover:bg-blue-600 rounded-full border border-slate-700 disabled:opacity-30 disabled:hover:bg-slate-900/80 transition-all shadow-xl">
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Right sidebar */}
                <div className="w-72 h-full bg-slate-900/50 border-l border-slate-800 flex flex-col overflow-hidden">
                    {/* Metadata + quick tools */}
                    <div className="p-4 border-b border-slate-800 space-y-4">
                        <div>
                            <div className="flex items-center gap-2 mb-1">
                                <Info className="w-3.5 h-3.5 text-slate-400" />
                                <span className="text-[10px] font-bold text-slate-300 uppercase tracking-wider">Image Metadata</span>
                            </div>
                            <p className="text-[10px] font-mono text-slate-500 truncate">ID: {currentItem.image_id}</p>
                        </div>

                        {/* Quick Prompt */}
                        <div>
                            <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 block">AI Quick Prompt</label>
                            <div className="flex gap-1.5">
                                <input type="text" value={refinePrompt} onChange={(e) => setRefinePrompt(e.target.value)}
                                    placeholder="e.g. 'traffic sign'"
                                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg py-1.5 px-2.5 text-xs text-white outline-none focus:border-blue-500"
                                    onKeyDown={(e) => e.key === 'Enter' && handleRefine()} />
                                <button onClick={handleRefine} className="p-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors">
                                    <Plus className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>

                        {/* Class distribution */}
                        {Object.keys(classDistribution).length > 0 && (
                            <div>
                                <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 block">Class Distribution</label>
                                <div className="flex flex-wrap gap-1">
                                    {Object.entries(classDistribution).map(([name, count]) => (
                                        <span key={name} className="px-2 py-0.5 bg-slate-800 text-slate-300 rounded text-[10px]">
                                            {name}: {count}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Annotations list */}
                    <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Annotations ({currentItem.annotations.length})</h3>
                        </div>

                        {currentItem.annotations.map((ann) => (
                            <div key={ann.id} onClick={() => setSelectedAnnId(ann.id)}
                                className={`p-2.5 rounded-xl border transition-all cursor-pointer group relative
                                    ${selectedAnnId === ann.id
                                        ? 'bg-blue-600/10 border-blue-500 shadow-lg shadow-blue-900/10'
                                        : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800 hover:border-slate-600'}`}>
                                <div className="flex justify-between items-start gap-2">
                                    <div className="flex-1 min-w-0">
                                        <div className="mb-1.5 min-w-0">
                                            <select
                                                value={ann.class_name}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => updateAnnotation(ann.id, { class_name: e.target.value })}
                                                className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg py-1 px-1.5 text-[11px] font-bold text-slate-200 hover:border-blue-500/50 transition-colors cursor-pointer outline-none truncate"
                                            >
                                                {currentJob?.class_hierarchy.classes.map(c => (
                                                    <option key={c.id} value={c.name}>{c.name}</option>
                                                ))}
                                                {!currentJob?.class_hierarchy.classes.some(c => c.name === ann.class_name) && (
                                                    <option value={ann.class_name}>{ann.class_name}</option>
                                                )}
                                            </select>
                                        </div>
                                        <div className="flex items-center gap-1.5 font-mono text-[9px] text-slate-500">
                                            <span className={`w-1.5 h-1.5 rounded-full ${selectedAnnId === ann.id ? 'bg-blue-500' : 'bg-slate-600'}`} />
                                            <span>{Math.round(ann.confidence * 100)}%</span>
                                            {ann.segmentation && <span className="bg-purple-900/20 text-purple-400 px-1 rounded-[4px] border border-purple-500/20">MASK</span>}
                                            {ann.bbox && <span className="bg-blue-900/20 text-blue-400 px-1 rounded-[4px] border border-blue-500/20">BBOX</span>}
                                        </div>
                                    </div>
                                    <button onClick={(e) => { e.stopPropagation(); deleteAnnotation(ann.id); }}
                                        className="p-1 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity mt-1">
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                            </div>
                        ))}

                        {currentItem.annotations.length === 0 && (
                            <div className="flex flex-col items-center justify-center py-10 text-slate-600 space-y-2">
                                <AlertCircle className="w-8 h-8 opacity-20" />
                                <p className="text-xs italic">No annotations</p>
                                <p className="text-[10px] text-slate-700">Use B or P to start annotating</p>
                            </div>
                        )}
                    </div>

                    {/* Keyboard shortcuts help */}
                    <div className="p-3 border-t border-slate-800 bg-slate-900/80">
                        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[9px] text-slate-600">
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">V</kbd> Select</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">B</kbd> BBox</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">P</kbd> Polygon</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">Del</kbd> Delete</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">⌘Z</kbd> Undo</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">⌘Y</kbd> Redo</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">F</kbd> Fit</span>
                            <span><kbd className="px-1 bg-slate-800 rounded text-slate-400">⌘S</kbd> Save</span>
                        </div>
                    </div>
                </div>
            </main>

            {/* Class Selector Popup */}
            {showClassSelector && (
                <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center backdrop-blur-sm" onClick={cancelPendingAnnotation}>
                    <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6 w-80 max-h-96 overflow-y-auto" onClick={(e) => e.stopPropagation()}>
                        <h3 className="text-sm font-bold text-white mb-3">Select Class</h3>
                        <div className="space-y-1.5 mb-4">
                            {currentJob?.class_hierarchy.classes.map(c => (
                                <button key={c.id} onClick={() => commitAnnotation(c.id, c.name)}
                                    className="w-full text-left px-3 py-2 rounded-lg hover:bg-blue-600/20 text-sm text-slate-200 transition-colors border border-transparent hover:border-blue-500/50">
                                    {c.name}
                                </button>
                            ))}
                        </div>
                        <div className="border-t border-slate-700 pt-3">
                            <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 block">Add New Class</label>
                            <div className="flex gap-1.5">
                                <input type="text" value={newClassInput} onChange={(e) => setNewClassInput(e.target.value)}
                                    placeholder="Class name..."
                                    className="flex-1 bg-slate-800 border border-slate-600 rounded-lg py-1.5 px-2.5 text-xs text-white outline-none focus:border-blue-500"
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && newClassInput.trim()) {
                                            const newCls = handleAddClass(newClassInput);
                                            if (newCls) {
                                                commitAnnotation(newCls.id, newCls.name);
                                                setNewClassInput('');
                                            }
                                        }
                                    }} />
                                <button onClick={() => {
                                    if (newClassInput.trim()) {
                                        const newCls = handleAddClass(newClassInput);
                                        if (newCls) {
                                            commitAnnotation(newCls.id, newCls.name);
                                            setNewClassInput('');
                                        }
                                    }
                                }} className="p-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors">
                                    <Plus className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                        <button onClick={cancelPendingAnnotation} className="w-full mt-3 py-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors">Cancel</button>
                    </div>
                </div>
            )}

            {/* Toast */}
            {toast && (
                <div className={`fixed bottom-6 right-6 px-4 py-2.5 rounded-xl shadow-2xl border flex items-center gap-2 animate-in slide-in-from-bottom-4 z-50
                    ${toast.type === 'success' ? 'bg-emerald-900/90 border-emerald-500/50 text-emerald-100' : 'bg-red-900/90 border-red-500/50 text-red-100'}`}>
                    {toast.type === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                    <span className="font-medium text-xs">{toast.message}</span>
                </div>
            )}
        </div>
    );
}

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;
