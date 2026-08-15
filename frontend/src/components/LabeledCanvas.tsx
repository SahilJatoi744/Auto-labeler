/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { ImageLabels, LabelAnnotation } from '@/types';

export type AnnotationTool = 'select' | 'bbox' | 'polygon';

interface LabeledCanvasProps {
    imageLabels: ImageLabels;
    selectedId?: number | null;
    onSelect?: (id: number | null) => void;
    onAnnotationCreated?: (ann: Partial<LabelAnnotation>) => void;
    onAnnotationUpdated?: (id: number, updates: Partial<LabelAnnotation>) => void;
    tool?: AnnotationTool;
    className?: string;
    zoom?: number;
    onZoomChange?: (zoom: number) => void;
}

const HANDLE_SIZE = 6;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;

// Colors per class for visual distinction
const CLASS_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#10b981',
    '#0ea5e9', '#a855f7', '#eab308', '#06b6d4', '#64748b',
];

function getClassColor(classId: number): string {
    return CLASS_COLORS[classId % CLASS_COLORS.length];
}

/**
 * Point-in-polygon test using ray casting.
 */
function isPointInPolygon(px: number, py: number, polygon: number[][]): boolean {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

export default function LabeledCanvas({
    imageLabels,
    selectedId,
    onSelect,
    onAnnotationCreated,
    onAnnotationUpdated,
    tool = 'select',
    className,
    zoom: externalZoom,
    onZoomChange,
}: LabeledCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);

    // View transform
    const [internalZoom, setInternalZoom] = useState(1);
    const zoom = externalZoom ?? internalZoom;
    const setZoom = useCallback((z: number) => {
        const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
        setInternalZoom(clamped);
        onZoomChange?.(clamped);
    }, [onZoomChange]);

    const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });
    const [fitScale, setFitScale] = useState(1);

    // Hover
    const [hoverId, setHoverId] = useState<number | null>(null);

    // BBox drawing
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPos, setStartPos] = useState<{ x: number, y: number } | null>(null);
    const [mousePos, setMousePos] = useState<{ x: number, y: number } | null>(null);

    // Polygon drawing
    const [currentPoints, setCurrentPoints] = useState<number[][]>([]);

    // Resize handles for bbox
    const [resizing, setResizing] = useState<{ annId: number; handle: string; startBbox: { x: number; y: number; width: number; height: number }; startMouse: { x: number; y: number } } | null>(null);

    // Polygon vertex dragging
    const [draggingVertex, setDraggingVertex] = useState<{ annId: number; contourIdx: number; pointIdx: number } | null>(null);

    // dragging entire annotation
    const [draggingAnnotation, setDraggingAnnotation] = useState<{ annId: number; startBbox: { x: number; y: number; width: number; height: number }; startPolygon?: number[][][]; startMouse: { x: number; y: number } } | null>(null);


    /**
     * Convert screen coordinates to image-space coordinates.
     */
    const screenToImage = useCallback((sx: number, sy: number): { x: number; y: number } => {
        return {
            x: (sx - panOffset.x) / (fitScale * zoom),
            y: (sy - panOffset.y) / (fitScale * zoom),
        };
    }, [panOffset, fitScale, zoom]);

    /**
     * Convert image-space coordinates to screen coordinates.
     */
    const imageToScreen = useCallback((ix: number, iy: number): { x: number; y: number } => {
        return {
            x: ix * fitScale * zoom + panOffset.x,
            y: iy * fitScale * zoom + panOffset.y,
        };
    }, [panOffset, fitScale, zoom]);

    // Image load error state
    const [_imageError, setImageError] = useState(false);

    // Load image and compute fit scale
    useEffect(() => {
        if (!imageLabels.image_url) return;
        setImageError(false);
        const img = new Image();
        img.src = imageLabels.image_url;
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            imageRef.current = img;
            const container = containerRef.current;
            if (!container) return;
            const cW = container.clientWidth;
            const cH = container.clientHeight || 600;
            const s = Math.min(cW / img.width, cH / img.height);
            setFitScale(s);
            // Center the image
            setPanOffset({
                x: (cW - img.width * s) / 2,
                y: (cH - img.height * s) / 2,
            });
        };
        img.onerror = () => {
            console.error('[LabeledCanvas] Failed to load image:', imageLabels.image_url);
            setImageError(true);
            imageRef.current = null;
        };
    }, [imageLabels.image_url]);

    // -- Drawing --
    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        const img = imageRef.current;
        if (!canvas || !container || !img) return;

        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight || 600;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const s = fitScale * zoom;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw image
        ctx.save();
        ctx.translate(panOffset.x, panOffset.y);
        ctx.scale(s, s);
        ctx.drawImage(img, 0, 0);

        // Draw annotations
        imageLabels.annotations.forEach((ann: LabelAnnotation) => {
            const isSelected = selectedId === ann.id;
            const isHovered = hoverId === ann.id;
            const color = getClassColor(ann.class_id);

            // Draw segmentation polygon
            if (ann.segmentation?.polygon) {
                ann.segmentation.polygon.forEach(contour => {
                    if (contour.length < 2) return;
                    ctx.beginPath();
                    contour.forEach((point, i) => {
                        if (i === 0) ctx.moveTo(point[0], point[1]);
                        else ctx.lineTo(point[0], point[1]);
                    });
                    ctx.closePath();
                    ctx.fillStyle = isSelected ? `${color}66` : (isHovered ? `${color}33` : `${color}1a`);
                    ctx.fill();
                    ctx.strokeStyle = isSelected ? color : (isHovered ? color : `${color}99`);
                    ctx.lineWidth = (isSelected ? 2.5 : 1.5) / s;
                    ctx.stroke();

                    // Draw vertices for selected polygon
                    if (isSelected) {
                        contour.forEach(point => {
                            ctx.beginPath();
                            ctx.arc(point[0], point[1], 4 / s, 0, Math.PI * 2);
                            ctx.fillStyle = 'white';
                            ctx.fill();
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.5 / s;
                            ctx.stroke();
                        });
                    }
                });
            }

            // Draw bounding box
            if (ann.bbox) {
                const { x, y, width, height } = ann.bbox;
                ctx.strokeStyle = isSelected ? color : (isHovered ? color : `${color}99`);
                ctx.lineWidth = (isSelected ? 2.5 : 1.5) / s;
                ctx.strokeRect(x, y, width, height);

                // Label badge
                const label = `${ann.class_name || 'Object'} ${Math.round(ann.confidence * 100)}%`;
                const fontSize = Math.max(11, 13 / s);
                ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
                const textW = ctx.measureText(label).width;
                const badgeH = fontSize + 6;
                const badgeY = y - badgeH - 2;

                ctx.fillStyle = isSelected ? color : (isHovered ? color : 'rgba(30, 41, 59, 0.92)');
                const radius = 3 / s;
                const bx = x, by = badgeY, bw = textW + 8, bh = badgeH;
                ctx.beginPath();
                ctx.moveTo(bx + radius, by);
                ctx.lineTo(bx + bw - radius, by);
                ctx.arcTo(bx + bw, by, bx + bw, by + radius, radius);
                ctx.lineTo(bx + bw, by + bh);
                ctx.lineTo(bx, by + bh);
                ctx.arcTo(bx, by, bx + radius, by, radius);
                ctx.closePath();
                ctx.fill();

                ctx.fillStyle = 'white';
                ctx.fillText(label, x + 4, badgeY + fontSize + 1);

                // Draw resize handles for selected bbox
                if (isSelected) {
                    const hs = HANDLE_SIZE / s;
                    const handles = [
                        { x: x, y: y }, { x: x + width / 2, y: y }, { x: x + width, y: y },
                        { x: x, y: y + height / 2 }, { x: x + width, y: y + height / 2 },
                        { x: x, y: y + height }, { x: x + width / 2, y: y + height }, { x: x + width, y: y + height },
                    ];
                    handles.forEach(h => {
                        ctx.fillStyle = 'white';
                        ctx.fillRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 1 / s;
                        ctx.strokeRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
                    });
                }
            }
        });
        ctx.restore();

        // Draw active bbox drawing (in screen space)
        if (isDrawing && tool === 'bbox' && startPos && mousePos) {
            const s1 = imageToScreen(startPos.x, startPos.y);
            const s2 = imageToScreen(mousePos.x, mousePos.y);
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(
                Math.min(s1.x, s2.x), Math.min(s1.y, s2.y),
                Math.abs(s1.x - s2.x), Math.abs(s1.y - s2.y)
            );
            ctx.setLineDash([]);
        }

        // Draw active polygon drawing (in screen space)
        if (tool === 'polygon' && currentPoints.length > 0) {
            ctx.strokeStyle = '#f59e0b';
            ctx.fillStyle = 'rgba(245, 158, 11, 0.15)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            currentPoints.forEach((p, i) => {
                const sp = imageToScreen(p[0], p[1]);
                if (i === 0) ctx.moveTo(sp.x, sp.y);
                else ctx.lineTo(sp.x, sp.y);
            });
            if (mousePos) {
                const sp = imageToScreen(mousePos.x, mousePos.y);
                ctx.lineTo(sp.x, sp.y);
            }
            ctx.stroke();
            if (currentPoints.length > 2) ctx.fill();

            // Vertices
            currentPoints.forEach((p, i) => {
                const sp = imageToScreen(p[0], p[1]);
                ctx.beginPath();
                ctx.arc(sp.x, sp.y, i === 0 ? 5 : 3, 0, Math.PI * 2);
                ctx.fillStyle = i === 0 ? '#f59e0b' : 'white';
                ctx.fill();
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            });
        }

    }, [imageLabels, selectedId, hoverId, fitScale, zoom, panOffset, isDrawing, startPos, mousePos, currentPoints, tool, imageToScreen]);

    // ─── Resize handle hit test ───
    const getResizeHandle = useCallback((ix: number, iy: number, ann: LabelAnnotation): string | null => {
        if (!ann.bbox) return null;
        const { x, y, width, height } = ann.bbox;
        const hs = HANDLE_SIZE / (fitScale * zoom) * 1.5;
        const handles: { name: string; hx: number; hy: number }[] = [
            { name: 'nw', hx: x, hy: y },
            { name: 'n', hx: x + width / 2, hy: y },
            { name: 'ne', hx: x + width, hy: y },
            { name: 'w', hx: x, hy: y + height / 2 },
            { name: 'e', hx: x + width, hy: y + height / 2 },
            { name: 'sw', hx: x, hy: y + height },
            { name: 's', hx: x + width / 2, hy: y + height },
            { name: 'se', hx: x + width, hy: y + height },
        ];
        for (const h of handles) {
            if (Math.abs(ix - h.hx) < hs && Math.abs(iy - h.hy) < hs) return h.name;
        }
        return null;
    }, [fitScale, zoom]);

    // ─── Polygon vertex hit test ───
    const getPolygonVertex = useCallback((ix: number, iy: number, ann: LabelAnnotation): { contourIdx: number; pointIdx: number } | null => {
        if (!ann.segmentation?.polygon) return null;
        const threshold = 6 / (fitScale * zoom);
        for (let ci = 0; ci < ann.segmentation.polygon.length; ci++) {
            const contour = ann.segmentation.polygon[ci];
            for (let pi = 0; pi < contour.length; pi++) {
                const dx = ix - contour[pi][0];
                const dy = iy - contour[pi][1];
                if (Math.sqrt(dx * dx + dy * dy) < threshold) {
                    return { contourIdx: ci, pointIdx: pi };
                }
            }
        }
        return null;
    }, [fitScale, zoom]);

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const imgPos = screenToImage(sx, sy);

        // Middle-click pan
        if (e.button === 1) {
            e.preventDefault();
            setIsPanning(true);
            setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
            return;
        }

        if (tool === 'select') {
            // Check resize handles on selected annotation
            if (selectedId != null) {
                const selectedAnn = imageLabels.annotations.find(a => a.id === selectedId);
                if (selectedAnn) {
                    const handle = getResizeHandle(imgPos.x, imgPos.y, selectedAnn);
                    if (handle && selectedAnn.bbox) {
                        setResizing({
                            annId: selectedAnn.id,
                            handle,
                            startBbox: { ...selectedAnn.bbox },
                            startMouse: imgPos,
                        });
                        return;
                    }

                    // Check polygon vertex
                    const vertex = getPolygonVertex(imgPos.x, imgPos.y, selectedAnn);
                    if (vertex) {
                        setDraggingVertex({ annId: selectedAnn.id, ...vertex });
                        return;
                    }

                    // If we didn't hit a handle or vertex, check if we hit the body for dragging
                    // Check polygon body
                    let hit = false;
                    if (selectedAnn.segmentation?.polygon) {
                        for (const contour of selectedAnn.segmentation.polygon) {
                            if (isPointInPolygon(imgPos.x, imgPos.y, contour)) {
                                hit = true;
                                break;
                            }
                        }
                    }
                    // Check bbox body
                    if (!hit && selectedAnn.bbox) {
                        const { x, y, width, height } = selectedAnn.bbox;
                        if (imgPos.x >= x && imgPos.x <= x + width && imgPos.y >= y && imgPos.y <= y + height) {
                            hit = true;
                        }
                    }

                    if (hit) {
                        setDraggingAnnotation({
                            annId: selectedAnn.id,
                            startBbox: selectedAnn.bbox ? { ...selectedAnn.bbox } : { x: 0, y: 0, width: 0, height: 0 },
                            startPolygon: selectedAnn.segmentation?.polygon ? JSON.parse(JSON.stringify(selectedAnn.segmentation.polygon)) : undefined,
                            startMouse: imgPos,
                        });
                        return;
                    }
                }
            }

            // If nothing hit on selected, check if we should select another one
            if (hoverId !== null && hoverId !== selectedId) {
                onSelect?.(hoverId);
                // We'll let the user drag on the next click for simplicity, or we could set draggingAnnotation here
                // Setting it here for better UX:
                const ann = imageLabels.annotations.find(a => a.id === hoverId);
                if (ann) {
                    setDraggingAnnotation({
                        annId: ann.id,
                        startBbox: ann.bbox ? { ...ann.bbox } : { x: 0, y: 0, width: 0, height: 0 },
                        startPolygon: ann.segmentation?.polygon ? JSON.parse(JSON.stringify(ann.segmentation.polygon)) : undefined,
                        startMouse: imgPos,
                    });
                }
            } else if (hoverId === null) {
                onSelect?.(null);
            }
            setIsDrawing(true);
            setStartPos(imgPos);
            setMousePos(imgPos);
        } else if (tool === 'polygon') {
            // Check if clicking near first point to close
            if (currentPoints.length >= 3) {
                const fp = currentPoints[0];
                const dist = Math.sqrt(Math.pow(imgPos.x - fp[0], 2) + Math.pow(imgPos.y - fp[1], 2));
                if (dist * fitScale * zoom < 10) {
                    completePolygon();
                    return;
                }
            }
            setCurrentPoints([...currentPoints, [imgPos.x, imgPos.y]]);
        }
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const imgPos = screenToImage(sx, sy);

        setMousePos(imgPos);

        // Panning
        if (isPanning) {
            setPanOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
            return;
        }

        // Resizing bbox
        if (resizing && onAnnotationUpdated) {
            const { handle, startBbox, startMouse, annId } = resizing;
            const dx = imgPos.x - startMouse.x;
            const dy = imgPos.y - startMouse.y;
            let newBbox = { ...startBbox };

            if (handle.includes('w')) { newBbox.x = startBbox.x + dx; newBbox.width = startBbox.width - dx; }
            if (handle.includes('e')) { newBbox.width = startBbox.width + dx; }
            if (handle.includes('n')) { newBbox.y = startBbox.y + dy; newBbox.height = startBbox.height - dy; }
            if (handle.includes('s')) { newBbox.height = startBbox.height + dy; }

            // Prevent negative dimensions
            if (newBbox.width < 5) { newBbox.width = 5; }
            if (newBbox.height < 5) { newBbox.height = 5; }

            onAnnotationUpdated(annId, { bbox: newBbox });
            return;
        }

        // Dragging polygon vertex
        if (draggingVertex && onAnnotationUpdated) {
            const ann = imageLabels.annotations.find(a => a.id === draggingVertex.annId);
            if (ann?.segmentation?.polygon) {
                const newPolygon = ann.segmentation.polygon.map((contour, ci) =>
                    ci === draggingVertex.contourIdx
                        ? contour.map((pt, pi) =>
                            pi === draggingVertex.pointIdx ? [imgPos.x, imgPos.y] : [...pt]
                        )
                        : contour.map(pt => [...pt])
                );
                onAnnotationUpdated(draggingVertex.annId, { segmentation: { polygon: newPolygon } });
            }
            return;
        }

        // Dragging entire annotation
        if (draggingAnnotation && onAnnotationUpdated) {
            const { annId, startBbox, startPolygon, startMouse } = draggingAnnotation;
            const dx = imgPos.x - startMouse.x;
            const dy = imgPos.y - startMouse.y;

            const updates: Partial<LabelAnnotation> = {};

            if (startBbox.width > 0) {
                updates.bbox = {
                    ...startBbox,
                    x: startBbox.x + dx,
                    y: startBbox.y + dy
                };
            }

            if (startPolygon) {
                updates.segmentation = {
                    polygon: startPolygon.map(contour =>
                        contour.map(pt => [pt[0] + dx, pt[1] + dy])
                    )
                };
            }

            onAnnotationUpdated(annId, updates);
            return;
        }


        // Hover detection in select mode
        if (tool === 'select' && !resizing && !draggingVertex) {
            let foundId: number | null = null;
            let minArea = Infinity;

            for (const ann of imageLabels.annotations) {
                // Check polygon first (more precise)
                if (ann.segmentation?.polygon) {
                    for (const contour of ann.segmentation.polygon) {
                        if (isPointInPolygon(imgPos.x, imgPos.y, contour)) {
                            // Use bbox area as priority (smaller = preferred)
                            const area = ann.bbox ? ann.bbox.width * ann.bbox.height : Infinity;
                            if (area < minArea) {
                                minArea = area;
                                foundId = ann.id;
                            }
                        }
                    }
                }
                // Check bbox
                if (ann.bbox) {
                    const { x, y, width, height } = ann.bbox;
                    if (imgPos.x >= x && imgPos.x <= x + width && imgPos.y >= y && imgPos.y <= y + height) {
                        const area = width * height;
                        if (area < minArea) {
                            minArea = area;
                            foundId = ann.id;
                        }
                    }
                }
            }
            setHoverId(foundId);
        }
    };

    const handleMouseUp = (_e: React.MouseEvent<HTMLCanvasElement>) => {
        if (isPanning) {
            setIsPanning(false);
            return;
        }

        if (resizing) {
            setResizing(null);
            return;
        }

        if (draggingVertex) {
            setDraggingVertex(null);
            return;
        }

        if (draggingAnnotation) {
            setDraggingAnnotation(null);
            return;
        }


        if (tool === 'bbox' && isDrawing && startPos && mousePos) {
            const x = Math.min(startPos.x, mousePos.x);
            const y = Math.min(startPos.y, mousePos.y);
            const w = Math.abs(startPos.x - mousePos.x);
            const h = Math.abs(startPos.y - mousePos.y);

            if (w > 5 && h > 5) {
                onAnnotationCreated?.({
                    bbox: { x, y, width: w, height: h }
                });
            }
            setIsDrawing(false);
            setStartPos(null);
        }
    };

    const completePolygon = useCallback(() => {
        if (currentPoints.length >= 3) {
            onAnnotationCreated?.({
                segmentation: { polygon: [currentPoints] }
            });
        }
        setCurrentPoints([]);
    }, [currentPoints, onAnnotationCreated]);

    const handleDoubleClick = () => {
        if (tool === 'polygon') {
            completePolygon();
        }
    };

    const handleClick = (_e: React.MouseEvent<HTMLCanvasElement>) => {
        // Don't select if we were panning, resizing, or drawing
        if (isPanning || resizing || draggingVertex || isDrawing) return;
        if (tool === 'select' && onSelect) {
            onSelect(hoverId);
        }
    };

    // Zoom with scroll wheel
    const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = zoom * delta;
        const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

        // Zoom towards cursor
        const factor = clamped / zoom;
        setPanOffset({
            x: sx - (sx - panOffset.x) * factor,
            y: sy - (sy - panOffset.y) * factor,
        });
        setZoom(clamped);
    };

    // Fit to screen
    const fitToScreen = useCallback(() => {
        const container = containerRef.current;
        const img = imageRef.current;
        if (!container || !img) return;
        const cW = container.clientWidth;
        const cH = container.clientHeight || 600;
        const s = Math.min(cW / img.width, cH / img.height);
        setFitScale(s);
        setZoom(1);
        setPanOffset({
            x: (cW - img.width * s) / 2,
            y: (cH - img.height * s) / 2,
        });
    }, [setZoom]);

    // Expose fitToScreen via a data attribute for parent to call
    useEffect(() => {
        const container = containerRef.current;
        if (container) {
            (container as any).__fitToScreen = fitToScreen;
        }
    }, [fitToScreen]);

    const cursorStyle = (() => {
        if (isPanning) return 'grabbing';
        if (tool === 'select') {
            if (resizing) {
                const h = resizing.handle;
                if (h === 'nw' || h === 'se') return 'nwse-resize';
                if (h === 'ne' || h === 'sw') return 'nesw-resize';
                if (h === 'n' || h === 's') return 'ns-resize';
                if (h === 'e' || h === 'w') return 'ew-resize';
            }
            return hoverId ? 'pointer' : 'default';
        }
        return 'crosshair';
    })();

    return (
        <div ref={containerRef} className={`relative w-full h-full bg-slate-900 overflow-hidden ${className}`}>
            <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onDoubleClick={handleDoubleClick}
                onClick={handleClick}
                onWheel={handleWheel}
                onContextMenu={(e) => e.preventDefault()}
                onMouseLeave={() => {
                    setHoverId(null);
                    setMousePos(null);
                    if (tool === 'bbox') setIsDrawing(false);
                    setIsPanning(false);
                }}
                className="absolute inset-0 w-full h-full"
                style={{ cursor: cursorStyle }}
            />
            {!imageLabels.image_url && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 space-y-2">
                    <span className="text-4xl">🖼️</span>
                    <span className="italic">Image source not available</span>
                </div>
            )}

            {/* Zoom indicator */}
            <div className="absolute bottom-3 left-3 px-2 py-1 bg-slate-900/80 backdrop-blur-sm rounded text-[10px] font-mono text-slate-400 border border-slate-700">
                {Math.round(zoom * 100)}%
            </div>
        </div>
    );
}
