import { useRef, useState } from 'react';
import { extraText, type ExtraLanguage } from './extraText';
import { hintText } from './hints';

/** Older projects may still carry realWidth/realLength; they are no longer used. */
export type FieldGeometry = { corners: number[][]; realWidth?: number; realLength?: number };

type Props = {
  language: ExtraLanguage;
  api: string;
  frame: { id: string };
  existing?: FieldGeometry | null;
  onSave: (geometry: FieldGeometry) => Promise<void>;
  onCancel: () => void;
};

const MAX_POINTS = 32;
const EDGE_SNAP_PX = 8;

/** True when two non-neighbouring edges of the closed outline cross (points clicked out of order). */
export function outlineCrosses(points: number[][]): boolean {
  const orientation = (a: number[], b: number[], c: number[]) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const count = points.length;
  for (let first = 0; first < count; first += 1) {
    for (let second = first + 1; second < count; second += 1) {
      if (second === first + 1 || (first === 0 && second === count - 1)) continue;
      const [a, b] = [points[first], points[(first + 1) % count]];
      const [c, d] = [points[second], points[(second + 1) % count]];
      if (orientation(c, d, a) * orientation(c, d, b) < 0 && orientation(a, b, c) * orientation(a, b, d) < 0) return true;
    }
  }
  return false;
}

/// Click at least four points along the field's edge on a reference frame; straight
/// lines join them into a closed outline. Points on the image edge cover a field that
/// leaves the frame. Points can be dragged and the last one removed. They are stored as
/// fractions (0..1) of the frame, the shape ml_worker.py's field_membership_checker() reads.
export default function FieldGeometryEditor({ language, api, frame, existing, onSave, onCancel }: Props) {
  const t = extraText[language];
  const [corners, setCorners] = useState<number[][]>(existing?.corners ?? []);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFailed, setImageFailed] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragIndex = useRef<number | null>(null);

  const count = corners.length;
  const crosses = count >= 4 && outlineCrosses(corners);
  const canSave = count >= 4 && !crosses && !saving;

  function fractionFrom(event: { clientX: number; clientY: number }): number[] | null {
    const box = imageRef.current?.getBoundingClientRect();
    if (!box || box.width <= 0 || box.height <= 0) return null;
    // Clicks just outside the image (the catcher around it) or within a few pixels of its
    // edge land exactly on the edge - that is where a field leaving the frame is marked.
    const snap = (offset: number, size: number) => offset < EDGE_SNAP_PX ? 0 : offset > size - EDGE_SNAP_PX ? 1 : offset / size;
    return [snap(event.clientX - box.left, box.width), snap(event.clientY - box.top, box.height)];
  }

  function addPoint(event: React.MouseEvent<HTMLDivElement>) {
    if (count >= MAX_POINTS) return;
    const point = fractionFrom(event);
    if (point) setCorners((current) => [...current, point]);
  }

  function startDrag(event: React.PointerEvent<HTMLSpanElement>, index: number) {
    event.stopPropagation();
    dragIndex.current = index;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveDrag(event: React.PointerEvent<HTMLSpanElement>) {
    const index = dragIndex.current;
    const point = fractionFrom(event);
    if (index == null || !point) return;
    setCorners((current) => current.map((entry, position) => position === index ? point : entry));
  }

  async function save() {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      await onSave({ corners });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setSaving(false);
    }
  }

  const points = corners.map(([x, y]) => `${x},${y}`).join(' ');
  const statusText = count < 4 ? t.pointPrompt(count + 1) : crosses ? `⚠ ${t.crossing}` : `✓ ${t.pointCount(count)}`;

  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <section className="field-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="field-editor-title">
      <h2 id="field-editor-title">{existing ? t.fieldEdit : t.fieldSet}</h2>
      <p className="field-editor-intro">{t.editorIntro}</p>
      <div className="field-editor-controls">
        <span className={count < 4 || crosses ? 'field-editor-status' : 'field-editor-status done'}>{statusText}</span>
        <button data-hint={hintText[language].removePoint} type="button" className="secondary-button" onClick={() => setCorners((current) => current.slice(0, -1))} disabled={!count}>{t.removeLast}</button>
        <button data-hint={hintText[language].resetPoints} type="button" className="secondary-button" onClick={() => setCorners([])} disabled={!count}>{t.reset}</button>
      </div>
      <p className="field-editor-hint">{t.moreHint}</p>
      <div className="field-editor-stage">
        {imageFailed ? <div className="field-editor-missing">{t.frameMissing}</div> : <div className="field-editor-image-wrap" onClick={addPoint}>
          <span className="field-editor-edge-catcher" aria-hidden="true" />
          <img ref={imageRef} src={`${api}/api/frame?id=${encodeURIComponent(frame.id)}`} alt="" draggable={false} onError={() => setImageFailed(true)} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            {count >= 2 && (count >= 4
              ? <polygon points={points} className={`field-editor-shape closed${crosses ? ' crossing' : ''}`} vectorEffect="non-scaling-stroke" />
              : <polyline points={points} className="field-editor-shape" vectorEffect="non-scaling-stroke" />)}
          </svg>
          {corners.map(([x, y], index) => <span
            key={index}
            className="field-editor-dot"
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
            onClick={(event) => event.stopPropagation()}
            onPointerDown={(event) => startDrag(event, index)}
            onPointerMove={moveDrag}
            onPointerUp={() => { dragIndex.current = null; }}
            onPointerCancel={() => { dragIndex.current = null; }}
          ><b>{index + 1}</b></span>)}
        </div>}
      </div>
      {error && <div className="benchmark-error">{error}</div>}
      <div className="field-editor-actions">
        <button data-hint={hintText[language].cancel} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
        <button data-hint={hintText[language].saveField} type="button" className="primary-button" onClick={() => void save()} disabled={!canSave}>{t.save}</button>
      </div>
    </section>
  </div>;
}
