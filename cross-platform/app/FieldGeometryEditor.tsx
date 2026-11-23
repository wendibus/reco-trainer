import { useRef, useState } from 'react';
import { extraText, type ExtraLanguage } from './extraText';
import { hintText } from './hints';

/** Older projects may still carry realWidth/realLength; they are no longer used. */
export type FieldGeometry = { corners: number[][]; realWidth?: number; realLength?: number };

type Props = {
  language: ExtraLanguage;
  api: string;
  frame: { id: string; videoName: string };
  existing?: FieldGeometry | null;
  /** Number of videos in the project; from two on, the outline can be used for all of them. */
  videoCount: number;
  onSave: (geometry: FieldGeometry, applyToAll: boolean) => Promise<void>;
  /** Deletes the stored outline (this video, or all with applyToAll). */
  onRemove: (applyToAll: boolean) => Promise<void>;
  onCancel: () => void;
};

const MAX_POINTS = 32;
const MIN_HIDE = 0.02;
type Rect = [number, number, number, number];
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
export default function FieldGeometryEditor({ language, api, frame, existing, videoCount, onSave, onRemove, onCancel }: Props) {
  const t = extraText[language];
  const [corners, setCorners] = useState<number[][]>(existing?.corners ?? []);
  const [saving, setSaving] = useState(false);
  const [applyToAll, setApplyToAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFailed, setImageFailed] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragIndex = useRef<number | null>(null);
  const [hidden, setHidden] = useState<Rect[]>([]);
  const [hiding, setHiding] = useState(false);
  const [draft, setDraft] = useState<Rect | null>(null);
  const [finding, setFinding] = useState(false);
  const [found, setFound] = useState<number | null>(null);

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
    if (hiding || count >= MAX_POINTS) return;
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

  function startHide(event: React.PointerEvent<HTMLDivElement>) {
    if (!hiding) return;
    const point = fractionFrom(event);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDraft([point[0], point[1], point[0], point[1]]);
  }

  function moveHide(event: React.PointerEvent<HTMLDivElement>) {
    const point = fractionFrom(event);
    if (draft && point) setDraft([draft[0], draft[1], point[0], point[1]]);
  }

  function endHide() {
    if (draft && Math.abs(draft[2] - draft[0]) >= MIN_HIDE && Math.abs(draft[3] - draft[1]) >= MIN_HIDE) {
      setHidden((current) => [...current, [Math.min(draft[0], draft[2]), Math.min(draft[1], draft[3]), Math.max(draft[0], draft[2]), Math.max(draft[1], draft[3])]]);
    }
    setDraft(null);
  }

  async function findAutomatically() {
    setFinding(true);
    setError(null);
    try {
      const response = await fetch(`${api}/api/field-detect`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frameId: frame.id, ignore: hidden, language }) });
      const result = (await response.json()) as { error?: string; corners?: number[][] };
      if (!response.ok || !result.corners) throw new Error(result.error || 'Local worker failed');
      setCorners(result.corners);
      setFound(result.corners.length);
      setHiding(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setFinding(false);
    }
  }

  async function save() {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      await onSave({ corners }, applyToAll);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setSaving(false);
    }
  }

  async function remove() {
    // Two clicks instead of a browser confirm(): the first arms the button, the second deletes.
    if (!confirmRemove) { setConfirmRemove(true); return; }
    setSaving(true);
    setError(null);
    try {
      await onRemove(applyToAll);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setSaving(false);
      setConfirmRemove(false);
    }
  }

  const points = corners.map(([x, y]) => `${x},${y}`).join(' ');
  const statusText = count < 4 ? t.pointPrompt(count + 1) : crosses ? `⚠ ${t.crossing}` : `✓ ${t.pointCount(count)}`;

  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <section className="field-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="field-editor-title">
      <h2 id="field-editor-title">{existing ? t.fieldEdit : t.fieldSet}</h2>
      <p className="field-editor-video" data-hint={t.fieldPerVideo}>🎥 {t.fieldVideo(frame.videoName)}</p>
      <p className="field-editor-intro">{t.editorIntro}</p>
      <div className="field-editor-controls">
        <span className={count < 4 || crosses ? 'field-editor-status' : 'field-editor-status done'}>{statusText}</span>
        <button data-hint={hintText[language].removePoint} type="button" className="secondary-button" onClick={() => setCorners((current) => current.slice(0, -1))} disabled={!count}>{t.removeLast}</button>
        <button data-hint={hintText[language].resetPoints} type="button" className="secondary-button" onClick={() => setCorners([])} disabled={!count}>{t.reset}</button>
      </div>
      <div className="field-editor-controls">
        <button data-hint={t.autoHint} type="button" className="primary-button" onClick={() => void findAutomatically()} disabled={finding || saving}>{finding ? t.autoFinding : t.autoFind}</button>
        <button data-hint={t.hideHint} type="button" className={`secondary-button${hiding ? ' active' : ''}`} onClick={() => setHiding((current) => !current)} disabled={finding}>{hiding ? t.hideDone : t.hideAreas}</button>
        {hidden.length > 0 && <button data-hint={t.hideClear} type="button" className="secondary-button" onClick={() => setHidden([])} disabled={finding}>{t.hideClear} · {t.hideCount(hidden.length)}</button>}
      </div>
      {hiding ? <p className="field-editor-hint">{t.hideHint}</p> : found !== null && <p className="field-editor-hint">{t.autoDone(found)}</p>}
      <p className="field-editor-hint">{t.moreHint}</p>
      <div className="field-editor-stage">
        {imageFailed ? <div className="field-editor-missing">{t.frameMissing}</div> : <div className={`field-editor-image-wrap${hiding ? ' hiding' : ''}`} onClick={addPoint} onPointerDown={startHide} onPointerMove={moveHide} onPointerUp={endHide} onPointerCancel={() => setDraft(null)}>
          <span className="field-editor-edge-catcher" aria-hidden="true" />
          <img ref={imageRef} src={`${api}/api/frame?id=${encodeURIComponent(frame.id)}`} alt="" draggable={false} onError={() => setImageFailed(true)} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            {count >= 4 && !crosses && <path className="field-editor-outside" fillRule="evenodd" d={`M0,0H1V1H0Z M${points.replace(/ /g, ' L').replace(/^/, '')}Z`} />}
            {[...hidden, ...(draft ? [[Math.min(draft[0], draft[2]), Math.min(draft[1], draft[3]), Math.max(draft[0], draft[2]), Math.max(draft[1], draft[3])] as Rect] : [])].map(([x0, y0, x1, y1], index) => <rect key={index} className="field-editor-hidden" x={x0} y={y0} width={x1 - x0} height={y1 - y0} vectorEffect="non-scaling-stroke" />)}
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
        {existing && <button data-hint={hintText[language].removeField} type="button" className={`secondary-button field-editor-remove${confirmRemove ? ' armed' : ''}`} onClick={() => void remove()} onBlur={() => setConfirmRemove(false)} disabled={saving}>{confirmRemove ? t.fieldRemoveConfirm : t.fieldRemove}</button>}
        {videoCount > 1 && <label className="cuda-toggle field-editor-all" data-hint={t.fieldPerVideo}><input type="checkbox" checked={applyToAll} onChange={(event) => setApplyToAll(event.target.checked)} />{t.fieldApplyAll(videoCount)}</label>}
        <button data-hint={hintText[language].cancel} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
        <button data-hint={hintText[language].saveField} type="button" className="primary-button" onClick={() => void save()} disabled={!canSave}>{t.save}</button>
      </div>
    </section>
  </div>;
}
