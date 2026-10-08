'use client';

import { useRef, useState } from 'react';
import { extraText, type ExtraLanguage } from './extraText';

export type FieldGeometry = { corners: number[][]; realWidth: number; realLength: number };

type Props = {
  language: ExtraLanguage;
  api: string;
  frame: { id: string };
  existing?: FieldGeometry | null;
  onSave: (geometry: FieldGeometry) => Promise<void>;
  onCancel: () => void;
};

const MAX_POINTS = 32;
const parse = (text: string) => Number(text.trim().replace(',', '.'));

/// Web counterpart of FieldGeometryEditor.swift: click the points of the field's
/// outline on a reference frame. Four points are the corners of a rectangular
/// field (top left, top right, bottom right, bottom left) and need the field's
/// real size; more points trace any other shape and make the size optional.
/// Points can be dragged and the last one removed. They are stored as fractions
/// (0..1) of the frame, the shape ml_worker.py's field_membership_checker() reads.
export default function FieldGeometryEditor({ language, api, frame, existing, onSave, onCancel }: Props) {
  const t = extraText[language];
  const [corners, setCorners] = useState<number[][]>(existing?.corners ?? []);
  const [widthText, setWidthText] = useState(existing && existing.realWidth > 0 ? String(existing.realWidth) : '');
  const [lengthText, setLengthText] = useState(existing && existing.realLength > 0 ? String(existing.realLength) : '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFailed, setImageFailed] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragIndex = useRef<number | null>(null);

  const count = corners.length;
  const widthEntered = widthText.trim() !== '';
  const lengthEntered = lengthText.trim() !== '';
  const realWidth = widthEntered ? parse(widthText) : 0;
  const realLength = lengthEntered ? parse(lengthText) : 0;
  const sizeValid = Number.isFinite(realWidth) && Number.isFinite(realLength) && (count === 4 ? realWidth > 0 && realLength > 0 : realWidth >= 0 && realLength >= 0);
  const canSave = count >= 4 && sizeValid && !saving;

  function fractionFrom(event: { clientX: number; clientY: number }): number[] | null {
    const box = imageRef.current?.getBoundingClientRect();
    if (!box || box.width <= 0 || box.height <= 0) return null;
    return [Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)), Math.min(1, Math.max(0, (event.clientY - box.top) / box.height))];
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
      await onSave({ corners, realWidth, realLength });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      setSaving(false);
    }
  }

  const points = corners.map(([x, y]) => `${x},${y}`).join(' ');
  const statusText = count < 4 ? t.cornerPrompt(count + 1, t.cornerLabels[count]) : count === 4 ? `✓ ${t.allSet}` : `✓ ${t.pointCount(count)}`;

  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <section className="field-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="field-editor-title">
      <h2 id="field-editor-title">{existing ? t.fieldEdit : t.fieldSet}</h2>
      <p className="field-editor-intro">{t.editorIntro}</p>
      <div className="field-editor-controls">
        <label>{t.widthLabel}<input type="text" inputMode="decimal" value={widthText} onChange={(event) => setWidthText(event.target.value)} /></label>
        <label>{t.lengthLabel}<input type="text" inputMode="decimal" value={lengthText} onChange={(event) => setLengthText(event.target.value)} /></label>
        {count > 4 && <small className="field-editor-optional">{t.sizeOptional.trim()}</small>}
        <span className={count < 4 ? 'field-editor-status' : 'field-editor-status done'}>{statusText}</span>
        <button type="button" className="secondary-button" onClick={() => setCorners((current) => current.slice(0, -1))} disabled={!count}>{t.removeLast}</button>
        <button type="button" className="secondary-button" onClick={() => setCorners([])} disabled={!count}>{t.reset}</button>
      </div>
      <p className="field-editor-hint">{t.moreHint}</p>
      <div className="field-editor-stage">
        {imageFailed ? <div className="field-editor-missing">{t.frameMissing}</div> : <div className="field-editor-image-wrap" onClick={addPoint}>
          <img ref={imageRef} src={`${api}/api/frame?id=${encodeURIComponent(frame.id)}`} alt="" draggable={false} onError={() => setImageFailed(true)} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            {count >= 2 && (count >= 4
              ? <polygon points={points} className="field-editor-shape closed" vectorEffect="non-scaling-stroke" />
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
        <button type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
        <button type="button" className="primary-button" onClick={() => void save()} disabled={!canSave}>{t.save}</button>
      </div>
    </section>
  </div>;
}
