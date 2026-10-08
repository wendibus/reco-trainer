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

const parse = (text: string) => Number(text.trim().replace(',', '.'));

/// Web counterpart of FieldGeometryEditor.swift: type the field's real size and
/// click its four corners (top left, top right, bottom right, bottom left) on a
/// reference frame. Corners are stored as fractions (0..1) of the frame so they
/// keep applying however frames are scaled - the same shape the Mac app writes
/// and ml_worker.py's field_membership_checker() reads.
export default function FieldGeometryEditor({ language, api, frame, existing, onSave, onCancel }: Props) {
  const t = extraText[language];
  const [corners, setCorners] = useState<number[][]>(existing?.corners ?? []);
  const [widthText, setWidthText] = useState(existing ? String(existing.realWidth) : '');
  const [lengthText, setLengthText] = useState(existing ? String(existing.realLength) : '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFailed, setImageFailed] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);

  const realWidth = parse(widthText);
  const realLength = parse(lengthText);
  const canSave = corners.length === 4 && realWidth > 0 && realLength > 0 && Number.isFinite(realWidth) && Number.isFinite(realLength) && !saving;

  function addCorner(event: React.MouseEvent<HTMLDivElement>) {
    if (corners.length >= 4 || !imageRef.current) return;
    const box = imageRef.current.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) return;
    const x = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const y = Math.min(1, Math.max(0, (event.clientY - box.top) / box.height));
    setCorners((current) => [...current, [x, y]]);
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

  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <section className="field-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="field-editor-title">
      <h2 id="field-editor-title">{existing ? t.fieldEdit : t.fieldSet}</h2>
      <p className="field-editor-intro">{t.editorIntro}</p>
      <div className="field-editor-controls">
        <label>{t.widthLabel}<input type="text" inputMode="decimal" value={widthText} onChange={(event) => setWidthText(event.target.value)} /></label>
        <label>{t.lengthLabel}<input type="text" inputMode="decimal" value={lengthText} onChange={(event) => setLengthText(event.target.value)} /></label>
        <span className={corners.length < 4 ? 'field-editor-status' : 'field-editor-status done'}>
          {corners.length < 4 ? t.cornerPrompt(corners.length + 1, t.cornerLabels[corners.length]) : `✓ ${t.allSet}`}
        </span>
        <button type="button" className="secondary-button" onClick={() => setCorners([])} disabled={!corners.length}>{t.reset}</button>
      </div>
      <div className="field-editor-stage">
        {imageFailed ? <div className="field-editor-missing">{t.frameMissing}</div> : <div className="field-editor-image-wrap" onClick={addCorner}>
          <img ref={imageRef} src={`${api}/api/frame?id=${encodeURIComponent(frame.id)}`} alt="" draggable={false} onError={() => setImageFailed(true)} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            {corners.length >= 2 && (corners.length === 4
              ? <polygon points={points} className="field-editor-shape closed" vectorEffect="non-scaling-stroke" />
              : <polyline points={points} className="field-editor-shape" vectorEffect="non-scaling-stroke" />)}
          </svg>
          {corners.map(([x, y], index) => <span key={index} className="field-editor-dot" style={{ left: `${x * 100}%`, top: `${y * 100}%` }}><b>{index + 1}</b></span>)}
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
