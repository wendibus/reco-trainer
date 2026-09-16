'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
type SimulationFrame = { file: string; timestamp: number; width: number; height: number; ball: { x: number; y: number; confidence: number } | null };
type Simulation = { fps: number; frames: SimulationFrame[] };
type TrackState = 'detected' | 'interpolated' | 'coasting' | 'lost';
type Resolved = { state: TrackState; x: number | null; y: number | null };

type Props = {
  language: Language;
  api: string;
  busy: boolean;
  operation?: string;
  progress: number;
  message?: string;
  error?: string | null;
  simulation?: Simulation | null;
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const text = {
  de: { title: 'Balltracking-Simulation', intro: 'Kurzes Video wählen und die Ballerkennung des aktiven Modells Bild für Bild abspielen. Das Video wird nur lokal verarbeitet, nie gespeichert oder trainiert.', pick: 'Video auswählen', empty: 'Wähle oben ein kurzes Video aus, um zu starten.', lookahead: 'Wie weit direkt in die Zukunft schauen', frames: 'Bilder', detected: 'Erkannt', interpolated: 'Interpoliert', coasting: 'Gehalten', lost: 'Verloren' },
  en: { title: 'Ball-tracking simulation', intro: "Pick a short clip and play back the active model's ball detection frame by frame. The video is only processed locally, never saved or trained on.", pick: 'Select video', empty: 'Select a short video above to get started.', lookahead: 'How far to look directly into the future', frames: 'frames', detected: 'Detected', interpolated: 'Interpolated', coasting: 'Held', lost: 'Lost' },
  es: { title: 'Simulación de seguimiento del balón', intro: 'Elige un vídeo corto y reproduce la detección del balón del modelo activo cuadro a cuadro. El vídeo solo se procesa localmente, nunca se guarda ni se entrena con él.', pick: 'Seleccionar vídeo', empty: 'Selecciona un vídeo corto arriba para empezar.', lookahead: 'Hasta dónde mirar directamente al futuro', frames: 'imágenes', detected: 'Detectado', interpolated: 'Interpolado', coasting: 'Mantenido', lost: 'Perdido' },
  fr: { title: 'Simulation de suivi du ballon', intro: 'Choisissez une courte vidéo et regardez la détection du ballon du modèle actif image par image. La vidéo n’est traitée que localement, jamais enregistrée ni utilisée pour l’entraînement.', pick: 'Sélectionner une vidéo', empty: 'Sélectionnez une courte vidéo ci-dessus pour commencer.', lookahead: 'Jusqu’où regarder directement dans le futur', frames: 'images', detected: 'Détecté', interpolated: 'Interpolé', coasting: 'Maintenu', lost: 'Perdu' },
} as const;

/// Resolves raw per-frame ball detections into a track: real detections pass
/// through, gaps are bridged by interpolating between the last known and the
/// next real detection when both are within lookaheadFrames, held at the
/// last known position when only a backward one exists within the window,
/// and otherwise marked lost. Pure and frame-index based - mirrors
/// BallTrackingResolver.swift (mac/Sources/RecoTrainerMac/BallTrackingResolver.swift)
/// so the two platforms behave identically; keep both in sync by hand, no
/// shared-code mechanism exists between them.
function resolve(detections: Array<{ x: number; y: number } | null>, lookaheadFrames: number): Resolved[] {
  const window = Math.max(0, lookaheadFrames);
  return detections.map((point, index) => {
    if (point) return { state: 'detected' as const, x: point.x, y: point.y };

    let last: { x: number; y: number; distance: number } | null = null;
    for (let back = index - 1; back >= 0 && index - back <= window; back--) {
      const candidate = detections[back];
      if (candidate) { last = { x: candidate.x, y: candidate.y, distance: index - back }; break; }
    }
    if (!last) return { state: 'lost' as const, x: null, y: null };

    let next: { x: number; y: number; distance: number } | null = null;
    for (let forward = index + 1; forward < detections.length && forward - index <= window; forward++) {
      const candidate = detections[forward];
      if (candidate) { next = { x: candidate.x, y: candidate.y, distance: forward - index }; break; }
    }
    if (!next) return { state: 'coasting' as const, x: last.x, y: last.y };

    const span = last.distance + next.distance;
    const t = span > 0 ? last.distance / span : 0;
    return { state: 'interpolated' as const, x: last.x + (next.x - last.x) * t, y: last.y + (next.y - last.y) * t };
  });
}

const stateColor: Record<TrackState, string> = { detected: '#2fae6b', interpolated: '#e0902f', coasting: '#e0c22f', lost: 'transparent' };

export default function BallTrackingPlayer(props: Props) {
  const t = text[props.language];
  const [lookaheadFrames, setLookaheadFrames] = useState(8);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 640, height: 360 });

  const frames = props.simulation?.frames ?? [];
  const fps = props.simulation?.fps || 12;
  const detections = useMemo(() => frames.map((frame) => frame.ball ? { x: frame.ball.x, y: frame.ball.y } : null), [frames]);
  const resolved = useMemo(() => resolve(detections, lookaheadFrames), [detections, lookaheadFrames]);

  useEffect(() => { setCurrentIndex(0); }, [props.simulation]);

  useEffect(() => {
    if (!isPlaying || frames.length < 2) return;
    const id = setInterval(() => setCurrentIndex((index) => (index + 1) % frames.length), Math.max(30, 1000 / fps));
    return () => clearInterval(id);
  }, [isPlaying, frames.length, fps]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  async function pick() {
    try { await props.post('/api/simulate-ball-tracking', { model: 'nano', threshold: 0.25, language: props.language }); }
    catch { /* surfaced via props.error from worker status */ }
  }

  const frame = frames[currentIndex];
  const position = resolved[currentIndex];
  const fitted = frame ? fitRect(frame.width, frame.height, containerSize.width, containerSize.height) : null;
  const lookaheadSeconds = fps > 0 ? (lookaheadFrames / fps).toFixed(1) : '0.0';

  return <section className="ball-tracking-workspace">
    <div className="benchmark-heading"><div><span className="benchmark-kicker">◆ SIMULATION</span><h2>{t.title}</h2><p>{t.intro}</p></div></div>
    <div className="ball-tracking-controls">
      <button type="button" className="primary-button" onClick={() => void pick()} disabled={props.busy}>{t.pick}</button>
      {props.busy && props.operation?.startsWith('simulation') && <div className="benchmark-progress"><span style={{ width: `${Math.max(props.progress, .02) * 100}%` }} /></div>}
    </div>
    {props.error && <div className="benchmark-error">{props.error}</div>}
    {!frames.length ? <div className="benchmark-empty">{t.empty}</div> : <>
      <div className="ball-tracking-stage" ref={containerRef}>
        {frame && fitted && <>
          <img src={`${props.api}/api/simulation-frame?file=${encodeURIComponent(frame.file)}`} alt="" style={{ position: 'absolute', left: fitted.x, top: fitted.y, width: fitted.width, height: fitted.height }} />
          {position?.x != null && position.y != null && position.state !== 'lost' && <span
            className={`ball-marker ball-marker-${position.state}`}
            style={{
              left: fitted.x + (position.x / frame.width) * fitted.width,
              top: fitted.y + (position.y / frame.height) * fitted.height,
              borderColor: stateColor[position.state],
              background: position.state === 'detected' ? stateColor.detected : 'transparent',
            }}
          />}
        </>}
      </div>
      <div className="ball-tracking-legend">
        <span><i style={{ background: stateColor.detected }} />{t.detected}</span>
        <span><i style={{ borderColor: stateColor.interpolated }} />{t.interpolated}</span>
        <span><i style={{ borderColor: stateColor.coasting }} />{t.coasting}</span>
        <span><i className="lost" />{t.lost}</span>
      </div>
      <div className="ball-tracking-transport">
        <button type="button" className="secondary-button" onClick={() => setIsPlaying((value) => !value)}>{isPlaying ? '⏸' : '▶'}</button>
        <input type="range" min={0} max={Math.max(0, frames.length - 1)} value={currentIndex} onChange={(event) => setCurrentIndex(Number(event.target.value))} />
        <span className="ball-tracking-frame-count">{currentIndex + 1} / {frames.length}</span>
      </div>
      <label className="ball-tracking-lookahead">
        <span>{t.lookahead}</span>
        <input type="range" min={0} max={Math.max(1, Math.round(fps * 5))} value={lookaheadFrames} onChange={(event) => setLookaheadFrames(Number(event.target.value))} />
        <strong>{lookaheadSeconds}s ({lookaheadFrames} {t.frames})</strong>
      </label>
    </>}
  </section>;
}

function fitRect(imageWidth: number, imageHeight: number, containerWidth: number, containerHeight: number) {
  if (imageWidth <= 0 || imageHeight <= 0 || containerWidth <= 0 || containerHeight <= 0) {
    return { x: 0, y: 0, width: containerWidth, height: containerHeight };
  }
  const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height };
}
