import { useEffect, useMemo, useRef, useState } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type SimulationFrame = { file: string; timestamp: number; width: number; height: number; ball: { x: number; y: number; confidence: number } | null };
type Simulation = { id?: string; fps: number; frames: SimulationFrame[]; slot?: 'a' | 'b'; video?: string; model?: string; package?: string | null; threshold?: number };
type TrackState = 'detected' | 'interpolated' | 'coasting' | 'lost';
type Resolved = { state: TrackState; x: number | null; y: number | null };
type Slot = 'a' | 'b';
type LibraryModel = { packageID: string; displayName?: string; modelSize?: string; classes: string[]; sport?: string; isActive?: boolean };

type Props = {
  language: Language;
  api: string;
  busy: boolean;
  operation?: string;
  progress: number;
  message?: string;
  error?: string | null;
  simulation?: Simulation | null;
  simulationB?: Simulation | null;
  models?: LibraryModel[];
  modelSize?: string;
  sport?: string;
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const text = {
  de: { title: 'Balltracking-Simulation', intro: 'Kurzes Video wählen und die Ballerkennung Bild für Bild abspielen. Im Vergleich laufen zwei Videos mit zwei Modellen synchron nebeneinander. Videos werden nur lokal verarbeitet. Bilder kommen nur ins Projekt, wenn du Lücken als Prüfkandidaten übernimmst.', pick: 'Video auswählen', empty: 'Wähle oben ein kurzes Video aus, um zu starten.', emptySide: 'Noch keine Simulation auf dieser Seite.', lookahead: 'Wie weit direkt in die Zukunft schauen', frames: 'Bilder', detected: 'Erkannt', interpolated: 'Interpoliert', coasting: 'Gehalten', lost: 'Verloren', compare: 'Zwei Modelle vergleichen', single: 'Nur eine Simulation', model: 'Modell', activeModel: 'Aktives Modell', sameVideo: 'Gleiches Video wie A', tracked: 'Ball verfolgt', avgConfidence: 'Ø Sicherheit', onlyA: 'Nur A erkennt', onlyB: 'Nur B erkennt', both: 'Beide erkennen', sameClip: 'Gleiches Video: Unterschiede Bild für Bild', noBallModels: 'Keine Modelle mit Ball-Klasse in der Bibliothek; es läuft das aktive Modell.', gaps: 'Lücken als Prüfkandidaten übernehmen' },
  en: { title: 'Ball-tracking simulation', intro: 'Pick a short clip and play back the ball detection frame by frame. In comparison mode, two videos with two models play in sync side by side. Videos are only processed locally. Images only enter the project when you add gaps as review candidates.', pick: 'Select video', empty: 'Select a short video above to get started.', emptySide: 'No simulation on this side yet.', lookahead: 'How far to look directly into the future', frames: 'frames', detected: 'Detected', interpolated: 'Interpolated', coasting: 'Held', lost: 'Lost', compare: 'Compare two models', single: 'Single simulation', model: 'Model', activeModel: 'Active model', sameVideo: 'Same video as A', tracked: 'Ball tracked', avgConfidence: 'Avg. confidence', onlyA: 'Only A detects', onlyB: 'Only B detects', both: 'Both detect', sameClip: 'Same video: frame-by-frame differences', noBallModels: 'No models with a ball class in the library; the active model is used.', gaps: 'Add gaps as review candidates' },
  es: { title: 'Simulación de seguimiento del balón', intro: 'Elige un vídeo corto y reproduce la detección del balón cuadro a cuadro. En la comparación, dos vídeos con dos modelos se reproducen sincronizados uno al lado del otro. Los vídeos solo se procesan localmente. Las imágenes solo entran en el proyecto cuando añades huecos como candidatos.', pick: 'Seleccionar vídeo', empty: 'Selecciona un vídeo corto arriba para empezar.', emptySide: 'Aún no hay simulación en este lado.', lookahead: 'Hasta dónde mirar directamente al futuro', frames: 'imágenes', detected: 'Detectado', interpolated: 'Interpolado', coasting: 'Mantenido', lost: 'Perdido', compare: 'Comparar dos modelos', single: 'Una sola simulación', model: 'Modelo', activeModel: 'Modelo activo', sameVideo: 'Mismo vídeo que A', tracked: 'Balón seguido', avgConfidence: 'Confianza media', onlyA: 'Solo A detecta', onlyB: 'Solo B detecta', both: 'Ambos detectan', sameClip: 'Mismo vídeo: diferencias imagen por imagen', noBallModels: 'No hay modelos con clase de balón en la biblioteca; se usa el modelo activo.', gaps: 'Añadir huecos como candidatos' },
  fr: { title: 'Simulation de suivi du ballon', intro: 'Choisissez une courte vidéo et regardez la détection du ballon image par image. En mode comparaison, deux vidéos avec deux modèles sont lues côte à côte de façon synchronisée. Les vidéos ne sont traitées que localement. Les images n’entrent dans le projet que si vous ajoutez des lacunes comme candidates.', pick: 'Sélectionner une vidéo', empty: 'Sélectionnez une courte vidéo ci-dessus pour commencer.', emptySide: 'Pas encore de simulation de ce côté.', lookahead: 'Jusqu’où regarder directement dans le futur', frames: 'images', detected: 'Détecté', interpolated: 'Interpolé', coasting: 'Maintenu', lost: 'Perdu', compare: 'Comparer deux modèles', single: 'Une seule simulation', model: 'Modèle', activeModel: 'Modèle actif', sameVideo: 'Même vidéo que A', tracked: 'Ballon suivi', avgConfidence: 'Confiance moy.', onlyA: 'Seul A détecte', onlyB: 'Seul B détecte', both: 'Les deux détectent', sameClip: 'Même vidéo : différences image par image', noBallModels: 'Aucun modèle avec une classe ballon dans la bibliothèque ; le modèle actif est utilisé.', gaps: 'Ajouter les lacunes comme candidates' },
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

type Prepared = { simulation: Simulation; frames: SimulationFrame[]; fps: number; duration: number; resolved: Resolved[] };

function useStableSimulation(simulation: Simulation | null | undefined) {
  // The app re-fetches the worker status every ~900 ms, so the simulation prop is a
  // fresh object on every poll even though nothing changed. Key everything off the
  // simulation's id instead - depending on the object itself restarted the clip from
  // frame 0 on every poll, so it never played past its first second.
  const id = simulation?.id ?? (simulation ? `${simulation.frames.length}:${simulation.fps}` : '');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return { id, simulation: useMemo(() => simulation ?? null, [id]) };
}

function prepare(simulation: Simulation | null, lookaheadFrames: number): Prepared | null {
  if (!simulation || !simulation.frames.length) return null;
  const fps = simulation.fps || 12;
  const detections = simulation.frames.map((frame) => frame.ball ? { x: frame.ball.x, y: frame.ball.y } : null);
  return { simulation, frames: simulation.frames, fps, duration: simulation.frames.length / fps, resolved: resolve(detections, lookaheadFrames) };
}

/// Frame shown at a playback time; clips shorter than the longest one hold their last frame.
function indexAt(prepared: Prepared, time: number) {
  return Math.min(prepared.frames.length - 1, Math.max(0, Math.floor(time * prepared.fps + 1e-6)));
}

function summarize(prepared: Prepared) {
  const counts: Record<TrackState, number> = { detected: 0, interpolated: 0, coasting: 0, lost: 0 };
  for (const item of prepared.resolved) counts[item.state] += 1;
  const confidences = prepared.frames.flatMap((frame) => frame.ball ? [frame.ball.confidence] : []);
  const total = prepared.frames.length;
  return {
    counts, total,
    tracked: total ? (total - counts.lost) / total : 0,
    avgConfidence: confidences.length ? confidences.reduce((sum, value) => sum + value, 0) / confidences.length : null,
  };
}

const percent = (value: number) => `${Math.round(value * 100)} %`;

export default function BallTrackingPlayer(props: Props) {
  const t = text[props.language];
  const h = hintText[props.language];
  const [lookaheadFrames, setLookaheadFrames] = useState(8);
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const a = useStableSimulation(props.simulation);
  const b = useStableSimulation(props.simulationB);
  const [compare, setCompare] = useState(Boolean(props.simulationB));
  const [choice, setChoice] = useState<Record<Slot, string>>({ a: '', b: '' });
  // The worker deletes a slot's old frames before extracting new ones, so that slot's
  // player is hidden while it is being re-simulated instead of requesting missing images.
  // Hidden until that slot reports a new simulation id, or the run fails.
  const [running, setRunning] = useState<{ slot: Slot; previousId: string } | null>(null);
  // Set once gaps were sent to the project, so the worker's result line stays visible afterwards.
  const [gapsSent, setGapsSent] = useState(false);
  const ballClass = props.sport === 'hockey' ? 'puck' : 'ball';
  const ballModels = useMemo(() => (props.models ?? []).filter((model) => model.classes?.includes(ballClass) && (!props.sport || !model.sport || model.sport === props.sport)), [props.models, props.sport, ballClass]);

  const preparedA = useMemo(() => prepare(a.simulation, lookaheadFrames), [a.simulation, lookaheadFrames]);
  const preparedB = useMemo(() => compare ? prepare(b.simulation, lookaheadFrames) : null, [compare, b.simulation, lookaheadFrames]);
  const shown = [preparedA, preparedB].filter((item): item is Prepared => item !== null);
  const duration = Math.max(0, ...shown.map((item) => item.duration));
  const step = 1 / Math.max(1, ...shown.map((item) => item.fps));

  // Restart only when a different simulation arrives (reset while rendering, not in an effect).
  const shownKey = `${a.id}|${compare ? b.id : ''}`;
  const [lastKey, setLastKey] = useState(shownKey);
  if (lastKey !== shownKey) { setLastKey(shownKey); setTime(0); }

  useEffect(() => {
    if (!isPlaying || duration <= step) return;
    const id = setInterval(() => setTime((value) => (value + step >= duration - 1e-6 ? 0 : value + step)), Math.max(30, 1000 * step));
    return () => clearInterval(id);
  }, [isPlaying, duration, step]);

  async function addGaps(slot: Slot) {
    setGapsSent(true);
    try { await props.post('/api/simulation-gaps', { slot, model: props.modelSize || 'nano', language: props.language }); } catch { /* surfaced via props.error from worker status */ }
  }

  async function run(slot: Slot, sameVideo = false) {
    setGapsSent(false);
    setRunning({ slot, previousId: slot === 'a' ? a.id : b.id });
    try {
      await props.post('/api/simulate-ball-tracking', {
        slot, sameVideo, model: props.modelSize || 'nano', package: choice[slot] || undefined, threshold: 0.25, language: props.language,
      });
    } catch { setRunning(null); /* surfaced via props.error from worker status */ }
  }

  const sameClip = Boolean(preparedA && preparedB && preparedA.simulation.video === preparedB.simulation.video && preparedA.frames.length === preparedB.frames.length);
  const difference = useMemo(() => {
    if (!sameClip || !preparedA || !preparedB) return null;
    let onlyA = 0, onlyB = 0, both = 0;
    preparedA.frames.forEach((frame, index) => {
      const inA = Boolean(frame.ball), inB = Boolean(preparedB.frames[index]?.ball);
      if (inA && inB) both += 1; else if (inA) onlyA += 1; else if (inB) onlyB += 1;
    });
    return { onlyA, onlyB, both };
  }, [sameClip, preparedA, preparedB]);

  const simulating = Boolean(props.busy && props.operation?.startsWith('simulation'));
  const failed = !props.busy && (props.operation === 'error' || props.operation === 'cancelled');
  const runningSlot = running && !failed && (running.slot === 'a' ? a.id : b.id) === running.previousId ? running.slot : null;
  const maxFps = Math.max(12, ...shown.map((item) => item.fps));
  const lookaheadSeconds = (lookaheadFrames / maxFps).toFixed(1);

  const modelPicker = (slot: Slot) => compare && <label className="ball-tracking-model" data-hint={h.simulationModel}>
    <span>{t.model}</span>
    <select value={choice[slot]} disabled={props.busy} onChange={(event) => setChoice((current) => ({ ...current, [slot]: event.target.value }))}>
      <option value="">{t.activeModel}</option>
      {ballModels.map((model) => <option key={model.packageID} value={model.packageID}>{model.displayName || model.packageID}{model.modelSize ? ` · ${model.modelSize}` : ''}{model.isActive ? ' ●' : ''}</option>)}
    </select>
  </label>;

  return <section className="ball-tracking-workspace">
    <div className="benchmark-heading"><div><span className="benchmark-kicker">◆ SIMULATION</span><h2>{t.title}</h2><p>{t.intro}</p></div>
      <button data-hint={h.compareModels} type="button" className={compare ? 'secondary-button active-view' : 'secondary-button'} onClick={() => setCompare((value) => !value)}>{compare ? t.single : `⇆ ${t.compare}`}</button>
    </div>
    {compare && !ballModels.length && <div className="ball-tracking-note">{t.noBallModels}</div>}
    <div className="ball-tracking-controls">
      {!compare && <button data-hint={h.pickClip} type="button" className="primary-button" onClick={() => void run('a')} disabled={props.busy}>{t.pick}</button>}
      {simulating && <div className="ball-tracking-progress"><div className="benchmark-progress"><span style={{ width: `${Math.max(props.progress, .02) * 100}%` }} /></div><small>{props.message} · {Math.round(props.progress * 100)} %</small></div>}
    </div>
    {props.error && <div className="benchmark-error">{props.error}</div>}
    {gapsSent && !props.busy && !props.error && props.message && <div className="ball-tracking-note">{props.message}</div>}
    {!compare && !preparedA ? <div className="benchmark-empty">{t.empty}</div> : <>
      <div className={compare ? 'ball-tracking-sides' : 'ball-tracking-sides single'}>
        {(compare ? (['a', 'b'] as Slot[]) : (['a'] as Slot[])).map((slot) => {
          const prepared = runningSlot === slot ? null : slot === 'a' ? preparedA : preparedB;
          return <div className="ball-tracking-side" key={slot}>
            {compare && <div className="ball-tracking-side-controls">
              <strong className="ball-tracking-slot">{slot.toUpperCase()}</strong>
              {modelPicker(slot)}
              <button data-hint={h.pickClip} type="button" className="primary-button" onClick={() => void run(slot)} disabled={props.busy}>{t.pick}</button>
              {slot === 'b' && <button data-hint={h.sameVideo} type="button" className="secondary-button" onClick={() => void run('b', true)} disabled={props.busy || !preparedA}>{t.sameVideo}</button>}
            </div>}
            {runningSlot === slot ? <div className="benchmark-empty ball-tracking-side-empty">{simulating ? props.message : '…'}</div> : prepared ? <SimulationStage prepared={prepared} index={indexAt(prepared, time)} api={props.api} slot={slot} /> : <div className="benchmark-empty ball-tracking-side-empty">{t.emptySide}</div>}
            {prepared && <SideSummary prepared={prepared} t={t} />}
            {prepared && <button data-hint={h.simulationGaps} type="button" className="secondary-button" onClick={() => void addGaps(slot)} disabled={props.busy || !prepared.resolved.some((item) => item.state === 'interpolated')}>{t.gaps}</button>}
          </div>;
        })}
      </div>
      {difference && <div className="ball-tracking-difference"><span>{t.sameClip}</span><b>{t.both}: {difference.both}</b><b>{t.onlyA}: {difference.onlyA}</b><b>{t.onlyB}: {difference.onlyB}</b></div>}
      <div className="ball-tracking-legend">
        <span><i style={{ background: stateColor.detected }} />{t.detected}</span>
        <span><i style={{ borderColor: stateColor.interpolated }} />{t.interpolated}</span>
        <span><i style={{ borderColor: stateColor.coasting }} />{t.coasting}</span>
        <span><i className="lost" />{t.lost}</span>
      </div>
      {shown.length > 0 && <div className="ball-tracking-transport">
        <button data-hint={h.playPause} type="button" className="secondary-button" onClick={() => setIsPlaying((value) => !value)}>{isPlaying ? '⏸' : '▶'}</button>
        <input type="range" min={0} max={Math.max(0, duration - step)} step={step} value={Math.min(time, Math.max(0, duration - step))} onChange={(event) => setTime(Number(event.target.value))} />
        <span className="ball-tracking-frame-count">{time.toFixed(1)} / {duration.toFixed(1)} s</span>
      </div>}
      <label className="ball-tracking-lookahead">
        <span>{t.lookahead}</span>
        <input type="range" min={0} max={Math.max(1, Math.round(maxFps * 5))} value={lookaheadFrames} onChange={(event) => setLookaheadFrames(Number(event.target.value))} />
        <strong>{lookaheadSeconds}s ({lookaheadFrames} {t.frames})</strong>
      </label>
    </>}
  </section>;
}

function SimulationStage({ prepared, index, api, slot }: { prepared: Prepared; index: number; api: string; slot: Slot }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 640, height: 360 });
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
  const frame = prepared.frames[index];
  const position = prepared.resolved[index];
  const fitted = frame ? fitRect(frame.width, frame.height, containerSize.width, containerSize.height) : null;
  return <div className="ball-tracking-stage" ref={containerRef}>
    {frame && fitted && <>
      <img src={`${api}/api/simulation-frame?slot=${slot}&file=${encodeURIComponent(frame.file)}&v=${encodeURIComponent(prepared.simulation.id ?? '')}`} alt="" style={{ position: 'absolute', left: fitted.x, top: fitted.y, width: fitted.width, height: fitted.height }} />
      {position?.x != null && position.y != null && position.state !== 'lost' && <span
        className={`ball-marker ball-marker-${position.state}`}
        style={{
          left: fitted.x + (position.x / frame.width) * fitted.width,
          top: fitted.y + (position.y / frame.height) * fitted.height,
          borderColor: stateColor[position.state],
          background: position.state === 'detected' ? stateColor.detected : 'transparent',
        }}
      />}
      <span className="ball-tracking-frame-badge">{index + 1} / {prepared.frames.length}</span>
    </>}
  </div>;
}

function SideSummary({ prepared, t }: { prepared: Prepared; t: (typeof text)[Language] }) {
  const summary = useMemo(() => summarize(prepared), [prepared]);
  const share = (state: TrackState) => percent(summary.total ? summary.counts[state] / summary.total : 0);
  return <div className="ball-tracking-summary">
    <div className="ball-tracking-summary-title"><b>{prepared.simulation.model || '—'}</b>{prepared.simulation.video && <small>{prepared.simulation.video}</small>}</div>
    <div className="ball-tracking-summary-stats">
      <span><b>{percent(summary.tracked)}</b>{t.tracked}</span>
      <span><b>{share('detected')}</b>{t.detected}</span>
      <span><b>{share('interpolated')}</b>{t.interpolated}</span>
      <span><b>{share('coasting')}</b>{t.coasting}</span>
      <span><b>{share('lost')}</b>{t.lost}</span>
      <span><b>{summary.avgConfidence == null ? '—' : percent(summary.avgConfidence)}</b>{t.avgConfidence}</span>
    </div>
  </div>;
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
