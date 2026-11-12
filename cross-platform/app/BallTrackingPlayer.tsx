import { useEffect, useMemo, useRef, useState } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type SimulationFrame = { file: string; timestamp: number; width: number; height: number; ball: { x: number; y: number; confidence: number; width?: number; height?: number } | null };
type SceneKind = 'gap' | 'jump' | 'weak' | 'lost';
/// A stretch of the clip worth a look (teachers.tracking_scenes in the worker); start/end are frame indices.
type Scene = { id: string; kind: SceneKind; start: number; end: number; startTime: number; endTime: number; picks: Array<{ index: number }> };
type Simulation = { id?: string; fps: number; every?: number; scenesTotal?: number; start?: number; length?: number; frames: SimulationFrame[]; slot?: 'a' | 'b'; video?: string; model?: string; package?: string | null; threshold?: number; scenes?: Scene[]; partial?: { done: number; total: number } };
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
  stage?: { index: number; count: number; label: string } | null;
  simulation?: Simulation | null;
  simulationB?: Simulation | null;
  models?: LibraryModel[];
  modelSize?: string;
  sport?: string;
  /** The open project; start, length and density are remembered per project. */
  projectKey?: string;
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const text = {
  de: { title: 'Balltracking-Simulation', intro: 'Video wählen und die Ballerkennung Bild für Bild abspielen. Bei langen Videos wertest du einen Abschnitt aus (Start und Länge) und wählst, ob alle 12 Bilder pro Sekunde oder nur jedes n-te Bild gezeigt werden. Im Vergleich laufen zwei Videos mit zwei Modellen synchron nebeneinander. Videos werden nur lokal verarbeitet. Bilder kommen nur ins Projekt, wenn du Szenen als Prüfkandidaten übernimmst.', pick: 'Video auswählen', start: 'Start (Sekunden)', every: 'Bilder', everyAll: 'alle', everyNth: 'jedes {n}.', perSecond: '{r} pro Sekunde', sparseNote: 'Bei jedem 4. Bild oder weniger werden Lücken, Sprünge und verlorene Stücke nur grob erkannt: Ein schneller Ball kann zwischen zwei Bildern verschwinden.', moreScenes: '{n} weitere Szenen werden nicht angezeigt.', length: 'Länge', imagesAbout: 'ergibt bis zu {n} Bilder', empty: 'Wähle oben ein Video aus, um zu starten.', emptySide: 'Noch keine Simulation auf dieser Seite.', lookahead: 'Wie weit direkt in die Zukunft schauen', frames: 'Bilder', detected: 'Erkannt', interpolated: 'Interpoliert', coasting: 'Gehalten', lost: 'Verloren', compare: 'Zwei Modelle vergleichen', single: 'Nur eine Simulation', model: 'Modell', activeModel: 'Aktives Modell', sameVideo: 'Gleiches Video wie A', tracked: 'Ball verfolgt', avgConfidence: 'Ø Sicherheit', onlyA: 'Nur A erkennt', onlyB: 'Nur B erkennt', both: 'Beide erkennen', sameClip: 'Gleiches Video: Unterschiede Bild für Bild', noBallModels: 'Keine Modelle mit Ball-Klasse in der Bibliothek; es läuft das aktive Modell.', gaps: 'Lücken als Prüfkandidaten übernehmen', cancel: 'Abbrechen', cancelTitle: 'Simulation abbrechen?', cancelBody: 'Die Bilder, in denen der Ball schon gesucht wurde, kannst du als kürzere Simulation behalten. Oder du löschst alle Bilder dieses Laufs.', cancelBodyEarly: 'Bisher ist noch kein Bild ausgewertet. Beim Abbruch werden die Bilder dieses Laufs gelöscht.', cancelKeep: 'Abbrechen, Ausgewertetes behalten', cancelDelete: 'Abbrechen und Bilder löschen', cancelContinue: 'Weiter simulieren', stepOf: 'Schritt {i} von {n}', partial: 'Abgebrochener Lauf: {done} von {total} Bildern ausgewertet.', scenes: 'Auffällige Szenen', scenesIntro: 'Ein Klick springt zur Szene. Angekreuzte Szenen kommen mit wenigen Bildern als Prüfkandidaten in die Bilderliste; dort prüfst du jedes Bild einzeln.', noScenes: 'Keine auffälligen Szenen gefunden.', kindGap: 'Lücke', kindJump: 'Sprung', kindWeak: 'Unsicher', kindLost: 'Verloren', images: 'Bilder', adopt: '{scenes} Szenen übernehmen ({images} Bilder)', prevScene: '◀ Szene', nextScene: 'Szene ▶' },
  en: { title: 'Ball-tracking simulation', intro: 'Pick a video and play back the ball detection frame by frame. For a long video you analyse one section (start and length) and choose whether to use all 12 images per second or only every n-th. In comparison mode, two videos with two models play in sync side by side. Videos are only processed locally. Images only enter the project when you add scenes as review candidates.', pick: 'Select video', start: 'Start (seconds)', every: 'Images', everyAll: 'all', everyNth: 'one in {n}', perSecond: '{r} per second', sparseNote: 'With every 4th image or fewer, gaps, jumps and lost stretches are only seen roughly: a fast ball can vanish between two images.', moreScenes: '{n} more scenes are not shown.', length: 'Length', imagesAbout: 'gives up to {n} images', empty: 'Select a video above to get started.', emptySide: 'No simulation on this side yet.', lookahead: 'How far to look directly into the future', frames: 'frames', detected: 'Detected', interpolated: 'Interpolated', coasting: 'Held', lost: 'Lost', compare: 'Compare two models', single: 'Single simulation', model: 'Model', activeModel: 'Active model', sameVideo: 'Same video as A', tracked: 'Ball tracked', avgConfidence: 'Avg. confidence', onlyA: 'Only A detects', onlyB: 'Only B detects', both: 'Both detect', sameClip: 'Same video: frame-by-frame differences', noBallModels: 'No models with a ball class in the library; the active model is used.', gaps: 'Add gaps as review candidates', cancel: 'Cancel', cancelTitle: 'Cancel the simulation?', cancelBody: 'You can keep the images in which the ball has already been looked for, as a shorter simulation. Or delete all images of this run.', cancelBodyEarly: 'No image has been analysed yet. Cancelling deletes the images of this run.', cancelKeep: 'Cancel, keep what is analysed', cancelDelete: 'Cancel and delete the images', cancelContinue: 'Keep simulating', stepOf: 'Step {i} of {n}', partial: 'Cancelled run: {done} of {total} images analysed.', scenes: 'Notable scenes', scenesIntro: 'A click jumps to the scene. Ticked scenes go into the image list as review candidates, a few images each; there you check every image yourself.', noScenes: 'No notable scenes found.', kindGap: 'Gap', kindJump: 'Jump', kindWeak: 'Unsure', kindLost: 'Lost', images: 'images', adopt: 'Add {scenes} scenes ({images} images)', prevScene: '◀ Scene', nextScene: 'Scene ▶' },
  es: { title: 'Simulación de seguimiento del balón', intro: 'Elige un vídeo y reproduce la detección del balón cuadro a cuadro. En un vídeo largo analizas un tramo (inicio y duración) y eliges si usar las 12 imágenes por segundo o solo una de cada n. En la comparación, dos vídeos con dos modelos se reproducen sincronizados uno al lado del otro. Los vídeos solo se procesan localmente. Las imágenes solo entran en el proyecto cuando añades escenas como candidatas.', pick: 'Seleccionar vídeo', start: 'Inicio (segundos)', every: 'Imágenes', everyAll: 'todas', everyNth: 'una de cada {n}', perSecond: '{r} por segundo', sparseNote: 'Con una de cada 4 imágenes o menos, los huecos, saltos y tramos perdidos solo se ven de forma aproximada: un balón rápido puede desaparecer entre dos imágenes.', moreScenes: 'No se muestran {n} escenas más.', length: 'Duración', imagesAbout: 'da hasta {n} imágenes', empty: 'Selecciona un vídeo arriba para empezar.', emptySide: 'Aún no hay simulación en este lado.', lookahead: 'Hasta dónde mirar directamente al futuro', frames: 'imágenes', detected: 'Detectado', interpolated: 'Interpolado', coasting: 'Mantenido', lost: 'Perdido', compare: 'Comparar dos modelos', single: 'Una sola simulación', model: 'Modelo', activeModel: 'Modelo activo', sameVideo: 'Mismo vídeo que A', tracked: 'Balón seguido', avgConfidence: 'Confianza media', onlyA: 'Solo A detecta', onlyB: 'Solo B detecta', both: 'Ambos detectan', sameClip: 'Mismo vídeo: diferencias imagen por imagen', noBallModels: 'No hay modelos con clase de balón en la biblioteca; se usa el modelo activo.', gaps: 'Añadir huecos como candidatos', cancel: 'Cancelar', cancelTitle: '¿Cancelar la simulación?', cancelBody: 'Puedes conservar las imágenes en las que ya se buscó el balón, como una simulación más corta. O eliminar todas las imágenes de esta ejecución.', cancelBodyEarly: 'Aún no se ha analizado ninguna imagen. Al cancelar se eliminan las imágenes de esta ejecución.', cancelKeep: 'Cancelar y conservar lo analizado', cancelDelete: 'Cancelar y eliminar las imágenes', cancelContinue: 'Seguir simulando', stepOf: 'Paso {i} de {n}', partial: 'Ejecución cancelada: {done} de {total} imágenes analizadas.', scenes: 'Escenas llamativas', scenesIntro: 'Un clic salta a la escena. Las escenas marcadas pasan a la lista de imágenes como candidatas, con pocas imágenes cada una; allí revisas cada imagen.', noScenes: 'No se encontraron escenas llamativas.', kindGap: 'Hueco', kindJump: 'Salto', kindWeak: 'Dudoso', kindLost: 'Perdido', images: 'imágenes', adopt: 'Añadir {scenes} escenas ({images} imágenes)', prevScene: '◀ Escena', nextScene: 'Escena ▶' },
  fr: { title: 'Simulation de suivi du ballon', intro: 'Choisissez une vidéo et regardez la détection du ballon image par image. Pour une vidéo longue, vous analysez un passage (début et durée) et choisissez toutes les 12 images par seconde ou seulement une sur n. En mode comparaison, deux vidéos avec deux modèles sont lues côte à côte de façon synchronisée. Les vidéos ne sont traitées que localement. Les images n’entrent dans le projet que si vous ajoutez des scènes comme candidates.', pick: 'Sélectionner une vidéo', start: 'Début (secondes)', every: 'Images', everyAll: 'toutes', everyNth: 'une sur {n}', perSecond: '{r} par seconde', sparseNote: 'À partir d’une image sur 4, les lacunes, sauts et passages perdus ne sont vus que grossièrement : un ballon rapide peut disparaître entre deux images.', moreScenes: '{n} autres scènes ne sont pas affichées.', length: 'Durée', imagesAbout: 'donne jusqu’à {n} images', empty: 'Sélectionnez une vidéo ci-dessus pour commencer.', emptySide: 'Pas encore de simulation de ce côté.', lookahead: 'Jusqu’où regarder directement dans le futur', frames: 'images', detected: 'Détecté', interpolated: 'Interpolé', coasting: 'Maintenu', lost: 'Perdu', compare: 'Comparer deux modèles', single: 'Une seule simulation', model: 'Modèle', activeModel: 'Modèle actif', sameVideo: 'Même vidéo que A', tracked: 'Ballon suivi', avgConfidence: 'Confiance moy.', onlyA: 'Seul A détecte', onlyB: 'Seul B détecte', both: 'Les deux détectent', sameClip: 'Même vidéo : différences image par image', noBallModels: 'Aucun modèle avec une classe ballon dans la bibliothèque ; le modèle actif est utilisé.', gaps: 'Ajouter les lacunes comme candidates', cancel: 'Annuler', cancelTitle: 'Annuler la simulation ?', cancelBody: 'Vous pouvez garder les images dans lesquelles le ballon a déjà été cherché, comme simulation plus courte. Ou supprimer toutes les images de cette exécution.', cancelBodyEarly: 'Aucune image n’est encore analysée. L’annulation supprime les images de cette exécution.', cancelKeep: 'Annuler et garder l’analyse', cancelDelete: 'Annuler et supprimer les images', cancelContinue: 'Continuer la simulation', stepOf: 'Étape {i} sur {n}', partial: 'Exécution annulée : {done} images analysées sur {total}.', scenes: 'Scènes notables', scenesIntro: 'Un clic saute à la scène. Les scènes cochées entrent dans la liste d’images comme candidates, avec quelques images chacune ; vous y vérifiez chaque image.', noScenes: 'Aucune scène notable trouvée.', kindGap: 'Lacune', kindJump: 'Saut', kindWeak: 'Incertain', kindLost: 'Perdu', images: 'images', adopt: 'Ajouter {scenes} scènes ({images} images)', prevScene: '◀ Scène', nextScene: 'Scène ▶' },
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

const sceneColor: Record<SceneKind, string> = { gap: '#e0902f', jump: '#d9534f', weak: '#c9a700', lost: '#8e8e93' };
/// Gaps and jumps are the model's clear mistakes, so they start ticked; weak and lost stretches are offered.
const defaultPicked = (simulation: Simulation | null) => new Set((simulation?.scenes ?? []).filter((scene) => scene.kind === 'gap' || scene.kind === 'jump').map((scene) => scene.id));
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

// The look-ahead is set in seconds, so it means the same at every density of images.
function prepare(simulation: Simulation | null, lookaheadSeconds: number): Prepared | null {
  if (!simulation || !simulation.frames.length) return null;
  const fps = simulation.fps || 12;
  const detections = simulation.frames.map((frame) => frame.ball ? { x: frame.ball.x, y: frame.ball.y } : null);
  return { simulation, frames: simulation.frames, fps, duration: simulation.frames.length / fps, resolved: resolve(detections, Math.round(lookaheadSeconds * fps)) };
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

const SECTION_STORAGE = 'reco-simulation-section:';
type SavedSection = { start: string; length: number; every: number };

/** Start, length and density of the last run in this project (per viewer; may be unavailable). */
function loadSection(projectKey: string | undefined): SavedSection | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(SECTION_STORAGE + (projectKey ?? '')) ?? 'null');
    if (value && typeof value.start === 'string' && Number.isFinite(value.length) && Number.isFinite(value.every)) return { start: value.start, length: Number(value.length), every: Number(value.every) };
  } catch { /* storage blocked or damaged: use the defaults */ }
  return null;
}

const percent = (value: number) => `${Math.round(value * 100)} %`;

export default function BallTrackingPlayer(props: Props) {
  const t = text[props.language];
  const h = hintText[props.language];
  const [lookaheadSeconds, setLookaheadSeconds] = useState(0.7);
  // The section of the video the simulation looks at: 12 images per second from `start` for `length` seconds.
  const [start, setStart] = useState(() => loadSection(props.projectKey)?.start ?? '0');
  const [length, setLength] = useState(() => loadSection(props.projectKey)?.length ?? 25);
  const [every, setEvery] = useState(() => loadSection(props.projectKey)?.every ?? 1);
  const sectionProject = useRef(props.projectKey);
  useEffect(() => {
    if (sectionProject.current === props.projectKey) {
      try { window.localStorage.setItem(SECTION_STORAGE + (props.projectKey ?? ''), JSON.stringify({ start, length, every })); } catch { /* not remembered */ }
      return;
    }
    // Another project was opened: take over what was used there last.
    sectionProject.current = props.projectKey;
    const saved = loadSection(props.projectKey);
    setStart(saved?.start ?? '0'); setLength(saved?.length ?? 25); setEvery(saved?.every ?? 1);
  }, [props.projectKey, start, length, every]);
  // One run holds at most 2,160 images (180 s at 12 per second), so a sparser run may be longer.
  const maxLength = 180 * every;
  const sectionLength = Math.min(length, maxLength);
  const lengthChoices = [25, 60, 120, 180, 360, 720, 1080, 2160].filter((seconds) => seconds <= maxLength);
  const startSeconds = Math.max(0, Number(start.replace(',', '.')) || 0);
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const a = useStableSimulation(props.simulation);
  const b = useStableSimulation(props.simulationB);
  const [compare, setCompare] = useState(Boolean(props.simulationB));
  const [choice, setChoice] = useState<Record<Slot, string>>({ a: '', b: '' });
  // The worker deletes a slot's old frames before extracting new ones, so that slot's
  // player is hidden while it is being re-simulated instead of requesting missing images.
  // Hidden until that slot reports a new simulation id, or the run fails.
  const [running, setRunning] = useState<{ slot: Slot } | null>(null);
  const [cancelOpen, setCancelOpen] = useState(false);
  // Ticked scenes per side; they belong to one simulation and start over with the next one.
  const [picked, setPicked] = useState<Record<Slot, { id: string; scenes: Set<string> }>>({ a: { id: '', scenes: new Set() }, b: { id: '', scenes: new Set() } });
  // Set once gaps were sent to the project, so the worker's result line stays visible afterwards.
  const [gapsSent, setGapsSent] = useState(false);
  const ballClass = props.sport === 'hockey' ? 'puck' : 'ball';
  const ballModels = useMemo(() => (props.models ?? []).filter((model) => model.classes?.includes(ballClass) && (!props.sport || !model.sport || model.sport === props.sport)), [props.models, props.sport, ballClass]);

  const preparedA = useMemo(() => prepare(a.simulation, lookaheadSeconds), [a.simulation, lookaheadSeconds]);
  const preparedB = useMemo(() => compare ? prepare(b.simulation, lookaheadSeconds) : null, [compare, b.simulation, lookaheadSeconds]);
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

  const pickedFor = (slot: Slot) => {
    const stable = slot === 'a' ? a : b;
    return picked[slot].id === stable.id ? picked[slot].scenes : defaultPicked(stable.simulation);
  };
  function toggleScene(slot: Slot, sceneId: string) {
    const next = new Set(pickedFor(slot));
    if (next.has(sceneId)) next.delete(sceneId); else next.add(sceneId);
    setPicked((current) => ({ ...current, [slot]: { id: (slot === 'a' ? a : b).id, scenes: next } }));
  }
  function jumpTo(prepared: Prepared, scene: Scene) {
    setIsPlaying(false);
    setTime((scene.start + 0.01) / prepared.fps);
  }
  async function addScenes(slot: Slot, scenes: string[]) {
    setGapsSent(true);
    try { await props.post('/api/simulation-gaps', { slot, scenes, model: props.modelSize || 'nano', language: props.language }); } catch { /* surfaced via props.error from worker status */ }
  }
  async function cancelRun(keep: boolean) {
    setCancelOpen(false);
    try { await props.post('/api/simulation-cancel', { keep }); } catch { /* surfaced via props.error from worker status */ }
  }

  async function run(slot: Slot, sameVideo = false) {
    setGapsSent(false);
    setRunning({ slot });
    try {
      await props.post('/api/simulate-ball-tracking', {
        slot, sameVideo, start: startSeconds, length: sectionLength, every, model: props.modelSize || 'nano', package: choice[slot] || undefined, threshold: 0.25, language: props.language,
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
  const runningSlot = simulating && running ? running.slot : null;
  // Only frames the detector has finished can be kept; that starts with step 3.
  const canKeep = (props.stage?.index ?? 0) >= 3;
  const lookaheadFrames = Math.round(lookaheadSeconds * (shown[0]?.fps || 12));

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
      <label className="ball-tracking-section" data-hint={h.simulationStart}><span>{t.start}</span>
        <input type="number" min={0} step={1} value={start} disabled={props.busy} onChange={(event) => setStart(event.target.value)} />
      </label>
      <label className="ball-tracking-section" data-hint={h.simulationLength}><span>{t.length}</span>
        <select value={sectionLength} disabled={props.busy} onChange={(event) => setLength(Number(event.target.value))}>
          {lengthChoices.map((seconds) => <option key={seconds} value={seconds}>{seconds} s</option>)}
        </select>
      </label>
      <label className="ball-tracking-section" data-hint={h.simulationEvery}><span>{t.every}</span>
        <select value={every} disabled={props.busy} onChange={(event) => setEvery(Number(event.target.value))}>
          {[1, 2, 3, 4, 6, 12].map((n) => <option key={n} value={n}>{n === 1 ? t.everyAll : t.everyNth.replace('{n}', String(n))} · {t.perSecond.replace('{r}', String(12 / n))}</option>)}
        </select>
      </label>
      {every >= 4 && <small className="ball-tracking-section-note ball-tracking-sparse">{t.sparseNote}</small>}
      <small className="ball-tracking-section-note">{t.imagesAbout.replace('{n}', String(Math.round(sectionLength * 12 / every)))}</small>
      {!compare && <button data-hint={h.pickClip} type="button" className="primary-button" onClick={() => void run('a')} disabled={props.busy}>{t.pick}</button>}
      {simulating && <div className="ball-tracking-progress">
        {props.stage && <b className="ball-tracking-step">{t.stepOf.replace('{i}', String(props.stage.index)).replace('{n}', String(props.stage.count))} · {props.stage.label}</b>}
        <div className="benchmark-progress"><span style={{ width: `${Math.max(props.progress, .02) * 100}%` }} /></div>
        <small>{props.message} · {Math.round(props.progress * 100)} %</small>
      </div>}
      {simulating && <button data-hint={h.simulationCancel} type="button" className="secondary-button" onClick={() => setCancelOpen(true)}>{t.cancel}</button>}
    </div>
    {cancelOpen && simulating && <div className="walkthrough-backdrop" role="presentation">
      <section className="onnx-dialog ball-tracking-cancel" role="alertdialog" aria-modal="true" aria-labelledby="simulation-cancel-title">
        <h2 id="simulation-cancel-title">{t.cancelTitle}</h2>
        <p>{canKeep ? t.cancelBody : t.cancelBodyEarly}</p>
        <div className="ball-tracking-cancel-actions">
          <button data-hint={h.simulationCancelContinue} type="button" className="secondary-button" onClick={() => setCancelOpen(false)}>{t.cancelContinue}</button>
          <button data-hint={h.simulationCancelDelete} type="button" className="secondary-button" onClick={() => void cancelRun(false)}>{t.cancelDelete}</button>
          {canKeep && <button data-hint={h.simulationCancelKeep} type="button" className="primary-button" onClick={() => void cancelRun(true)}>{t.cancelKeep}</button>}
        </div>
      </section>
    </div>}
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
            {prepared && <SceneList prepared={prepared} index={indexAt(prepared, time)} picked={pickedFor(slot)} busy={props.busy} t={t} h={h} onJump={(scene) => jumpTo(prepared, scene)} onToggle={(sceneId) => toggleScene(slot, sceneId)} onAdopt={(scenes) => void addScenes(slot, scenes)} />}
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
        <input type="range" min={0} max={5} step={0.1} value={lookaheadSeconds} onChange={(event) => setLookaheadSeconds(Number(event.target.value))} />
        <strong>{lookaheadSeconds.toFixed(1)} s ({lookaheadFrames} {t.frames})</strong>
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
    {prepared.simulation.partial && <div className="ball-tracking-note">{t.partial.replace('{done}', String(prepared.simulation.partial.done)).replace('{total}', String(prepared.simulation.partial.total))}</div>}
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

/// The scenes of one simulation: a strip along the clip, a list to jump from, and the choice of
/// which scenes go into the project as review candidates.
function SceneList(props: { prepared: Prepared; index: number; picked: Set<string>; busy: boolean; t: (typeof text)[Language]; h: (typeof hintText)[Language]; onJump: (scene: Scene) => void; onToggle: (sceneId: string) => void; onAdopt: (scenes: string[]) => void }) {
  const { prepared, index, picked, t, h } = props;
  const scenes = prepared.simulation.scenes ?? [];
  const total = Math.max(1, prepared.frames.length);
  const kindName: Record<SceneKind, string> = { gap: t.kindGap, jump: t.kindJump, weak: t.kindWeak, lost: t.kindLost };
  const kindHint: Record<SceneKind, string> = { gap: h.sceneGap, jump: h.sceneJump, weak: h.sceneWeak, lost: h.sceneLost };
  const chosen = scenes.filter((scene) => picked.has(scene.id));
  const images = chosen.reduce((sum, scene) => sum + scene.picks.length, 0);
  const previous = [...scenes].reverse().find((scene) => scene.end < index);
  const next = scenes.find((scene) => scene.start > index);
  if (!scenes.length) return <div className="ball-tracking-scenes"><strong>{t.scenes}</strong><p>{t.noScenes}</p></div>;
  return <div className="ball-tracking-scenes">
    <div className="ball-tracking-scenes-head">
      <strong>{t.scenes} · {scenes.length}</strong>
      <button data-hint={h.scenePrevious} type="button" className="secondary-button" disabled={!previous} onClick={() => previous && props.onJump(previous)}>{t.prevScene}</button>
      <button data-hint={h.sceneNext} type="button" className="secondary-button" disabled={!next} onClick={() => next && props.onJump(next)}>{t.nextScene}</button>
    </div>
    <div className="ball-tracking-scene-strip">
      {scenes.map((scene) => <button data-hint={kindHint[scene.kind]} type="button" key={scene.id} aria-label={`${kindName[scene.kind]} ${scene.startTime.toFixed(1)} s`} className={picked.has(scene.id) ? 'picked' : ''} style={{ left: `${(scene.start / total) * 100}%`, width: `max(5px, ${((scene.end - scene.start + 1) / total) * 100}%)`, background: sceneColor[scene.kind] }} onClick={() => props.onJump(scene)} />)}
      <i style={{ left: `${((index + .5) / total) * 100}%` }} />
    </div>
    <p>{t.scenesIntro}</p>
    {(prepared.simulation.scenesTotal ?? 0) > scenes.length && <p className="split-problem">{t.moreScenes.replace('{n}', String((prepared.simulation.scenesTotal ?? 0) - scenes.length))}</p>}
    <div className="ball-tracking-scene-list">
      {scenes.map((scene) => <div key={scene.id} className={index >= scene.start && index <= scene.end ? 'ball-tracking-scene current' : 'ball-tracking-scene'}>
        <input type="checkbox" aria-label={`${kindName[scene.kind]} ${scene.startTime.toFixed(1)} s`} checked={picked.has(scene.id)} disabled={props.busy} onChange={() => props.onToggle(scene.id)} />
        <button data-hint={kindHint[scene.kind]} type="button" onClick={() => props.onJump(scene)}>
          <i style={{ background: sceneColor[scene.kind] }} /><b>{kindName[scene.kind]}</b>
          <span>{scene.startTime.toFixed(1)}–{scene.endTime.toFixed(1)} s</span>
          <small>{scene.picks.length} {t.images}</small>
        </button>
      </div>)}
    </div>
    <button data-hint={h.simulationGaps} type="button" className="primary-button" disabled={props.busy || !images} onClick={() => props.onAdopt(chosen.map((scene) => scene.id))}>{t.adopt.replace('{scenes}', String(chosen.length)).replace('{images}', String(images))}</button>
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
