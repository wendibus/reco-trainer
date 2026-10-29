import { useEffect, useRef, useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';

export type JobStatus = 'queued' | 'running' | 'cancelling' | 'cancelled' | 'done' | 'failed';
export type Job = {
  id: string; type: string; status: JobStatus; cancellable: boolean;
  progress?: number | null; message?: string; error?: string | null; operation?: string;
  stage?: { index: number; count: number; label: string } | null;
  detail?: { phase: 'probe' | 'extract'; videoIndex: number; videoCount: number; videoName: string; framesDone: number; framesTarget: number; startedAt: number } | null;
  startedAt?: number | null; elapsedSeconds?: number; etaSeconds?: number | null;
  result?: Record<string, unknown> | null;
};
export type Jobs = { current: Job | null; queue: Job[]; recent: Job[] };

type Props = {
  api: string;
  language: Language;
  /** Job list from the regular /api/status poll - used while the event stream is not connected. */
  fallback?: Jobs;
  cancel: (jobId: string) => Promise<unknown>;
  /** Called when a job ends, so the page reloads project data at once. */
  onJobEnded?: (job: Job) => void;
  /** Optional extra content for a finished job (for example "Show folder" after an export). */
  renderResult?: (job: Job, close: () => void) => React.ReactNode;
};

const SHOW_AFTER_SECONDS = 0.8;

const text: Record<Language, {
  titles: Record<string, string>; fallbackTitle: string; step: (index: number, count: number, label: string) => string;
  probe: (index: number, count: number) => string; video: (index: number, count: number) => string; frames: (done: number, target: number) => string;
  elapsed: (value: string) => string; remaining: (value: string) => string; queued: (count: number) => string;
  cancel: string; cancelling: string; confirmCancel: string; confirmCancelTraining: string; minimize: string; restore: string; close: string;
  done: string; failed: string; cancelled: string; working: string;
}> = {
  de: {
    titles: { prepare: 'Videos vorbereiten', 'active-learning': 'Datensatz erweitern', 'independent-validation': 'Unabhängigen Test vorbereiten', 'simulate-ball-tracking': 'Balltracking-Simulation', 'import-model': 'Modell importieren', 'import-onnx': 'ONNX-Modell importieren', 'import-labels': 'Labels importieren', setup: 'ML einrichten', autolabel: 'Automatisch markieren', train: 'Lokal trainieren', 'export-coreml': 'Apple-Modell exportieren', 'export-onnx': 'CPU-Modell exportieren', 'refine-boxes': 'Boxen verfeinern', 'package-model': 'Austauschpaket erstellen', 'combine-models': 'Modelle kombinieren', benchmark: 'Modellvergleich', 'activate-model': 'Modell aktivieren', 'rename-model': 'Modell umbenennen', 'delete-model': 'Modell löschen', 'export-dataset': 'Datensatz exportieren', 'download-models': 'Modelle herunterladen' },
    fallbackTitle: 'Vorgang läuft', step: (index, count, label) => `Schritt ${index}/${count}: ${label}`,
    probe: (index, count) => `Prüfe Video ${index} von ${count}`, video: (index, count) => `Video ${index} von ${count}`, frames: (done, target) => `${done} von ${target} Bildern`,
    elapsed: (value) => `Läuft seit ${value}`, remaining: (value) => `noch etwa ${value}`, queued: (count) => count === 1 ? 'Danach wartet 1 weiterer Vorgang.' : `Danach warten ${count} weitere Vorgänge.`,
    cancel: 'Abbrechen', cancelling: 'Wird abgebrochen …', confirmCancel: 'Diesen Vorgang wirklich abbrechen? Halbfertige Dateien werden entfernt, das Projekt bleibt unverändert.',
    confirmCancelTraining: 'Training wirklich abbrechen? Der zuletzt gespeicherte Checkpoint bleibt erhalten; ein neues Training setzt dort fort.',
    minimize: 'Minimieren', restore: 'Fortschritt anzeigen', close: 'Schließen', done: 'Fertig', failed: 'Fehlgeschlagen', cancelled: 'Abgebrochen', working: 'Bitte warten …',
  },
  en: {
    titles: { prepare: 'Preparing videos', 'active-learning': 'Extending the dataset', 'independent-validation': 'Preparing an independent test', 'simulate-ball-tracking': 'Ball-tracking simulation', 'import-model': 'Importing a model', 'import-onnx': 'Importing an ONNX model', 'import-labels': 'Importing labels', setup: 'Setting up ML', autolabel: 'Auto-labeling', train: 'Training locally', 'export-coreml': 'Exporting the Apple model', 'export-onnx': 'Exporting the CPU model', 'refine-boxes': 'Refining boxes', 'package-model': 'Creating an exchange package', 'combine-models': 'Combining models', benchmark: 'Model comparison', 'activate-model': 'Activating a model', 'rename-model': 'Renaming a model', 'delete-model': 'Deleting a model', 'export-dataset': 'Exporting the dataset', 'download-models': 'Downloading models' },
    fallbackTitle: 'Operation running', step: (index, count, label) => `Step ${index}/${count}: ${label}`,
    probe: (index, count) => `Checking video ${index} of ${count}`, video: (index, count) => `Video ${index} of ${count}`, frames: (done, target) => `${done} of ${target} images`,
    elapsed: (value) => `Running for ${value}`, remaining: (value) => `about ${value} left`, queued: (count) => count === 1 ? '1 more operation is waiting.' : `${count} more operations are waiting.`,
    cancel: 'Cancel', cancelling: 'Cancelling …', confirmCancel: 'Really cancel this operation? Half-written files are removed; the project stays unchanged.',
    confirmCancelTraining: 'Really cancel training? The last saved checkpoint is kept; a new training run continues from there.',
    minimize: 'Minimize', restore: 'Show progress', close: 'Close', done: 'Done', failed: 'Failed', cancelled: 'Cancelled', working: 'Please wait …',
  },
  es: {
    titles: { prepare: 'Preparando vídeos', 'active-learning': 'Ampliando el conjunto de datos', 'independent-validation': 'Preparando una prueba independiente', 'simulate-ball-tracking': 'Simulación de seguimiento del balón', 'import-model': 'Importando un modelo', 'import-onnx': 'Importando un modelo ONNX', 'import-labels': 'Importando etiquetas', setup: 'Configurando ML', autolabel: 'Marcado automático', train: 'Entrenando localmente', 'export-coreml': 'Exportando el modelo Apple', 'export-onnx': 'Exportando el modelo CPU', 'refine-boxes': 'Refinando cuadros', 'package-model': 'Creando un paquete de intercambio', 'combine-models': 'Combinando modelos', benchmark: 'Comparación de modelos', 'activate-model': 'Activando un modelo', 'rename-model': 'Renombrando un modelo', 'delete-model': 'Eliminando un modelo', 'export-dataset': 'Exportando el conjunto de datos', 'download-models': 'Descargando modelos' },
    fallbackTitle: 'Operación en curso', step: (index, count, label) => `Paso ${index}/${count}: ${label}`,
    probe: (index, count) => `Comprobando vídeo ${index} de ${count}`, video: (index, count) => `Vídeo ${index} de ${count}`, frames: (done, target) => `${done} de ${target} imágenes`,
    elapsed: (value) => `En curso desde hace ${value}`, remaining: (value) => `quedan unos ${value}`, queued: (count) => count === 1 ? 'Espera 1 operación más.' : `Esperan ${count} operaciones más.`,
    cancel: 'Cancelar', cancelling: 'Cancelando …', confirmCancel: '¿Cancelar esta operación? Los archivos a medio escribir se eliminan; el proyecto no cambia.',
    confirmCancelTraining: '¿Cancelar el entrenamiento? Se conserva el último checkpoint guardado; un nuevo entrenamiento continúa desde ahí.',
    minimize: 'Minimizar', restore: 'Mostrar progreso', close: 'Cerrar', done: 'Listo', failed: 'Error', cancelled: 'Cancelado', working: 'Espera …',
  },
  fr: {
    titles: { prepare: 'Préparation des vidéos', 'active-learning': 'Extension du jeu de données', 'independent-validation': 'Préparation d’un test indépendant', 'simulate-ball-tracking': 'Simulation de suivi du ballon', 'import-model': 'Import d’un modèle', 'import-onnx': 'Import d’un modèle ONNX', 'import-labels': 'Import des annotations', setup: 'Configuration du ML', autolabel: 'Marquage automatique', train: 'Entraînement local', 'export-coreml': 'Export du modèle Apple', 'export-onnx': 'Export du modèle CPU', 'refine-boxes': 'Affinage des boîtes', 'package-model': 'Création d’un paquet d’échange', 'combine-models': 'Combinaison de modèles', benchmark: 'Comparaison des modèles', 'activate-model': 'Activation d’un modèle', 'rename-model': 'Renommage d’un modèle', 'delete-model': 'Suppression d’un modèle', 'export-dataset': 'Export du jeu de données', 'download-models': 'Téléchargement des modèles' },
    fallbackTitle: 'Opération en cours', step: (index, count, label) => `Étape ${index}/${count} : ${label}`,
    probe: (index, count) => `Vérification de la vidéo ${index} sur ${count}`, video: (index, count) => `Vidéo ${index} sur ${count}`, frames: (done, target) => `${done} images sur ${target}`,
    elapsed: (value) => `En cours depuis ${value}`, remaining: (value) => `encore environ ${value}`, queued: (count) => count === 1 ? '1 autre opération attend.' : `${count} autres opérations attendent.`,
    cancel: 'Annuler', cancelling: 'Annulation …', confirmCancel: 'Vraiment annuler cette opération ? Les fichiers incomplets sont supprimés ; le projet reste inchangé.',
    confirmCancelTraining: 'Vraiment annuler l’entraînement ? Le dernier checkpoint enregistré est conservé ; un nouvel entraînement reprend à partir de là.',
    minimize: 'Réduire', restore: 'Afficher la progression', close: 'Fermer', done: 'Terminé', failed: 'Échec', cancelled: 'Annulé', working: 'Veuillez patienter …',
  },
};

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}` : `${minutes}:${String(rest).padStart(2, '0')}`;
}

/** Progress window for every long operation: live via server-sent events, cancellable, minimizable. */
export function JobDialog({ api, language, fallback, cancel, onJobEnded, renderResult }: Props) {
  const t = text[language];
  const [live, setLive] = useState<Jobs | null>(null);
  const [minimizedJob, setMinimizedJob] = useState<string | null>(null);
  const [cancelRequested, setCancelRequested] = useState<string | null>(null);
  const [shownJobs, setShownJobs] = useState<Set<string>>(new Set());
  const [finishedJob, setFinishedJob] = useState<Job | null>(null);
  const [, setTick] = useState(0);
  const lastCurrent = useRef<string | null>(null);

  useEffect(() => {
    if (typeof EventSource === 'undefined') return undefined;
    const source = new EventSource(`${api}/api/events`);
    source.addEventListener('jobs', (event) => {
      try { setLive(JSON.parse((event as MessageEvent).data) as Jobs); } catch { /* ignore a malformed event */ }
    });
    source.onerror = () => setLive(null); // the browser reconnects by itself; the poll covers the gap
    return () => source.close();
  }, [api]);

  const jobs = live ?? fallback ?? { current: null, queue: [], recent: [] };
  const current = jobs.current;

  useEffect(() => {
    if (!current) return undefined;
    const timer = window.setInterval(() => setTick((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [current?.id]);

  // Notice the end of a job: reload project data, and keep the window open for errors,
  // cancels and results of jobs the user was already watching.
  useEffect(() => {
    const previous = lastCurrent.current;
    lastCurrent.current = current?.id ?? null;
    if (!previous || previous === current?.id) return;
    const ended = jobs.recent.find((job) => job.id === previous);
    if (!ended) return;
    onJobEnded?.(ended);
    setCancelRequested(null);
    if (ended.status === 'failed' || (shownJobs.has(ended.id) && (ended.status === 'cancelled' || ended.result))) setFinishedJob(ended);
  }, [current?.id, jobs.recent]); // eslint-disable-line react-hooks/exhaustive-deps

  // startedAt is the worker's clock (same machine); recomputed every second by the tick above.
  const elapsed = current ? (current.startedAt ? Math.max(Date.now() / 1000 - current.startedAt, current.elapsedSeconds ?? 0) : current.elapsedSeconds ?? 0) : 0;
  const visible = !!current && (elapsed >= SHOW_AFTER_SECONDS || current.status === 'cancelling' || shownJobs.has(current.id));
  useEffect(() => {
    if (visible && current && !shownJobs.has(current.id)) setShownJobs((set) => new Set(set).add(current.id));
  }, [visible, current?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const title = (job: Job) => t.titles[job.type] ?? t.fallbackTitle;

  async function requestCancel(job: Job) {
    if (!window.confirm(job.type === 'train' ? t.confirmCancelTraining : t.confirmCancel)) return;
    setCancelRequested(job.id);
    try { await cancel(job.id); } catch { setCancelRequested(null); }
  }

  if (finishedJob && !current) {
    const statusText = finishedJob.status === 'failed' ? t.failed : finishedJob.status === 'cancelled' ? t.cancelled : t.done;
    return <div className="walkthrough-backdrop" role="presentation"><section className="progress-dialog" role="dialog" aria-modal="true" aria-labelledby="job-title">
      <h2 id="job-title">{title(finishedJob)} · {statusText}</h2>
      {finishedJob.status === 'failed'
        ? <pre className="progress-dialog-error">{finishedJob.error || finishedJob.message}</pre>
        : finishedJob.message && <p className="progress-dialog-message">{finishedJob.message}</p>}
      {renderResult?.(finishedJob, () => setFinishedJob(null))}
      <div className="onnx-actions"><button type="button" className="primary-button" onClick={() => setFinishedJob(null)}>{t.close}</button></div>
    </section></div>;
  }

  if (!current || !visible) return null;

  const progress = typeof current.progress === 'number' ? Math.min(Math.max(current.progress, 0), 1) : null;
  const detail = current.detail;
  const cancelling = current.status === 'cancelling' || cancelRequested === current.id;
  const percent = progress === null ? null : Math.round(progress * 100);
  const remaining = progress !== null && progress > 0.03 && progress < 1 ? elapsed * (1 - progress) / progress : null;

  if (minimizedJob === current.id) {
    return <button type="button" className="job-chip" onClick={() => setMinimizedJob(null)} title={t.restore}>
      <span className={`job-chip-bar${progress === null ? ' indeterminate' : ''}`}><span style={progress === null ? undefined : { width: `${Math.max(progress, 0.02) * 100}%` }} /></span>
      {title(current)}{percent !== null ? ` · ${percent} %` : ''}
    </button>;
  }

  return <div className="walkthrough-backdrop" role="presentation"><section className="progress-dialog" role="dialog" aria-modal="true" aria-labelledby="job-title">
    <div className="progress-dialog-head"><h2 id="job-title">{title(current)}</h2><button type="button" className="tool-button" onClick={() => setMinimizedJob(current.id)} title={t.minimize} aria-label={t.minimize}>–</button></div>
    {current.stage && <p className="progress-dialog-stage">{t.step(current.stage.index, current.stage.count, current.stage.label)}</p>}
    {detail
      ? <p className="progress-dialog-video"><strong>{detail.phase === 'probe' ? t.probe(detail.videoIndex, detail.videoCount) : t.video(detail.videoIndex, detail.videoCount)}</strong><span>{detail.videoName}</span></p>
      : <p className="progress-dialog-message">{cancelling ? t.cancelling : current.message || t.working}</p>}
    <div className={`progress-dialog-bar${progress === null ? ' indeterminate' : ''}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined}>
      <span style={progress === null ? undefined : { width: `${Math.max(progress, 0.02) * 100}%` }} />
    </div>
    <div className="progress-dialog-facts">
      {percent !== null && <span>{percent} %</span>}
      {detail?.phase === 'extract' && <span>{t.frames(detail.framesDone, detail.framesTarget)}</span>}
      <span>{t.elapsed(formatDuration(elapsed))}{remaining !== null ? ` · ${t.remaining(formatDuration(remaining))}` : ''}</span>
    </div>
    {jobs.queue.length > 0 && <small>{t.queued(jobs.queue.length)}</small>}
    <div className="onnx-actions">
      {current.cancellable && <button type="button" className="secondary-button" disabled={cancelling} onClick={() => void requestCancel(current)}>{cancelling ? t.cancelling : t.cancel}</button>}
    </div>
  </section></div>;
}
