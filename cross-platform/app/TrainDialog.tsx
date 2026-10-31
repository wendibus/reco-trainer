import { useEffect, useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
export type TrainSize = 'nano' | 'small' | 'medium' | 'large';
const SIZES: TrainSize[] = ['nano', 'small', 'medium', 'large'];
const SIZE_NAMES: Record<TrainSize, string> = { nano: 'Nano', small: 'Small', medium: 'Medium', large: 'Large' };

type PerClass = { threshold: number; f1: number; precision: number; recall: number; groundTruth: number };
type BatchEntry = {
  model: TrainSize; status: 'waiting' | 'running' | 'done' | 'failed' | 'skipped'; error?: string; sweepError?: string; minutes?: number; epochs?: number | null;
  mAP50?: number; bestThreshold?: number; f1?: number; imagesPerSecond?: number; validationImages?: number; perClass?: Record<string, PerClass>; testMetrics?: Record<string, number>;
};
type BatchReport = { startedAt?: string; finishedAt?: string | null; autoEpochs?: boolean; models?: BatchEntry[]; recommended?: TrainSize | null };
export type TrainRequest = { models: TrainSize[]; autoEpochs: boolean; patience: number; epochs: number; freshStart: boolean; sweep: boolean };

const TEXT = {
  de: {
    title: 'Lokal trainieren', which: 'Welche Modelle?', current: 'aktuell', epochMode: 'Wie lange?', fixed: 'Feste Epochenzahl', auto: 'Automatisch, bis keine Verbesserung mehr kommt', patience: 'Stoppen nach Epochen ohne Verbesserung', fresh: 'Von vorn beginnen (für einen fairen Vergleich)', sweep: 'Danach beste Confidence je Modell und Klasse ermitteln', cancel: 'Abbrechen', start: 'Training starten', largeNote: 'Medium und Large sind langsam und brauchen viel Speicher. Ohne starke GPU kaum sinnvoll.', night: 'Der Computer bleibt während des Trainings wach. Netzteil anschließen und beim Mac den Deckel offen lassen.', order: 'Reihenfolge: kleinstes Modell zuerst. Scheitert eines, geht es mit dem nächsten weiter.',
    report: 'Letztes Batch-Training', model: 'Modell', score: 'mAP50 (Validierung)', conf: 'Beste Confidence', perClass: 'Je Klasse', speed: 'Bilder/s', epochs: 'Epochen', time: 'Dauer', test: 'Test mAP', statusNames: { waiting: 'wartet', running: 'abgebrochen oder läuft', done: 'fertig', failed: 'fehlgeschlagen', skipped: 'übersprungen' }, best: 'Empfehlung', apply: 'Confidence übernehmen', basis: (count: number) => `Gemessen auf ${count} Validierungsbildern. Bei gleich guten Modellen wird das kleinere empfohlen.`, smallSet: 'Wenige Validierungsbilder: Kleine Unterschiede sind Zufall.', hints: { cancel: 'Schließt den Dialog ohne zu trainieren.', start: 'Startet das Training mit diesen Einstellungen.', apply: 'Stellt den Regler „Confidence“ auf den besten Wert dieses Modells.' },
  },
  en: {
    title: 'Train locally', which: 'Which models?', current: 'current', epochMode: 'How long?', fixed: 'Fixed number of epochs', auto: 'Automatic, until there is no more improvement', patience: 'Stop after epochs without improvement', fresh: 'Start from scratch (for a fair comparison)', sweep: 'Afterwards find the best confidence per model and class', cancel: 'Cancel', start: 'Start training', largeNote: 'Medium and Large are slow and need a lot of memory. Hardly useful without a strong GPU.', night: 'The computer stays awake during training. Connect the power supply and keep the Mac’s lid open.', order: 'Order: smallest model first. If one fails, the next one continues.',
    report: 'Last batch training', model: 'Model', score: 'mAP50 (validation)', conf: 'Best confidence', perClass: 'Per class', speed: 'Images/s', epochs: 'Epochs', time: 'Duration', test: 'Test mAP', statusNames: { waiting: 'waiting', running: 'cancelled or running', done: 'done', failed: 'failed', skipped: 'skipped' }, best: 'Recommended', apply: 'Use this confidence', basis: (count: number) => `Measured on ${count} validation images. Between equally good models, the smaller one is recommended.`, smallSet: 'Few validation images: small differences are chance.', hints: { cancel: 'Closes the dialog without training.', start: 'Starts training with these settings.', apply: 'Sets the confidence slider to this model’s best value.' },
  },
  es: {
    title: 'Entrenar localmente', which: '¿Qué modelos?', current: 'actual', epochMode: '¿Cuánto tiempo?', fixed: 'Número fijo de épocas', auto: 'Automático, hasta que no haya más mejora', patience: 'Parar tras épocas sin mejora', fresh: 'Empezar desde cero (para una comparación justa)', sweep: 'Después, buscar la mejor confianza por modelo y clase', cancel: 'Cancelar', start: 'Iniciar entrenamiento', largeNote: 'Medium y Large son lentos y necesitan mucha memoria. Poco útiles sin una GPU potente.', night: 'El equipo permanece activo durante el entrenamiento. Conecta el cargador y deja abierta la tapa del Mac.', order: 'Orden: primero el modelo más pequeño. Si uno falla, continúa el siguiente.',
    report: 'Último entrenamiento por lotes', model: 'Modelo', score: 'mAP50 (validación)', conf: 'Mejor confianza', perClass: 'Por clase', speed: 'Imágenes/s', epochs: 'Épocas', time: 'Duración', test: 'mAP de prueba', statusNames: { waiting: 'en espera', running: 'cancelado o en curso', done: 'terminado', failed: 'fallido', skipped: 'omitido' }, best: 'Recomendado', apply: 'Usar esta confianza', basis: (count: number) => `Medido en ${count} imágenes de validación. Entre modelos igual de buenos se recomienda el más pequeño.`, smallSet: 'Pocas imágenes de validación: las diferencias pequeñas son casualidad.', hints: { cancel: 'Cierra el diálogo sin entrenar.', start: 'Inicia el entrenamiento con estos ajustes.', apply: 'Pone el control de confianza en el mejor valor de este modelo.' },
  },
  fr: {
    title: 'Entraîner localement', which: 'Quels modèles ?', current: 'actuel', epochMode: 'Combien de temps ?', fixed: 'Nombre fixe d’époques', auto: 'Automatique, jusqu’à ce qu’il n’y ait plus d’amélioration', patience: 'Arrêter après des époques sans amélioration', fresh: 'Repartir de zéro (pour une comparaison équitable)', sweep: 'Ensuite, trouver la meilleure confiance par modèle et par classe', cancel: 'Annuler', start: 'Lancer l’entraînement', largeNote: 'Medium et Large sont lents et demandent beaucoup de mémoire. Peu utiles sans GPU puissant.', night: 'L’ordinateur reste actif pendant l’entraînement. Branchez l’alimentation et laissez le capot du Mac ouvert.', order: 'Ordre : le plus petit modèle d’abord. Si l’un échoue, le suivant continue.',
    report: 'Dernier entraînement par lots', model: 'Modèle', score: 'mAP50 (validation)', conf: 'Meilleure confiance', perClass: 'Par classe', speed: 'Images/s', epochs: 'Époques', time: 'Durée', test: 'mAP de test', statusNames: { waiting: 'en attente', running: 'annulé ou en cours', done: 'terminé', failed: 'échec', skipped: 'ignoré' }, best: 'Recommandé', apply: 'Utiliser cette confiance', basis: (count: number) => `Mesuré sur ${count} images de validation. À qualité égale, le plus petit modèle est recommandé.`, smallSet: 'Peu d’images de validation : les petites différences sont dues au hasard.', hints: { cancel: 'Ferme la boîte de dialogue sans entraîner.', start: 'Lance l’entraînement avec ces réglages.', apply: 'Règle le curseur de confiance sur la meilleure valeur de ce modèle.' },
  },
};

export function TrainDialog({ language, current, epochs, freshStart, onCancel, onStart }: { language: Language; current: TrainSize; epochs: number; freshStart: boolean; onCancel: () => void; onStart: (request: TrainRequest) => void }) {
  const t = TEXT[language];
  const [chosen, setChosen] = useState<TrainSize[]>([current]);
  const [auto, setAuto] = useState(false);
  const [patience, setPatience] = useState(10);
  const [count, setCount] = useState(epochs);
  const [fresh, setFresh] = useState(freshStart);
  const [sweep, setSweep] = useState(true);
  const toggle = (size: TrainSize) => {
    const next = chosen.includes(size) ? chosen.filter((item) => item !== size) : [...chosen, size];
    // Several models are only comparable from the same starting point.
    if (next.length > 1 && chosen.length <= 1) setFresh(true);
    setChosen(next);
  };
  return <div className="walkthrough-backdrop train-backdrop"><div className="train-dialog" role="dialog" aria-modal="true" aria-label={t.title}>
    <h2>{t.title}</h2>
    <fieldset><legend>{t.which}</legend>
      {SIZES.map((size) => <label key={size}><input type="checkbox" checked={chosen.includes(size)} onChange={() => toggle(size)} /> RF-DETR {SIZE_NAMES[size]}{size === current ? ` (${t.current})` : ''}</label>)}
      {chosen.some((size) => size === 'medium' || size === 'large') && <p className="train-note">{t.largeNote}</p>}
      {chosen.length > 1 && <p className="train-note">{t.order}</p>}
    </fieldset>
    <fieldset><legend>{t.epochMode}</legend>
      <label><input type="radio" name="epoch-mode" checked={!auto} onChange={() => setAuto(false)} /> {t.fixed} <input type="number" min={1} max={500} value={count} disabled={auto} onChange={(event) => setCount(Math.max(1, Math.min(500, Number(event.target.value) || 1)))} /></label>
      <label><input type="radio" name="epoch-mode" checked={auto} onChange={() => setAuto(true)} /> {t.auto}</label>
      {auto && <label className="train-indented">{t.patience} <input type="number" min={3} max={50} value={patience} onChange={(event) => setPatience(Math.max(3, Math.min(50, Number(event.target.value) || 10)))} /></label>}
    </fieldset>
    <fieldset>
      <label><input type="checkbox" checked={fresh} onChange={(event) => setFresh(event.target.checked)} /> {t.fresh}</label>
      <label><input type="checkbox" checked={sweep} onChange={(event) => setSweep(event.target.checked)} /> {t.sweep}</label>
    </fieldset>
    <p className="train-note">{t.night}</p>
    <div className="train-actions">
      <button data-hint={t.hints.cancel} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
      <button data-hint={t.hints.start} type="button" className="primary-button" disabled={!chosen.length} onClick={() => onStart({ models: chosen, autoEpochs: auto, patience, epochs: count, freshStart: fresh, sweep })}>{t.start}</button>
    </div>
  </div></div>;
}

const percent = (value?: number) => value == null ? '–' : `${Math.round(value * 1000) / 10} %`;
const share = (value?: number) => value == null ? '–' : `${Math.round(value * 100)} %`;

export function BatchReportPanel({ language, api, busy, onApply }: { language: Language; api: string; busy: boolean; onApply: (threshold: number) => void }) {
  const t = TEXT[language];
  const [report, setReport] = useState<BatchReport | null>(null);
  // A finished job may have written a new report.
  useEffect(() => {
    if (busy) return;
    fetch(`${api}/api/batch-report`).then((response) => response.json()).then((value: BatchReport) => setReport(value.models ? value : null)).catch(() => setReport(null));
  }, [api, busy]);
  if (!report?.models) return null;
  const images = report.models.find((item) => item.validationImages)?.validationImages;
  return <section className="batch-report"><strong>{t.report}</strong>
    <div className="batch-report-scroll"><table><thead><tr><th>{t.model}</th><th>{t.score}</th><th>{t.conf}</th><th>{t.perClass}</th><th>{t.test}</th><th>{t.speed}</th><th>{t.epochs}</th><th>{t.time}</th><th /></tr></thead><tbody>
      {report.models.map((item) => <tr key={item.model} className={report.recommended === item.model ? 'recommended' : ''}>
        <td>{SIZE_NAMES[item.model]}{report.recommended === item.model ? ` ★ ${t.best}` : ''}</td>
        {item.status === 'done'
          ? <><td>{percent(item.mAP50)}</td><td>{share(item.bestThreshold)}</td><td>{Object.entries(item.perClass ?? {}).map(([name, entry]) => `${name} ${share(entry.threshold)}`).join(', ') || (item.sweepError ?? '–')}</td><td>{percent(item.testMetrics?.['test/mAP_50_95'])}</td><td>{item.imagesPerSecond == null ? '–' : item.imagesPerSecond.toFixed(1)}</td><td>{item.epochs ?? '–'}</td><td>{item.minutes == null ? '–' : `${item.minutes} min`}</td>
            <td>{item.bestThreshold != null && <button data-hint={t.hints.apply} type="button" className="secondary-button" disabled={busy} onClick={() => onApply(item.bestThreshold as number)}>{t.apply}</button>}</td></>
          : <td colSpan={8}>{t.statusNames[item.status]}{item.error ? `: ${item.error}` : ''}</td>}
      </tr>)}
    </tbody></table></div>
    {images != null && <p className="train-note">{t.basis(images)}{images < 30 ? ` ${t.smallSet}` : ''}</p>}
  </section>;
}
