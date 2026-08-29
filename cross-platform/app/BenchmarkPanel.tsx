'use client';

import { useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
type GroundTruth = { createdAt?: string; datasetID?: string; sport?: string; frameCount: number; annotationCount: number; classes: string[] };
type Metrics = { qualityScore: number; mAP50: number; precision: number; recall: number; f1: number; meanIoU: number; truePositives: number; falsePositives: number; falseNegatives: number };
type Result = { rank?: number | null; packageID: string; modelSize?: string; status: string; metrics?: Metrics; meanLatencyMs?: number; error?: string };
type Report = { createdAt?: string; runID?: string; datasetID?: string; matchesGroundTruth?: boolean; frameCount: number; annotationCount: number; modelCount: number; successfulModelCount: number; threshold: number; device: string; rankingMethod: string; results: Result[] };
type ModelPackage = { packageID: string; sport: string; modelSize: string; classes: string[] };

type Props = {
  language: Language;
  busy: boolean;
  operation?: string;
  progress: number;
  message?: string;
  error?: string | null;
  log?: string;
  frameCount: number;
  automaticCount: number;
  sport?: string | null;
  models: ModelPackage[];
  benchmark?: { groundTruth?: GroundTruth | null; latest?: Report | null };
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const text = {
  de: { title: 'Lokaler Modellvergleich', intro: 'Alle kompatiblen Modelle erhalten dieselben geprüften Bilder. Bilder und Vorhersagen verlassen diesen Rechner nicht.', reference: '1 · Richtige Antworten festlegen', referenceText: 'Prüfe zuerst jede Box. Bilder ohne Box werden nach dem Festlegen bewusst als negative Beispiele gewertet.', freeze: 'Geprüfte Antworten festlegen', frozen: 'Referenz festgelegt', frames: 'Bilder', boxes: 'richtige Boxen', pending: 'Auto-Vorschläge offen', models: '2 · Modelle prüfen', compatible: 'kompatible lokale Modelle', noModels: 'Noch keine passenden .recomodel-Pakete importiert.', run: '3 · Vergleich starten', threshold: 'Auswertungsschwelle', start: 'Alle Modelle lokal testen', running: 'Modelle werden nacheinander getestet …', ranking: 'Automatisches Ranking', rank: 'Rang', model: 'Modell', score: 'Qualität', precision: 'Präzision', recall: 'Trefferquote', f1: 'F1', latency: 'Zeit/Bild', mistakes: 'FP / FN', noResult: 'Noch kein Vergleich für diese Referenz vorhanden.', stale: 'Die Referenz wurde geändert. Bitte den Vergleich erneut starten.', method: 'Ranking: 70 % mAP@0.50 + 30 % F1; Geschwindigkeit entscheidet nur bei Gleichstand.', privacy: 'Nur lokale JSON-Ergebnisse werden unter .reco-training/benchmarks gespeichert.' },
  en: { title: 'Local model benchmark', intro: 'Every compatible model receives the same reviewed images. Images and predictions never leave this computer.', reference: '1 · Freeze the correct answers', referenceText: 'Review every box first. Images without boxes become intentional negative examples when frozen.', freeze: 'Freeze reviewed answers', frozen: 'Ground truth frozen', frames: 'images', boxes: 'correct boxes', pending: 'auto suggestions pending', models: '2 · Check models', compatible: 'compatible local models', noModels: 'No matching .recomodel packages have been imported yet.', run: '3 · Start comparison', threshold: 'Evaluation threshold', start: 'Test every model locally', running: 'Testing models sequentially …', ranking: 'Automatic ranking', rank: 'Rank', model: 'Model', score: 'Quality', precision: 'Precision', recall: 'Recall', f1: 'F1', latency: 'Time/image', mistakes: 'FP / FN', noResult: 'No benchmark exists for this ground truth yet.', stale: 'The ground truth changed. Run the benchmark again.', method: 'Ranking: 70% mAP@0.50 + 30% F1; speed is only the tie-breaker.', privacy: 'Only local JSON results are stored below .reco-training/benchmarks.' },
  es: { title: 'Comparación local de modelos', intro: 'Todos los modelos compatibles reciben las mismas imágenes revisadas. Las imágenes y predicciones nunca salen de este equipo.', reference: '1 · Fijar las respuestas correctas', referenceText: 'Revisa primero cada cuadro. Las imágenes sin cuadros se consideran ejemplos negativos intencionados.', freeze: 'Fijar respuestas revisadas', frozen: 'Referencia fijada', frames: 'imágenes', boxes: 'cuadros correctos', pending: 'sugerencias automáticas pendientes', models: '2 · Comprobar modelos', compatible: 'modelos locales compatibles', noModels: 'Aún no se han importado paquetes .recomodel compatibles.', run: '3 · Iniciar comparación', threshold: 'Umbral de evaluación', start: 'Probar todos los modelos localmente', running: 'Probando los modelos uno tras otro …', ranking: 'Clasificación automática', rank: 'Puesto', model: 'Modelo', score: 'Calidad', precision: 'Precisión', recall: 'Cobertura', f1: 'F1', latency: 'Tiempo/imagen', mistakes: 'FP / FN', noResult: 'Todavía no existe una comparación para esta referencia.', stale: 'La referencia ha cambiado. Ejecuta de nuevo la comparación.', method: 'Clasificación: 70 % mAP@0.50 + 30 % F1; la velocidad solo desempata.', privacy: 'Solo se guardan resultados JSON locales en .reco-training/benchmarks.' },
  fr: { title: 'Comparaison locale des modèles', intro: 'Tous les modèles compatibles reçoivent les mêmes images vérifiées. Les images et prédictions ne quittent jamais cet ordinateur.', reference: '1 · Figer les bonnes réponses', referenceText: 'Vérifiez d’abord chaque boîte. Les images sans boîte deviennent des exemples négatifs intentionnels.', freeze: 'Figer les réponses vérifiées', frozen: 'Référence figée', frames: 'images', boxes: 'boîtes correctes', pending: 'suggestions automatiques en attente', models: '2 · Vérifier les modèles', compatible: 'modèles locaux compatibles', noModels: 'Aucun paquet .recomodel compatible n’a encore été importé.', run: '3 · Lancer la comparaison', threshold: 'Seuil d’évaluation', start: 'Tester tous les modèles localement', running: 'Test séquentiel des modèles …', ranking: 'Classement automatique', rank: 'Rang', model: 'Modèle', score: 'Qualité', precision: 'Précision', recall: 'Rappel', f1: 'F1', latency: 'Temps/image', mistakes: 'FP / FN', noResult: 'Aucune comparaison n’existe encore pour cette référence.', stale: 'La référence a changé. Relancez la comparaison.', method: 'Classement : 70 % mAP@0.50 + 30 % F1 ; la vitesse départage seulement les égalités.', privacy: 'Seuls des résultats JSON locaux sont stockés sous .reco-training/benchmarks.' },
};

const percent = (value?: number) => value == null ? '–' : `${(value * 100).toFixed(1)}%`;

export default function BenchmarkPanel(props: Props) {
  const t = text[props.language];
  const [threshold, setThreshold] = useState(.05);
  const [actionError, setActionError] = useState<string | null>(null);
  const groundTruth = props.benchmark?.groundTruth ?? null;
  const report = props.benchmark?.latest ?? null;
  const compatible = props.models.filter((model) => model.sport === props.sport);
  const isRunning = props.busy && props.operation === 'benchmark';

  async function freeze() {
    setActionError(null);
    try { await props.post('/api/benchmark-ground-truth'); }
    catch (error) { setActionError(String(error)); }
  }

  async function run() {
    setActionError(null);
    try { await props.post('/api/benchmark', { threshold, language: props.language }); }
    catch (error) { setActionError(String(error)); }
  }

  return <section className="benchmark-workspace">
    <div className="benchmark-heading"><div><span className="benchmark-kicker">◆ BENCHMARK</span><h2>{t.title}</h2><p>{t.intro}</p></div><div className="benchmark-private"><span>●</span>{t.privacy}</div></div>
    <div className="benchmark-steps">
      <article className={groundTruth ? 'benchmark-step complete' : 'benchmark-step'}><div className="step-number">1</div><h3>{t.reference}</h3><p>{t.referenceText}</p><div className="benchmark-facts"><span><strong>{groundTruth?.frameCount ?? props.frameCount}</strong>{t.frames}</span><span><strong>{groundTruth?.annotationCount ?? '–'}</strong>{t.boxes}</span><span className={props.automaticCount ? 'warning' : ''}><strong>{props.automaticCount}</strong>{t.pending}</span></div><button type="button" className="primary-button" onClick={() => void freeze()} disabled={props.busy || !props.frameCount || !!props.automaticCount}>{groundTruth ? `✓ ${t.frozen}` : t.freeze}</button></article>
      <article className="benchmark-step"><div className="step-number">2</div><h3>{t.models}</h3><strong className="model-count">{compatible.length} {t.compatible}</strong>{compatible.length ? <div className="benchmark-model-list">{compatible.map((model) => <span key={model.packageID}><b>{model.modelSize.toUpperCase()}</b>{model.packageID}<small>{model.classes.join(', ')}</small></span>)}</div> : <p className="benchmark-empty-small">{t.noModels}</p>}</article>
      <article className="benchmark-step"><div className="step-number">3</div><h3>{t.run}</h3><label className="benchmark-threshold">{t.threshold}<strong>{Math.round(threshold * 100)}%</strong><input type="range" min="0.01" max="0.50" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label><button type="button" className="primary-button" onClick={() => void run()} disabled={props.busy || !groundTruth || !compatible.length}>{isRunning ? t.running : t.start}</button>{isRunning && <div className="benchmark-progress"><span style={{ width: `${Math.max(props.progress, .02) * 100}%` }} /></div>}</article>
    </div>
    {(actionError || props.error) && <div className="benchmark-error">{actionError || props.error}</div>}
    <section className="ranking-card"><div className="ranking-title"><div><h3>{t.ranking}</h3><p>{t.method}</p></div>{report && <span>{new Date(report.createdAt ?? '').toLocaleString(props.language)}</span>}</div>
      {!report ? <div className="benchmark-empty">{t.noResult}</div> : report.matchesGroundTruth === false ? <div className="benchmark-empty stale">{t.stale}</div> : <div className="ranking-scroll"><table><thead><tr><th>{t.rank}</th><th>{t.model}</th><th>{t.score}</th><th>mAP@.50</th><th>{t.precision}</th><th>{t.recall}</th><th>{t.f1}</th><th>{t.latency}</th><th>{t.mistakes}</th></tr></thead><tbody>{report.results.map((result) => <tr key={result.packageID} className={result.rank === 1 ? 'winner' : ''}><td><strong>{result.rank ? `#${result.rank}` : '–'}</strong></td><td><b>{result.packageID}</b><small>{result.modelSize?.toUpperCase()}</small>{result.error && <em>{result.error}</em>}</td><td><strong>{result.metrics ? result.metrics.qualityScore.toFixed(1) : '–'}</strong></td><td>{percent(result.metrics?.mAP50)}</td><td>{percent(result.metrics?.precision)}</td><td>{percent(result.metrics?.recall)}</td><td>{percent(result.metrics?.f1)}</td><td>{result.meanLatencyMs == null ? '–' : `${result.meanLatencyMs.toFixed(0)} ms`}</td><td>{result.metrics ? `${result.metrics.falsePositives} / ${result.metrics.falseNegatives}` : '–'}</td></tr>)}</tbody></table></div>}
    </section>
    {(props.message || props.log) && <details className="worker-details benchmark-log"><summary>{props.message}</summary>{props.log && <pre className="worker-log">{props.log}</pre>}</details>}
  </section>;
}
