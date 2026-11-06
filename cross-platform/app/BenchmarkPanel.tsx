import { useState } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type GroundTruth = { createdAt?: string; datasetID?: string; sport?: string; frameCount: number; annotationCount: number; classes: string[] };
type PerClassMetrics = { groundTruth: number; predictions: number; truePositives: number; falsePositives: number; falseNegatives: number; precision: number; recall: number; f1: number; ap50: number; meanIoU: number };
type Metrics = { qualityScore: number; mAP50: number; precision: number; recall: number; f1: number; meanIoU: number; truePositives: number; falsePositives: number; falseNegatives: number; perClass?: Record<string, PerClassMetrics> };
type Result = { rank?: number | null; packageID: string; modelSize?: string; status: string; metrics?: Metrics; meanLatencyMs?: number; error?: string };
type Report = { createdAt?: string; runID?: string; datasetID?: string; matchesGroundTruth?: boolean; frameCount: number; annotationCount: number; modelCount: number; successfulModelCount: number; threshold: number; device: string; rankingMethod: string; classes: string[]; results: Result[] };
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
  /** Localized sport name for the hints, e.g. "Fußball". */
  sportLabel?: string;
  models: ModelPackage[];
  benchmark?: { groundTruth?: GroundTruth | null; latest?: Report | null };
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const text = {
  de: { title: 'Lokaler Modellvergleich', intro: 'Alle kompatiblen Modelle erhalten dieselben geprüften Bilder. Bilder und Vorhersagen verlassen diesen Rechner nicht.', reference: '1 · Richtige Antworten festlegen', referenceText: 'Prüfe zuerst jede Box. Bilder ohne Box werden nach dem Festlegen bewusst als negative Beispiele gewertet.', freeze: 'Geprüfte Antworten festlegen', frozen: 'Referenz festgelegt', frames: 'Bilder', boxes: 'richtige Boxen', pending: 'Auto-Vorschläge offen', models: '2 · Modelle prüfen', compatible: 'kompatible lokale Modelle', noModels: 'Noch keine passenden .recomodel-Pakete importiert.', run: '3 · Vergleich starten', threshold: 'Auswertungsschwelle', start: 'Alle Modelle lokal testen', running: 'Modelle werden nacheinander getestet …', ranking: 'Automatisches Ranking', rank: 'Rang', model: 'Modell', score: 'Qualität', precision: 'Präzision', recall: 'Trefferquote', f1: 'F1', latency: 'Zeit/Bild', mistakes: 'FP / FN', noResult: 'Noch kein Vergleich für diese Referenz vorhanden.', stale: 'Die Referenz wurde geändert. Bitte den Vergleich erneut starten.', method: 'Ranking: 70 % mAP@0.50 + 30 % F1; Geschwindigkeit entscheidet nur bei Gleichstand.', privacy: 'Nur lokale JSON-Ergebnisse werden unter .reco-training/benchmarks gespeichert.', perCategoryTitle: 'Vergleich je Kategorie (mAP@.50)', perCategoryHint: 'Zeigt z. B., ob ein Modell besser bei Bällen und ein anderes besser bei Schiedsrichtern ist. Bestwert je Zeile grün.', category: 'Kategorie', combineTitle: '4 · Kombiniertes Modell erstellen', combineHint: 'Für jede Kategorie das beste Modell wählen (vorausgefüllt) und ein neues, kombiniertes Modell erstellen - keine echte Gewichtsfusion, sondern jedes Modell übernimmt zur Laufzeit nur seine zugeordneten Kategorien.', combineNamePlaceholder: 'Name des kombinierten Modells', combineButton: 'Kombiniertes Modell erstellen', combining: 'Kombiniertes Modell wird erstellt …', combineNeedsTwo: 'Mindestens zwei erfolgreich getestete Modelle werden benötigt.' },
  en: { title: 'Local model benchmark', intro: 'Every compatible model receives the same reviewed images. Images and predictions never leave this computer.', reference: '1 · Freeze the correct answers', referenceText: 'Review every box first. Images without boxes become intentional negative examples when frozen.', freeze: 'Freeze reviewed answers', frozen: 'Ground truth frozen', frames: 'images', boxes: 'correct boxes', pending: 'auto suggestions pending', models: '2 · Check models', compatible: 'compatible local models', noModels: 'No matching .recomodel packages have been imported yet.', run: '3 · Start comparison', threshold: 'Evaluation threshold', start: 'Test every model locally', running: 'Testing models sequentially …', ranking: 'Automatic ranking', rank: 'Rank', model: 'Model', score: 'Quality', precision: 'Precision', recall: 'Recall', f1: 'F1', latency: 'Time/image', mistakes: 'FP / FN', noResult: 'No benchmark exists for this ground truth yet.', stale: 'The ground truth changed. Run the benchmark again.', method: 'Ranking: 70% mAP@0.50 + 30% F1; speed is only the tie-breaker.', privacy: 'Only local JSON results are stored below .reco-training/benchmarks.', perCategoryTitle: 'Per-category comparison (mAP@.50)', perCategoryHint: 'Shows e.g. whether one model is better at balls and another better at referees. Best value per row in green.', category: 'Category', combineTitle: '4 · Create a combined model', combineHint: 'Pick the best model for each category (pre-filled) and create a new combined model - not real weight merging, each model just handles its assigned categories at runtime.', combineNamePlaceholder: 'Combined model name', combineButton: 'Create combined model', combining: 'Creating combined model …', combineNeedsTwo: 'At least two successfully tested models are needed.' },
  es: { title: 'Comparación local de modelos', intro: 'Todos los modelos compatibles reciben las mismas imágenes revisadas. Las imágenes y predicciones nunca salen de este equipo.', reference: '1 · Fijar las respuestas correctas', referenceText: 'Revisa primero cada cuadro. Las imágenes sin cuadros se consideran ejemplos negativos intencionados.', freeze: 'Fijar respuestas revisadas', frozen: 'Referencia fijada', frames: 'imágenes', boxes: 'cuadros correctos', pending: 'sugerencias automáticas pendientes', models: '2 · Comprobar modelos', compatible: 'modelos locales compatibles', noModels: 'Aún no se han importado paquetes .recomodel compatibles.', run: '3 · Iniciar comparación', threshold: 'Umbral de evaluación', start: 'Probar todos los modelos localmente', running: 'Probando los modelos uno tras otro …', ranking: 'Clasificación automática', rank: 'Puesto', model: 'Modelo', score: 'Calidad', precision: 'Precisión', recall: 'Cobertura', f1: 'F1', latency: 'Tiempo/imagen', mistakes: 'FP / FN', noResult: 'Todavía no existe una comparación para esta referencia.', stale: 'La referencia ha cambiado. Ejecuta de nuevo la comparación.', method: 'Clasificación: 70 % mAP@0.50 + 30 % F1; la velocidad solo desempata.', privacy: 'Solo se guardan resultados JSON locales en .reco-training/benchmarks.', perCategoryTitle: 'Comparación por categoría (mAP@.50)', perCategoryHint: 'Muestra, por ejemplo, si un modelo es mejor con los balones y otro con los árbitros. El mejor valor de cada fila en verde.', category: 'Categoría', combineTitle: '4 · Crear modelo combinado', combineHint: 'Elige el mejor modelo para cada categoría (rellenado previamente) y crea un nuevo modelo combinado - no es una fusión real de pesos, cada modelo solo gestiona sus categorías asignadas en tiempo de ejecución.', combineNamePlaceholder: 'Nombre del modelo combinado', combineButton: 'Crear modelo combinado', combining: 'Creando modelo combinado …', combineNeedsTwo: 'Se necesitan al menos dos modelos probados correctamente.' },
  fr: { title: 'Comparaison locale des modèles', intro: 'Tous les modèles compatibles reçoivent les mêmes images vérifiées. Les images et prédictions ne quittent jamais cet ordinateur.', reference: '1 · Figer les bonnes réponses', referenceText: 'Vérifiez d’abord chaque boîte. Les images sans boîte deviennent des exemples négatifs intentionnels.', freeze: 'Figer les réponses vérifiées', frozen: 'Référence figée', frames: 'images', boxes: 'boîtes correctes', pending: 'suggestions automatiques en attente', models: '2 · Vérifier les modèles', compatible: 'modèles locaux compatibles', noModels: 'Aucun paquet .recomodel compatible n’a encore été importé.', run: '3 · Lancer la comparaison', threshold: 'Seuil d’évaluation', start: 'Tester tous les modèles localement', running: 'Test séquentiel des modèles …', ranking: 'Classement automatique', rank: 'Rang', model: 'Modèle', score: 'Qualité', precision: 'Précision', recall: 'Rappel', f1: 'F1', latency: 'Temps/image', mistakes: 'FP / FN', noResult: 'Aucune comparaison n’existe encore pour cette référence.', stale: 'La référence a changé. Relancez la comparaison.', method: 'Classement : 70 % mAP@0.50 + 30 % F1 ; la vitesse départage seulement les égalités.', privacy: 'Seuls des résultats JSON locaux sont stockés sous .reco-training/benchmarks.', perCategoryTitle: 'Comparaison par catégorie (mAP@.50)', perCategoryHint: 'Montre par exemple si un modèle est meilleur pour les ballons et un autre pour les arbitres. Meilleure valeur de chaque ligne en vert.', category: 'Catégorie', combineTitle: '4 · Créer un modèle combiné', combineHint: 'Choisissez le meilleur modèle pour chaque catégorie (pré-rempli) et créez un nouveau modèle combiné - pas une vraie fusion de poids, chaque modèle ne gère que ses catégories assignées au moment de l’exécution.', combineNamePlaceholder: 'Nom du modèle combiné', combineButton: 'Créer le modèle combiné', combining: 'Création du modèle combiné …', combineNeedsTwo: 'Au moins deux modèles testés avec succès sont nécessaires.' },
};

type Reason = { why: string; next: string };
/** Why "Test models" cannot run yet and what to do next - shown above the steps and on disabled buttons. */
const readiness = {
  de: {
    ready: 'Bereit: Alle Voraussetzungen erfüllt. Starte den Vergleich in Schritt 3.', notReady: 'Der Modellvergleich ist noch nicht nutzbar:', nextLabel: 'Nächster Schritt',
    noFrames: { why: 'Noch keine Bilder im Projekt.', next: 'Im Training einen Arbeitsordner wählen und Bilder markieren.' },
    pending: (count: number) => ({ why: count === 1 ? 'Ein automatischer Vorschlag ist noch nicht geprüft.' : `${count} automatische Vorschläge sind noch nicht geprüft.`, next: 'Im Training „Auto übernehmen“ oder „Auto verwerfen“ wählen, dann hier die Antworten festlegen.' }),
    noTestset: { why: 'Noch kein bestätigtes Testset.', next: 'Schritt 1: „Geprüfte Antworten festlegen“.' },
    noModel: (sport: string) => ({ why: `Noch kein trainiertes Modell für ${sport}.`, next: 'Im Training „Lokal trainieren“ oder ein Modell importieren (auch aus dem Model Hub).' }),
    busy: { why: 'Gerade läuft ein anderer Vorgang.', next: 'Warten, bis er fertig ist.' },
    combineBusy: 'Gerade läuft ein anderer Vorgang. Warten, bis er fertig ist.',
  },
  en: {
    ready: 'Ready: everything is in place. Start the comparison in step 3.', notReady: 'The model comparison cannot run yet:', nextLabel: 'Next step',
    noFrames: { why: 'The project has no images yet.', next: 'In Training, choose a working folder and annotate images.' },
    pending: (count: number) => ({ why: count === 1 ? 'One automatic suggestion is not reviewed yet.' : `${count} automatic suggestions are not reviewed yet.`, next: 'In Training, choose “Accept auto” or “Reject auto”, then freeze the answers here.' }),
    noTestset: { why: 'No confirmed test set yet.', next: 'Step 1: “Freeze reviewed answers”.' },
    noModel: (sport: string) => ({ why: `No trained model for ${sport} yet.`, next: 'In Training, use “Train locally” or import a model (also from the Model Hub).' }),
    busy: { why: 'Another operation is running.', next: 'Wait until it has finished.' },
    combineBusy: 'Another operation is running. Wait until it has finished.',
  },
  es: {
    ready: 'Listo: se cumplen todos los requisitos. Inicia la comparación en el paso 3.', notReady: 'La comparación de modelos aún no se puede usar:', nextLabel: 'Siguiente paso',
    noFrames: { why: 'El proyecto aún no tiene imágenes.', next: 'En Entrenamiento, elige una carpeta de trabajo y marca imágenes.' },
    pending: (count: number) => ({ why: count === 1 ? 'Una sugerencia automática aún no se ha revisado.' : `${count} sugerencias automáticas aún no se han revisado.`, next: 'En Entrenamiento, elige «Aceptar auto» o «Rechazar auto» y luego fija aquí las respuestas.' }),
    noTestset: { why: 'Aún no hay un conjunto de prueba confirmado.', next: 'Paso 1: «Fijar respuestas revisadas».' },
    noModel: (sport: string) => ({ why: `Aún no hay un modelo entrenado para ${sport}.`, next: 'En Entrenamiento, usa «Entrenar localmente» o importa un modelo (también desde el Model Hub).' }),
    busy: { why: 'Hay otra operación en curso.', next: 'Espera a que termine.' },
    combineBusy: 'Hay otra operación en curso. Espera a que termine.',
  },
  fr: {
    ready: 'Prêt : toutes les conditions sont remplies. Lancez la comparaison à l’étape 3.', notReady: 'La comparaison des modèles n’est pas encore utilisable :', nextLabel: 'Étape suivante',
    noFrames: { why: 'Le projet n’a pas encore d’images.', next: 'Dans Entraînement, choisissez un dossier de travail et annotez des images.' },
    pending: (count: number) => ({ why: count === 1 ? 'Une suggestion automatique n’est pas encore vérifiée.' : `${count} suggestions automatiques ne sont pas encore vérifiées.`, next: 'Dans Entraînement, choisissez « Accepter auto » ou « Refuser auto », puis figez les réponses ici.' }),
    noTestset: { why: 'Pas encore de jeu de test confirmé.', next: 'Étape 1 : « Figer les réponses vérifiées ».' },
    noModel: (sport: string) => ({ why: `Pas encore de modèle entraîné pour ${sport}.`, next: 'Dans Entraînement, utilisez « Entraîner localement » ou importez un modèle (aussi depuis le Model Hub).' }),
    busy: { why: 'Une autre opération est en cours.', next: 'Attendez qu’elle soit terminée.' },
    combineBusy: 'Une autre opération est en cours. Attendez qu’elle soit terminée.',
  },
};

const metricHelp = {
  de: [['Qualität', 'Gesamtrang: 70 % mAP@0.50 und 30 % F1. Höher ist besser.'], ['mAP@.50', 'Erkennungsqualität über viele Sicherheitsschwellen; eine Box zählt ab 50 % Überlappung als richtig.'], ['Präzision', 'Anteil der gemeldeten Treffer, die richtig sind. Hoch bedeutet wenige Fehlalarme.'], ['Recall', 'Anteil der vorhandenen Objekte, die gefunden wurden. Hoch bedeutet wenige übersehene Bälle.'], ['F1', 'Ausgleich zwischen Präzision und Recall.'], ['FP / FN', 'Falsche Treffer / übersehene echte Objekte. Weniger ist besser.'], ['Zeit/Bild', 'Mittlere Rechenzeit pro Bild. Niedriger ist schneller.'], ['Schwelle', 'Minimale Sicherheit. Niedriger findet mehr, erzeugt aber meist mehr Fehlalarme.']],
  en: [['Quality', 'Overall rank: 70% mAP@0.50 and 30% F1. Higher is better.'], ['mAP@.50', 'Detection quality across confidence levels; a box counts as correct from 50% overlap.'], ['Precision', 'Share of reported detections that are correct. High means fewer false alarms.'], ['Recall', 'Share of real objects that were found. High means fewer missed balls.'], ['F1', 'Balance between precision and recall.'], ['FP / FN', 'False detections / missed real objects. Lower is better.'], ['Time/image', 'Average processing time per image. Lower is faster.'], ['Threshold', 'Minimum confidence. Lower finds more but usually creates more false alarms.']],
  es: [['Calidad', 'Clasificación total: 70 % mAP@0.50 y 30 % F1. Un valor mayor es mejor.'], ['mAP@.50', 'Calidad de detección con varios niveles de confianza; un cuadro es correcto desde un 50 % de solapamiento.'], ['Precisión', 'Proporción de detecciones notificadas que son correctas.'], ['Cobertura', 'Proporción de objetos reales encontrados. Un valor alto significa menos balones omitidos.'], ['F1', 'Equilibrio entre precisión y cobertura.'], ['FP / FN', 'Detecciones falsas / objetos reales omitidos. Menos es mejor.'], ['Tiempo/imagen', 'Tiempo medio por imagen. Un valor menor es más rápido.'], ['Umbral', 'Confianza mínima. Un valor menor encuentra más, pero suele producir más falsas alarmas.']],
  fr: [['Qualité', 'Classement global : 70 % mAP@0.50 et 30 % F1. Plus élevé est meilleur.'], ['mAP@.50', 'Qualité de détection à plusieurs niveaux de confiance ; une boîte est correcte dès 50 % de chevauchement.'], ['Précision', 'Part des détections signalées qui sont correctes.'], ['Rappel', 'Part des objets réels trouvés. Une valeur élevée signifie moins de ballons manqués.'], ['F1', 'Équilibre entre précision et rappel.'], ['FP / FN', 'Fausses détections / objets réels manqués. Moins est meilleur.'], ['Temps/image', 'Temps moyen par image. Plus bas est plus rapide.'], ['Seuil', 'Confiance minimale. Plus bas trouve davantage, mais produit souvent plus de fausses alertes.']],
} as const;

const percent = (value?: number) => value == null ? '–' : `${(value * 100).toFixed(1)}%`;

export default function BenchmarkPanel(props: Props) {
  const t = text[props.language];
  const [threshold, setThreshold] = useState(.05);
  const [actionError, setActionError] = useState<string | null>(null);
  const [combineOverrides, setCombineOverrides] = useState<Record<string, string>>({});
  const [combineName, setCombineName] = useState('');
  const groundTruth = props.benchmark?.groundTruth ?? null;
  const report = props.benchmark?.latest ?? null;
  const compatible = props.models.filter((model) => model.sport === props.sport);
  const isRunning = props.busy && props.operation === 'benchmark';
  const isCombining = props.busy && props.operation === 'combine-models';
  const successfulResults = report?.results.filter((result) => result.status === 'completed') ?? [];
  const r = readiness[props.language];
  const sportName = props.sportLabel || props.sport || '–';
  // Each control's blocker, in the order the user has to resolve them.
  const busyReason = props.busy && !isRunning ? r.busy : null;
  const freezeReason: Reason | null = busyReason ?? (!props.frameCount ? r.noFrames : props.automaticCount ? r.pending(props.automaticCount) : null);
  const runReason: Reason | null = busyReason ?? (!props.frameCount ? r.noFrames : !groundTruth ? (props.automaticCount ? r.pending(props.automaticCount) : r.noTestset) : !compatible.length ? r.noModel(sportName) : null);
  const blockers: Reason[] = isRunning ? [] : [
    ...(busyReason ? [busyReason] : []),
    ...(!props.frameCount ? [r.noFrames] : []),
    ...(props.frameCount && !groundTruth && props.automaticCount ? [r.pending(props.automaticCount)] : []),
    ...(props.frameCount && !groundTruth ? [r.noTestset] : []),
    ...(!compatible.length ? [r.noModel(sportName)] : []),
  ];
  const reasonText = (reason: Reason | null) => reason ? `${reason.why} ${r.nextLabel}: ${reason.next}` : undefined;

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

  function bestPackageIDFor(category: string): string | undefined {
    let best: Result | undefined;
    for (const result of successfulResults) {
      const value = result.metrics?.perClass?.[category]?.ap50;
      if (value == null) continue;
      const bestValue = best?.metrics?.perClass?.[category]?.ap50;
      if (bestValue == null || value > bestValue) best = result;
    }
    return best?.packageID;
  }

  async function combine() {
    setActionError(null);
    const byModel = new Map<string, string[]>();
    for (const category of report?.classes ?? []) {
      const packageID = combineOverrides[category] ?? bestPackageIDFor(category);
      if (!packageID) continue;
      byModel.set(packageID, [...(byModel.get(packageID) ?? []), category]);
    }
    const members = [...byModel.entries()].map(([packageID, categories]) => ({ packageID, categories }));
    try { await props.post('/api/combine-models', { members, name: combineName, language: props.language }); setCombineName(''); setCombineOverrides({}); }
    catch (error) { setActionError(String(error)); }
  }

  return <section className="benchmark-workspace">
    <div className="benchmark-heading"><div><span className="benchmark-kicker">◆ BENCHMARK</span><h2>{t.title}</h2><p>{t.intro}</p></div><div className="benchmark-private"><span>●</span>{t.privacy}</div></div>
    {!isRunning && <div className={blockers.length ? 'benchmark-readiness blocked' : 'benchmark-readiness ready'} role="status">
      {blockers.length ? <><strong>{r.notReady}</strong><ul>{blockers.map((reason) => <li key={reason.why}><span>{reason.why}</span><small>{r.nextLabel}: {reason.next}</small></li>)}</ul></> : <strong>✓ {r.ready}</strong>}
    </div>}
    <div className="benchmark-steps">
      <article className={groundTruth ? 'benchmark-step complete' : 'benchmark-step'}><div className="step-number">1</div><h3>{t.reference}</h3><p>{t.referenceText}</p><div className="benchmark-facts"><span><strong>{groundTruth?.frameCount ?? props.frameCount}</strong>{t.frames}</span><span><strong>{groundTruth?.annotationCount ?? '–'}</strong>{t.boxes}</span><span className={props.automaticCount ? 'warning' : ''}><strong>{props.automaticCount}</strong>{t.pending}</span></div><span className="disabled-reason" title={reasonText(freezeReason)} data-hint={reasonText(freezeReason)}><button data-hint={reasonText(freezeReason) ?? hintText[props.language].freeze} type="button" className="primary-button" onClick={() => void freeze()} disabled={!!freezeReason || props.busy}>{groundTruth ? `✓ ${t.frozen}` : t.freeze}</button></span></article>
      <article className="benchmark-step"><div className="step-number">2</div><h3>{t.models}</h3><strong className="model-count">{compatible.length} {t.compatible}</strong>{compatible.length ? <div className="benchmark-model-list">{compatible.map((model) => <span key={model.packageID}><b>{model.modelSize.toUpperCase()}</b>{model.packageID}<small>{model.classes.join(', ')}</small></span>)}</div> : <p className="benchmark-empty-small">{t.noModels}</p>}</article>
      <article className="benchmark-step"><div className="step-number">3</div><h3>{t.run}</h3><label className="benchmark-threshold">{t.threshold}<strong>{Math.round(threshold * 100)}%</strong><input type="range" min="0.01" max="0.50" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label><span className="disabled-reason" title={reasonText(runReason)} data-hint={reasonText(runReason)}><button data-hint={reasonText(runReason) ?? hintText[props.language].benchmark} type="button" className="primary-button" onClick={() => void run()} disabled={!!runReason || props.busy}>{isRunning ? t.running : t.start}</button></span>{runReason && !isRunning && <small className="benchmark-disabled-note">{runReason.why}</small>}{isRunning && <div className="benchmark-progress"><span style={{ width: `${Math.max(props.progress, .02) * 100}%` }} /></div>}</article>
    </div>
    {(actionError || props.error) && <div className="benchmark-error">{actionError || props.error}</div>}
    <section className="ranking-card"><div className="ranking-title"><div><h3>{t.ranking}</h3><p>{t.method}</p></div>{report && <span>{new Date(report.createdAt ?? '').toLocaleString(props.language)}</span>}</div>
      <details className="benchmark-explanation"><summary>{props.language === 'de' ? 'Was bedeuten die Werte?' : props.language === 'es' ? '¿Qué significan los valores?' : props.language === 'fr' ? 'Que signifient les valeurs ?' : 'What do the metrics mean?'}</summary><div>{metricHelp[props.language].map(([name, description]) => <p key={name}><strong>{name}</strong><span>{description}</span></p>)}</div></details>
      {!report ? <div className="benchmark-empty">{t.noResult}</div> : report.matchesGroundTruth === false ? <div className="benchmark-empty stale">{t.stale}</div> : <div className="ranking-scroll"><table><thead><tr><th>{t.rank}</th><th>{t.model}</th><th>{t.score}</th><th>mAP@.50</th><th>{t.precision}</th><th>{t.recall}</th><th>{t.f1}</th><th>{t.latency}</th><th>{t.mistakes}</th></tr></thead><tbody>{report.results.map((result) => <tr key={result.packageID} className={result.rank === 1 ? 'winner' : ''}><td><strong>{result.rank ? `#${result.rank}` : '–'}</strong></td><td><b>{result.packageID}</b><small>{result.modelSize?.toUpperCase()}</small>{result.error && <em>{result.error}</em>}</td><td><strong>{result.metrics ? result.metrics.qualityScore.toFixed(1) : '–'}</strong></td><td>{percent(result.metrics?.mAP50)}</td><td>{percent(result.metrics?.precision)}</td><td>{percent(result.metrics?.recall)}</td><td>{percent(result.metrics?.f1)}</td><td>{result.meanLatencyMs == null ? '–' : `${result.meanLatencyMs.toFixed(0)} ms`}</td><td>{result.metrics ? `${result.metrics.falsePositives} / ${result.metrics.falseNegatives}` : '–'}</td></tr>)}</tbody></table></div>}
    </section>
    {report && report.matchesGroundTruth !== false && report.classes.length > 1 && successfulResults.length > 1 && <section className="ranking-card per-category-card">
      <div className="ranking-title"><div><h3>{t.perCategoryTitle}</h3><p>{t.perCategoryHint}</p></div></div>
      <div className="ranking-scroll"><table><thead><tr><th>{t.category}</th>{successfulResults.map((result) => <th key={result.packageID}>{result.packageID}</th>)}</tr></thead><tbody>{report.classes.map((category) => {
        const values = successfulResults.map((result) => result.metrics?.perClass?.[category]?.ap50);
        const bestValue = values.reduce<number | undefined>((best, value) => value == null ? best : (best == null || value > best ? value : best), undefined);
        return <tr key={category}><td>{category}</td>{successfulResults.map((result, index) => {
          const value = values[index];
          const isBest = value != null && bestValue != null && value === bestValue;
          return <td key={result.packageID} className={isBest ? 'winner' : ''}>{percent(value)}</td>;
        })}</tr>;
      })}</tbody></table></div>
    </section>}
    {report && report.matchesGroundTruth !== false && successfulResults.length > 1 && <section className="ranking-card combine-card">
      <div className="ranking-title"><div><h3>{t.combineTitle}</h3><p>{t.combineHint}</p></div></div>
      <div className="combine-form">
        {report.classes.map((category) => <label key={category} className="combine-category-row">{category}<select value={combineOverrides[category] ?? bestPackageIDFor(category) ?? ''} onChange={(event) => setCombineOverrides((current) => ({ ...current, [category]: event.target.value }))}>{successfulResults.map((result) => <option key={result.packageID} value={result.packageID}>{result.packageID}</option>)}</select></label>)}
        <input type="text" value={combineName} onChange={(event) => setCombineName(event.target.value)} placeholder={t.combineNamePlaceholder} className="combine-name-input" />
        <button data-hint={props.busy ? r.combineBusy : hintText[props.language].combine} title={props.busy ? r.combineBusy : undefined} type="button" className="primary-button" onClick={() => void combine()} disabled={props.busy}>{isCombining ? t.combining : t.combineButton}</button>
      </div>
    </section>}
    {(props.message || props.log) && <details className="worker-details benchmark-log"><summary>{props.message}</summary>{props.log && <pre className="worker-log">{props.log}</pre>}</details>}
  </section>;
}
