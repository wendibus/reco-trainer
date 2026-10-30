import { useCallback, useEffect, useState, type MouseEvent } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
type Role = 'player' | 'referee' | 'ball' | 'static' | 'ignore';
type Split = 'train' | 'valid' | 'test';
type Config = {
  variations?: number; cropShare?: number; zoom?: [number, number]; outputSize?: number; seed?: number;
  roles?: Record<string, Role>; limits?: { players: number; referees: number; balls: number };
  jerseyVariation?: boolean; yolo?: boolean; anonymize?: boolean; minVisible?: number;
  sessionOverrides?: Record<string, string>; sessionLinks?: string[][]; teamOverrides?: Record<string, number>;
  polygons?: Record<string, number[][]>; splitOverrides?: Record<string, Split>;
};
type Session = {
  id: string; images: string[]; split: Split; width: number; height: number; plateFile?: string | null;
  polygon: number[][]; polygonSource: 'clicked' | 'calibration' | 'estimated'; homography?: unknown;
  teamCounts: Record<string, number>; link: string;
};
type Instance = { id: string; session: string; role: string; team?: number; uncertain?: boolean; complete?: boolean; image: string };
type Analysis = { classes: string[]; sessions: Session[]; instances: Instance[]; splitWarning?: string | null; log: string[]; dropped: number };
type Report = { images: number; violations: Record<string, number>; foundThenRegenerated: Record<string, number>; rejectedWhileGenerating: Record<string, number>; labelsTooSmall: number; real: Record<Split, number> };
type StudioState = {
  source: string | null; config?: Config; analysis?: Analysis | null; report?: Report | null; previews?: string[]; generated?: number;
  training?: { mode: 'mixed' | 'real-only'; realWeight: number } | null;
};
type Props = {
  language: Language; api: string; busy: boolean; modelSize: string;
  post: (path: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

const MODEL_INPUT: Record<string, number> = { nano: 384, small: 512, medium: 576, large: 704 };
const DEFAULTS = { cropShare: 0.6, zoom: [0.2, 0.7] as [number, number], seed: 1, players: 5, referees: 3, balls: 1, minVisible: 0.3 };

const text: Record<Language, {
  title: string; intro: string; step1: string; choose: string; none: string; rolesTitle: string; roles: Record<Role, string>; step2: string;
  variations: string; cropShare: string; zoomLow: string; zoomHigh: string; output: string; seed: string; players: string; referees: string; balls: string; minVisible: string;
  jersey: string; yolo: string; anonymize: string; step3: string; analyze: string; reanalyze: string; correctionsHint: string; images: string; plate: string; split: Record<Split, string>;
  link: string; noLink: string; moveTo: string; newSession: string; imagesList: string; polygon: Record<Session['polygonSource'], string>; polygonClick: string; polygonSave: string; polygonReset: string; polygonEdit: string;
  team: (n: number) => string; outliers: string; referee: string; uncertainHint: string; notMovable: string; warnOne: string; warnTwo: string; noTrain: string; dropped: (n: number) => string; step4: string; generate: string; resume: (done: number) => string; needVariations: string;
  report: (n: number) => string; violationsNone: string; violations: string; regenerated: (n: number) => string; tooSmall: (n: number) => string; realCounts: (t: number, v: number, s: number) => string; preview: string; step5: string; trainingMode: string; modeOff: string; modeMixed: string; modeReal: string; realWeight: string; trainingHint: string;
  hints: { choose: string; analyze: string; generate: string; polygonSave: string; polygonReset: string; polygonEdit: string; instance: string };
}> = {
  de: {
    title: 'Augmentation Studio', intro: 'Erzeugt aus gelabelten Full-Court-Aufnahmen synthetische Trainingsbilder mit passender Perspektive, korrekten Labels und nie mehr als zwei Mannschaften.',
    step1: '1 · Quellordner', choose: 'Quellordner wählen', none: 'Bilder und COCO-Annotationen, zum Beispiel aus „Datensatz exportieren“.', rolesTitle: 'Rolle je Klasse', roles: { player: 'Spieler', referee: 'Schiedsrichter', ball: 'Ball', static: 'Statisch', ignore: 'Ignorieren' },
    step2: '2 · Einstellungen', variations: 'Anzahl Variationen', cropShare: 'Anteil Ausschnitte %', zoomLow: 'Zoom von % der Breite', zoomHigh: 'Zoom bis % der Breite', output: 'Ausgabegröße px', seed: 'Seed', players: 'Max. Spieler je Team', referees: 'Max. Schiedsrichter', balls: 'Max. Bälle', minVisible: 'Mind. sichtbar %',
    jersey: 'Trikotfarben variieren', yolo: 'Zusätzlich YOLO-Export', anonymize: 'Köpfe anonymisieren', step3: '3 · Analyse prüfen', analyze: 'Analysieren', reanalyze: 'Änderungen übernehmen und neu analysieren', correctionsHint: 'Fertige Masken und Platten bleiben erhalten, das geht deutlich schneller als beim ersten Mal.',
    images: 'Bilder', plate: 'Leere Hallenplatte', split: { train: 'Training', valid: 'Validierung', test: 'Test' }, link: 'Gleiche Paarung wie', noLink: 'eigene Paarung', moveTo: 'Verschieben nach', newSession: 'neue Session', imagesList: 'Bilder dieser Session',
    polygon: { clicked: 'Spielfeld angeklickt', calibration: 'Spielfeld aus Kalibrierung', estimated: 'Spielfeld geschätzt aus Fußpunkten, bitte prüfen' }, polygonClick: 'Spielfeldecken der Reihe nach anklicken.', polygonSave: 'Spielfeld speichern', polygonReset: 'Punkte löschen', polygonEdit: 'Spielfeld anklicken',
    team: (n) => `Team ${n + 1}`, outliers: 'Ausreißer (werden nicht verwendet)', referee: 'Schiedsrichter', uncertainHint: 'Orange umrandet = unsicher. Klick wechselt das Team.', notMovable: 'Halb verdeckte oder angeschnittene Personen sind blass und werden nicht verschoben.',
    warnOne: 'Nur eine Session: Es gibt keine echten Bilder für Validierung und Test. Für eine ehrliche Bewertung weitere Aufnahmen ergänzen.', warnTwo: 'Nur zwei Sessions: Es gibt keinen unabhängigen Test.', noTrain: 'Keine Session ist fürs Training eingeteilt.', dropped: (n) => `${n} Instanzen wegen schlechter Maske verworfen (siehe Protokoll).`,
    step4: '4 · Erzeugen', generate: 'Datensatz erzeugen', resume: (done) => `${done} Bilder sind schon fertig; gleiche Einstellungen setzen dort fort.`, needVariations: 'Bitte die Anzahl der Variationen angeben.',
    report: (n) => `${n} Bilder erzeugt.`, violationsNone: 'Validator: 0 Verstöße.', violations: 'Validator meldet Verstöße:', regenerated: (n) => `${n} Verstöße beim Erzeugen abgefangen und neu erzeugt.`, tooSmall: (n) => `${n} Labels waren in der Ausgabegröße kleiner als 1,5 px und wurden entfernt.`, realCounts: (t, v, s) => `Echte Bilder: ${t} Training, ${v} Validierung, ${s} Test.`, preview: 'Vorschau mit Labels und Teamfarben',
    step5: '5 · Training', trainingMode: 'Beim Training verwenden', modeOff: 'Nein, normale Projektbilder', modeMixed: 'Echt + synthetisch', modeReal: 'Nur echt, gleiche Aufteilung (Vergleich)', realWeight: 'Gewicht echter Bilder', trainingHint: 'Validierung und Test bestehen immer nur aus echten Bildern. „Lokal trainieren“ startet das Training; die Metriken beider Modi sind direkt vergleichbar.',
    hints: { choose: 'Wählt den Ordner mit Bildern und COCO-Annotationen.', analyze: 'Erzeugt Masken, Hallenplatten, Spielfeld und Teams.', generate: 'Erzeugt die Bilder und prüft sie danach automatisch.', polygonSave: 'Speichert diese Punkte als Spielfeld.', polygonReset: 'Löscht die gesetzten Punkte.', polygonEdit: 'Legt das Spielfeld auf der Hallenplatte fest.', instance: 'Wechselt das Team dieser Person.' },
  },
  en: {
    title: 'Augmentation Studio', intro: 'Creates synthetic training images from labelled full-court footage, with matching perspective, correct labels and never more than two teams.',
    step1: '1 · Source folder', choose: 'Select source folder', none: 'Images and COCO annotations, for example from “Export dataset”.', rolesTitle: 'Role per class', roles: { player: 'Player', referee: 'Referee', ball: 'Ball', static: 'Static', ignore: 'Ignore' },
    step2: '2 · Settings', variations: 'Number of variations', cropShare: 'Share of crops %', zoomLow: 'Zoom from % of width', zoomHigh: 'Zoom to % of width', output: 'Output size px', seed: 'Seed', players: 'Max. players per team', referees: 'Max. referees', balls: 'Max. balls', minVisible: 'Min. visible %',
    jersey: 'Vary jersey colours', yolo: 'Also export YOLO', anonymize: 'Anonymize heads', step3: '3 · Review the analysis', analyze: 'Analyze', reanalyze: 'Apply changes and analyze again', correctionsHint: 'Finished masks and plates are kept, so this is much faster than the first time.',
    images: 'images', plate: 'Empty-hall plate', split: { train: 'Training', valid: 'Validation', test: 'Test' }, link: 'Same pairing as', noLink: 'own pairing', moveTo: 'Move to', newSession: 'new session', imagesList: 'Images of this session',
    polygon: { clicked: 'Court clicked', calibration: 'Court from calibration', estimated: 'Court estimated from foot points, please check' }, polygonClick: 'Click the court corners in order.', polygonSave: 'Save court', polygonReset: 'Clear points', polygonEdit: 'Click the court',
    team: (n) => `Team ${n + 1}`, outliers: 'Outliers (not used)', referee: 'Referees', uncertainHint: 'Orange border = uncertain. A click changes the team.', notMovable: 'Half-hidden or cut-off people are pale and are not moved.',
    warnOne: 'Only one session: there are no real images for validation and test. Add more recordings for an honest evaluation.', warnTwo: 'Only two sessions: there is no independent test.', noTrain: 'No session is assigned to training.', dropped: (n) => `${n} instances dropped for a poor mask (see the log).`,
    step4: '4 · Generate', generate: 'Generate dataset', resume: (done) => `${done} images are already finished; the same settings continue from there.`, needVariations: 'Please enter the number of variations.',
    report: (n) => `${n} images generated.`, violationsNone: 'Validator: 0 violations.', violations: 'The validator reports violations:', regenerated: (n) => `${n} violations caught while generating and made again.`, tooSmall: (n) => `${n} labels were smaller than 1.5 px at the output size and were removed.`, realCounts: (t, v, s) => `Real images: ${t} training, ${v} validation, ${s} test.`, preview: 'Preview with labels and team colours',
    step5: '5 · Training', trainingMode: 'Use for training', modeOff: 'No, normal project images', modeMixed: 'Real + synthetic', modeReal: 'Real only, same split (comparison)', realWeight: 'Weight of real images', trainingHint: 'Validation and test always hold real images only. “Train locally” starts the training; the metrics of both modes can be compared directly.',
    hints: { choose: 'Selects the folder with images and COCO annotations.', analyze: 'Creates masks, hall plates, court and teams.', generate: 'Generates the images and then checks them automatically.', polygonSave: 'Saves these points as the court.', polygonReset: 'Clears the points.', polygonEdit: 'Sets the court on the hall plate.', instance: 'Changes the team of this person.' },
  },
  es: {
    title: 'Augmentation Studio', intro: 'Crea imágenes sintéticas de entrenamiento a partir de grabaciones etiquetadas de pista completa, con perspectiva adecuada, etiquetas correctas y nunca más de dos equipos.',
    step1: '1 · Carpeta de origen', choose: 'Seleccionar carpeta de origen', none: 'Imágenes y anotaciones COCO, por ejemplo de «Exportar conjunto de datos».', rolesTitle: 'Función por clase', roles: { player: 'Jugador', referee: 'Árbitro', ball: 'Balón', static: 'Estático', ignore: 'Ignorar' },
    step2: '2 · Ajustes', variations: 'Número de variaciones', cropShare: 'Parte de recortes %', zoomLow: 'Zoom desde % del ancho', zoomHigh: 'Zoom hasta % del ancho', output: 'Tamaño de salida px', seed: 'Semilla', players: 'Máx. jugadores por equipo', referees: 'Máx. árbitros', balls: 'Máx. balones', minVisible: 'Mín. visible %',
    jersey: 'Variar colores de camiseta', yolo: 'Exportar también YOLO', anonymize: 'Anonimizar cabezas', step3: '3 · Revisar el análisis', analyze: 'Analizar', reanalyze: 'Aplicar cambios y volver a analizar', correctionsHint: 'Las máscaras y placas terminadas se conservan, así que es mucho más rápido que la primera vez.',
    images: 'imágenes', plate: 'Placa de pabellón vacío', split: { train: 'Entrenamiento', valid: 'Validación', test: 'Prueba' }, link: 'Mismo enfrentamiento que', noLink: 'enfrentamiento propio', moveTo: 'Mover a', newSession: 'sesión nueva', imagesList: 'Imágenes de esta sesión',
    polygon: { clicked: 'Campo marcado', calibration: 'Campo de la calibración', estimated: 'Campo estimado por los pies, compruébalo' }, polygonClick: 'Haz clic en las esquinas del campo en orden.', polygonSave: 'Guardar campo', polygonReset: 'Borrar puntos', polygonEdit: 'Marcar el campo',
    team: (n) => `Equipo ${n + 1}`, outliers: 'Valores atípicos (no se usan)', referee: 'Árbitros', uncertainHint: 'Borde naranja = dudoso. Un clic cambia el equipo.', notMovable: 'Las personas semiocultas o cortadas aparecen pálidas y no se mueven.',
    warnOne: 'Solo una sesión: no hay imágenes reales para validación y prueba. Añade más grabaciones para una evaluación honesta.', warnTwo: 'Solo dos sesiones: no hay prueba independiente.', noTrain: 'Ninguna sesión está asignada al entrenamiento.', dropped: (n) => `${n} instancias descartadas por mala máscara (ver el registro).`,
    step4: '4 · Generar', generate: 'Generar conjunto de datos', resume: (done) => `${done} imágenes ya están listas; con los mismos ajustes continúa desde ahí.`, needVariations: 'Indica el número de variaciones.',
    report: (n) => `${n} imágenes generadas.`, violationsNone: 'Validador: 0 infracciones.', violations: 'El validador informa de infracciones:', regenerated: (n) => `${n} infracciones detectadas al generar y rehechas.`, tooSmall: (n) => `${n} etiquetas medían menos de 1,5 px en la salida y se eliminaron.`, realCounts: (t, v, s) => `Imágenes reales: ${t} entrenamiento, ${v} validación, ${s} prueba.`, preview: 'Vista previa con etiquetas y colores de equipo',
    step5: '5 · Entrenamiento', trainingMode: 'Usar en el entrenamiento', modeOff: 'No, imágenes normales del proyecto', modeMixed: 'Real + sintético', modeReal: 'Solo real, misma división (comparación)', realWeight: 'Peso de las imágenes reales', trainingHint: 'Validación y prueba solo contienen imágenes reales. «Entrenar localmente» inicia el entrenamiento; las métricas de ambos modos se pueden comparar directamente.',
    hints: { choose: 'Selecciona la carpeta con imágenes y anotaciones COCO.', analyze: 'Crea máscaras, placas, campo y equipos.', generate: 'Genera las imágenes y después las comprueba.', polygonSave: 'Guarda estos puntos como campo.', polygonReset: 'Borra los puntos.', polygonEdit: 'Fija el campo sobre la placa.', instance: 'Cambia el equipo de esta persona.' },
  },
  fr: {
    title: 'Augmentation Studio', intro: 'Crée des images d’entraînement synthétiques à partir de prises de vue annotées du terrain entier, avec une perspective adaptée, des annotations correctes et jamais plus de deux équipes.',
    step1: '1 · Dossier source', choose: 'Sélectionner le dossier source', none: 'Images et annotations COCO, par exemple via « Exporter le jeu de données ».', rolesTitle: 'Rôle par classe', roles: { player: 'Joueur', referee: 'Arbitre', ball: 'Ballon', static: 'Statique', ignore: 'Ignorer' },
    step2: '2 · Réglages', variations: 'Nombre de variations', cropShare: 'Part de recadrages %', zoomLow: 'Zoom de % de la largeur', zoomHigh: 'Zoom à % de la largeur', output: 'Taille de sortie px', seed: 'Graine', players: 'Max. joueurs par équipe', referees: 'Max. arbitres', balls: 'Max. ballons', minVisible: 'Min. visible %',
    jersey: 'Varier les couleurs de maillot', yolo: 'Exporter aussi en YOLO', anonymize: 'Anonymiser les têtes', step3: '3 · Vérifier l’analyse', analyze: 'Analyser', reanalyze: 'Appliquer et analyser à nouveau', correctionsHint: 'Les masques et plaques terminés sont conservés, c’est donc bien plus rapide que la première fois.',
    images: 'images', plate: 'Plaque de salle vide', split: { train: 'Entraînement', valid: 'Validation', test: 'Test' }, link: 'Même affiche que', noLink: 'affiche propre', moveTo: 'Déplacer vers', newSession: 'nouvelle session', imagesList: 'Images de cette session',
    polygon: { clicked: 'Terrain cliqué', calibration: 'Terrain issu de la calibration', estimated: 'Terrain estimé d’après les pieds, à vérifier' }, polygonClick: 'Cliquez les coins du terrain dans l’ordre.', polygonSave: 'Enregistrer le terrain', polygonReset: 'Effacer les points', polygonEdit: 'Cliquer le terrain',
    team: (n) => `Équipe ${n + 1}`, outliers: 'Valeurs aberrantes (non utilisées)', referee: 'Arbitres', uncertainHint: 'Bord orange = incertain. Un clic change l’équipe.', notMovable: 'Les personnes à moitié cachées ou coupées sont pâles et ne sont pas déplacées.',
    warnOne: 'Une seule session : aucune image réelle pour la validation et le test. Ajoutez d’autres prises de vue pour une évaluation honnête.', warnTwo: 'Deux sessions seulement : pas de test indépendant.', noTrain: 'Aucune session n’est affectée à l’entraînement.', dropped: (n) => `${n} instances rejetées pour un mauvais masque (voir le journal).`,
    step4: '4 · Générer', generate: 'Générer le jeu de données', resume: (done) => `${done} images sont déjà terminées ; avec les mêmes réglages, la génération reprend là.`, needVariations: 'Veuillez indiquer le nombre de variations.',
    report: (n) => `${n} images générées.`, violationsNone: 'Validateur : 0 infraction.', violations: 'Le validateur signale des infractions :', regenerated: (n) => `${n} infractions détectées à la génération et refaites.`, tooSmall: (n) => `${n} annotations faisaient moins de 1,5 px en sortie et ont été retirées.`, realCounts: (t, v, s) => `Images réelles : ${t} entraînement, ${v} validation, ${s} test.`, preview: 'Aperçu avec annotations et couleurs d’équipe',
    step5: '5 · Entraînement', trainingMode: 'Utiliser pour l’entraînement', modeOff: 'Non, images normales du projet', modeMixed: 'Réel + synthétique', modeReal: 'Réel seul, même répartition (comparaison)', realWeight: 'Poids des images réelles', trainingHint: 'La validation et le test ne contiennent que des images réelles. « Entraîner localement » lance l’entraînement ; les métriques des deux modes sont directement comparables.',
    hints: { choose: 'Sélectionne le dossier contenant images et annotations COCO.', analyze: 'Crée masques, plaques, terrain et équipes.', generate: 'Génère les images puis les vérifie automatiquement.', polygonSave: 'Enregistre ces points comme terrain.', polygonReset: 'Efface les points.', polygonEdit: 'Définit le terrain sur la plaque.', instance: 'Change l’équipe de cette personne.' },
  },
};

export function AugmentationStudio({ language, api, busy, modelSize, post }: Props) {
  const t = text[language];
  const [state, setState] = useState<StudioState>({ source: null });
  const [config, setConfig] = useState<Config>({});
  const [error, setError] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<{ session: string; points: number[][] } | null>(null);

  const accept = useCallback((next?: StudioState) => {
    if (!next) return;
    setState(next);
    if (next.config) setConfig(next.config);
  }, []);
  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${api}/api/augment/state`);
      accept(await response.json() as StudioState);
    } catch (problem) { setError(String(problem)); }
  }, [api, accept]);
  // A finished job changes what there is to show.
  useEffect(() => { if (!busy) void refresh(); }, [busy, refresh]);

  const run = async (path: string, payload: Record<string, unknown> = {}) => {
    setError(null);
    try { accept(((await post(path, payload)) as { augment?: StudioState }).augment); } catch (problem) { setError(String(problem instanceof Error ? problem.message : problem)); }
  };
  const save = (changes: Config) => { const next = { ...config, ...changes }; setConfig(next); void run('/api/augment/config', { config: changes }); };
  const settings: Config = { ...config, outputSize: config.outputSize ?? MODEL_INPUT[modelSize] ?? 560 };
  const startAnalyze = async () => { await run('/api/augment/config', { config: settings }); await run('/api/augment/analyze'); };
  const startGenerate = async () => {
    if (!settings.variations) { setError(t.needVariations); return; }
    await run('/api/augment/config', { config: settings });
    await run('/api/augment/generate');
  };

  const analysis = state.analysis;
  const limits = { players: config.limits?.players ?? DEFAULTS.players, referees: config.limits?.referees ?? DEFAULTS.referees, balls: config.limits?.balls ?? DEFAULTS.balls };
  const zoom = config.zoom ?? DEFAULTS.zoom;
  const file = (kind: string, name: string) => `${api}/api/augment/file?kind=${kind}&name=${encodeURIComponent(name)}`;
  const number = (label: string, value: number, change: (value: number) => void, min = 0, max?: number, step = 1) => <label className="studio-field">{label}<input type="number" value={value} min={min} max={max} step={step} onChange={(event) => change(Number(event.target.value))} /></label>;
  const toggle = (label: string, value: boolean | undefined, change: (value: boolean) => void) => <label className="cuda-toggle"><input type="checkbox" checked={!!value} onChange={(event) => change(event.target.checked)} />{label}</label>;

  const setLink = (session: Session, target: string) => {
    const links = (config.sessionLinks ?? []).filter((link) => !link.includes(session.id));
    save({ sessionLinks: target ? [...links, [target, session.id]] : links });
  };
  const moveImage = (image: string, target: string) => save({ sessionOverrides: { ...config.sessionOverrides, [image]: target === 'new' ? `${image}#moved` : target } });
  const cycleTeam = (instance: Instance) => {
    const next = instance.team === 0 ? 1 : instance.team === 1 ? -1 : 0;
    setState((current) => current.analysis ? { ...current, analysis: { ...current.analysis, instances: current.analysis.instances.map((item) => item.id === instance.id ? { ...item, team: next, uncertain: false } : item) } } : current);
    save({ teamOverrides: { ...config.teamOverrides, [instance.id]: next } });
  };
  const clickPlate = (event: MouseEvent<HTMLImageElement>, session: Session) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const point = [(event.clientX - rect.left) / rect.width * session.width, (event.clientY - rect.top) / rect.height * session.height];
    setDrawing((current) => current ? { ...current, points: [...current.points, point] } : current);
  };

  const gallery = (instances: Instance[]) => <div className="studio-gallery">{instances.map((instance) => <button data-hint={t.hints.instance} key={instance.id} type="button" className={`studio-thumb${instance.uncertain ? ' uncertain' : ''}${instance.complete ? '' : ' pale'}`} onClick={() => cycleTeam(instance)}><img src={file('thumb', `${instance.id}.jpg`)} alt="" loading="lazy" /></button>)}</div>;

  return <section className="benchmark-workspace studio" aria-label={t.title}>
    <div className="benchmark-heading"><div><h2>{t.title}</h2><p>{t.intro}</p></div></div>
    {error && <p className="studio-error" role="alert">{error}</p>}

    <section className="studio-card"><h3>{t.step1}</h3>
      <button type="button" data-hint={t.hints.choose} className="secondary-button" disabled={busy} onClick={() => void run('/api/augment/select-source')}>{t.choose}</button>
      <code className="studio-path">{state.source ?? t.none}</code>
      {config.roles && <div className="studio-row"><strong>{t.rolesTitle}</strong>{Object.entries(config.roles).map(([name, role]) => <label key={name} className="studio-field">{name}<select value={role} onChange={(event) => save({ roles: { ...config.roles, [name]: event.target.value as Role } })}>{(Object.keys(t.roles) as Role[]).map((item) => <option key={item} value={item}>{t.roles[item]}</option>)}</select></label>)}</div>}
    </section>

    {state.source && <section className="studio-card"><h3>{t.step2}</h3>
      <div className="studio-row">
        {number(t.variations, config.variations ?? 0, (value) => setConfig({ ...config, variations: value }), 0)}
        {number(t.cropShare, Math.round((config.cropShare ?? DEFAULTS.cropShare) * 100), (value) => setConfig({ ...config, cropShare: value / 100 }), 0, 100)}
        {number(t.zoomLow, Math.round(zoom[0] * 100), (value) => setConfig({ ...config, zoom: [value / 100, zoom[1]] }), 5, 100)}
        {number(t.zoomHigh, Math.round(zoom[1] * 100), (value) => setConfig({ ...config, zoom: [zoom[0], value / 100] }), 5, 100)}
        {number(t.output, settings.outputSize ?? 560, (value) => setConfig({ ...config, outputSize: value }), 64, 2048)}
        {number(t.seed, config.seed ?? DEFAULTS.seed, (value) => setConfig({ ...config, seed: value }), 0)}
        {number(t.players, limits.players, (value) => setConfig({ ...config, limits: { ...limits, players: value } }), 1, 30)}
        {number(t.referees, limits.referees, (value) => setConfig({ ...config, limits: { ...limits, referees: value } }), 0, 10)}
        {number(t.balls, limits.balls, (value) => setConfig({ ...config, limits: { ...limits, balls: value } }), 0, 5)}
        {number(t.minVisible, Math.round((config.minVisible ?? DEFAULTS.minVisible) * 100), (value) => setConfig({ ...config, minVisible: value / 100 }), 0, 100)}
      </div>
      <div className="studio-row">
        {toggle(t.jersey, config.jerseyVariation, (value) => setConfig({ ...config, jerseyVariation: value }))}
        {toggle(t.yolo, config.yolo, (value) => setConfig({ ...config, yolo: value }))}
        {toggle(t.anonymize, config.anonymize, (value) => setConfig({ ...config, anonymize: value }))}
      </div>
    </section>}

    {state.source && <section className="studio-card"><h3>{t.step3}</h3>
      <button type="button" data-hint={t.hints.analyze} className="primary-button" disabled={busy} onClick={() => void startAnalyze()}>{analysis ? t.reanalyze : t.analyze}</button>
      {analysis && <small>{t.correctionsHint}</small>}
      {analysis?.splitWarning === 'one-session' && <p className="folder-empty-hint">{t.warnOne}</p>}
      {analysis?.splitWarning === 'two-sessions' && <p className="folder-empty-hint">{t.warnTwo}</p>}
      {analysis && !analysis.sessions.some((session) => session.split === 'train') && <p className="folder-empty-hint">{t.noTrain}</p>}
      {!!analysis?.dropped && <small>{t.dropped(analysis.dropped)}</small>}
      {!!analysis?.log.length && <details className="studio-log"><summary>Log ({analysis.log.length})</summary><pre>{analysis.log.join('\n')}</pre></details>}
      {analysis && <><small>{t.uncertainHint} {t.notMovable}</small>{analysis.sessions.map((session) => {
        const people = analysis.instances.filter((item) => item.session === session.id);
        const players = people.filter((item) => item.role === 'player');
        const others = analysis.sessions.filter((item) => item.id !== session.id);
        const editing = drawing?.session === session.id;
        return <article key={session.id} className="studio-session">
          <header><strong>{session.id.split('|')[0]}</strong><span>{session.images.length} {t.images} · {session.width}×{session.height}</span>
            <select aria-label={t.split.train} value={session.split} onChange={(event) => save({ splitOverrides: { ...config.splitOverrides, [session.id]: event.target.value as Split } })}>{(['train', 'valid', 'test'] as Split[]).map((item) => <option key={item} value={item}>{t.split[item]}</option>)}</select>
            <label className="studio-field">{t.link}<select value={others.find((item) => item.link === session.link)?.id ?? ''} onChange={(event) => setLink(session, event.target.value)}><option value="">{t.noLink}</option>{others.map((item) => <option key={item.id} value={item.id}>{item.id.split('|')[0]}</option>)}</select></label>
          </header>
          {session.plateFile && <div className="studio-plate"><small>{t.plate} · {t.polygon[session.polygonSource]}{editing ? ` · ${t.polygonClick}` : ''}</small>
            <div className="studio-plate-image"><img src={file('plate', session.plateFile)} alt="" onClick={(event) => clickPlate(event, session)} className={editing ? 'drawing' : ''} />
              <svg viewBox={`0 0 ${session.width} ${session.height}`} aria-hidden="true"><polygon points={(editing ? drawing!.points : session.polygon).map((point) => point.join(',')).join(' ')} /></svg></div>
            <div className="studio-row">{editing
              ? <><button type="button" data-hint={t.hints.polygonSave} className="secondary-button" disabled={drawing!.points.length < 3} onClick={() => { save({ polygons: { ...config.polygons, [session.id]: drawing!.points } }); setDrawing(null); }}>{t.polygonSave}</button><button type="button" data-hint={t.hints.polygonReset} className="secondary-button" onClick={() => setDrawing({ session: session.id, points: [] })}>{t.polygonReset}</button></>
              : <button type="button" data-hint={t.hints.polygonEdit} className="secondary-button" onClick={() => setDrawing({ session: session.id, points: [] })}>{t.polygonEdit}</button>}</div></div>}
          <div className="studio-teams">
            {[0, 1].map((team) => <div key={team}><strong>{t.team(team)}</strong>{gallery(players.filter((item) => item.team === team))}</div>)}
            {players.some((item) => item.team === -1) && <div><strong>{t.outliers}</strong>{gallery(players.filter((item) => item.team === -1))}</div>}
          </div>
          <details><summary>{t.imagesList}</summary><ul className="studio-images">{session.images.map((image) => <li key={image}><code>{image}</code><label>{t.moveTo} <select value="" onChange={(event) => { if (event.target.value) moveImage(image, event.target.value); }}><option value="">–</option>{others.map((item) => <option key={item.id} value={item.id}>{item.id.split('|')[0]}</option>)}<option value="new">{t.newSession}</option></select></label></li>)}</ul></details>
        </article>;
      })}</>}
    </section>}

    {analysis && <section className="studio-card"><h3>{t.step4}</h3>
      <button type="button" data-hint={t.hints.generate} className="primary-button" disabled={busy} onClick={() => void startGenerate()}>{t.generate}</button>
      {!state.report && !!state.generated && <small>{t.resume(state.generated)}</small>}
      {state.report && <div className="studio-report">
        <strong>{t.report(state.report.images)}</strong>
        {Object.keys(state.report.violations).length === 0 ? <span>{t.violationsNone}</span> : <span className="studio-error">{t.violations} {JSON.stringify(state.report.violations)}</span>}
        {Object.keys(state.report.rejectedWhileGenerating).length > 0 && <span>{t.regenerated(Object.values(state.report.rejectedWhileGenerating).reduce((sum, value) => sum + value, 0))}</span>}
        {state.report.labelsTooSmall > 0 && <span>{t.tooSmall(state.report.labelsTooSmall)}</span>}
        <span>{t.realCounts(state.report.real.train, state.report.real.valid, state.report.real.test)}</span>
      </div>}
      {!!state.previews?.length && <><h4>{t.preview}</h4><div className="studio-previews">{state.previews.map((name) => <a key={name} href={file('preview', name)} target="_blank" rel="noreferrer"><img src={file('preview', name)} alt={name} loading="lazy" /></a>)}</div></>}
    </section>}

    {state.report && <section className="studio-card"><h3>{t.step5}</h3>
      <div className="studio-row">
        <label className="studio-field">{t.trainingMode}<select value={state.training?.mode ?? 'off'} onChange={(event) => void run('/api/augment/training', { mode: event.target.value, realWeight: state.training?.realWeight ?? 3 })}><option value="off">{t.modeOff}</option><option value="mixed">{t.modeMixed}</option><option value="real-only">{t.modeReal}</option></select></label>
        {state.training && number(t.realWeight, state.training.realWeight, (value) => void run('/api/augment/training', { mode: state.training?.mode, realWeight: value }), 1, 20)}
      </div>
      <small>{t.trainingHint}</small>
    </section>}
  </section>;
}
