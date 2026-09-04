'use client';

import { useEffect, useRef, useState } from 'react';
import BenchmarkPanel from './BenchmarkPanel';

type Platform = 'mac' | 'windows' | 'linux';
type Language = 'de' | 'en' | 'es' | 'fr';
type Sport = 'basketball' | 'football' | 'futsal' | 'handball' | 'hockey' | 'rugby' | 'lacrosse' | 'american_football';
type ToolMode = 'select' | 'draw' | 'pan';
type Handle = 'nw' | 'ne' | 'sw' | 'se';
type ModelSize = 'nano' | 'small';

type LocalAnnotation = { id: string; category: string; x: number; y: number; width: number; height: number; source: string; confidence?: number | null };
type LocalFrame = { id: string; relativePath: string; videoName: string; timestamp: number; width: number; height: number; annotations: LocalAnnotation[]; reviewStatus?: 'candidate' | 'reviewed' };
type TrainingResult = { completedAt: string; model: string; device: string; epochsRequested: number; frameCount: number; annotatedFrames: number; annotationCount: number; classes: string[]; splits: Record<string, number>; independentTest: boolean; checkpoint?: string | null; continuedFrom?: string | null; validationMetrics?: Record<string, number> };
type BenchmarkMetrics = { qualityScore: number; mAP50: number; precision: number; recall: number; f1: number; meanIoU: number; truePositives: number; falsePositives: number; falseNegatives: number };
type BenchmarkReport = { createdAt?: string; runID?: string; datasetID?: string; matchesGroundTruth?: boolean; frameCount: number; annotationCount: number; modelCount: number; successfulModelCount: number; threshold: number; device: string; rankingMethod: string; results: Array<{ rank?: number | null; packageID: string; modelSize?: string; status: string; metrics?: BenchmarkMetrics; meanLatencyMs?: number; error?: string }> };
type WorkerStatus = {
  connected: boolean; selectedFolder: string | null; folderName: string | null; sport?: Sport | null; operation: string; progress: number;
  framesPerVideo?: number;
  message: string; busy: boolean; error: string | null; frames: LocalFrame[]; log: string;
  stats?: { frameCount: number; candidateCount?: number; annotatedFrames: number; annotationCount: number; automaticCount: number; classes: string[] };
  hardware?: { machine: string; cpuCores: number; memoryGB: number | null; accelerator: string };
  lastTraining?: TrainingResult | null;
  trainingHistory?: TrainingResult[];
  modelLibrary?: { packages: Array<{ packageID: string; displayName?: string; createdAt?: string; sport: Sport; modelSize: ModelSize; classes: string[]; description?: string; source?: string; statistics?: Record<string, number>; validationMetrics?: Record<string, number>; testMetrics?: Record<string, number>; validationScore?: number; testScore?: number; isActive?: boolean; isBest?: boolean }>; activePackageID: string | null; bestPackageID?: string | null };
  benchmark?: { groundTruth?: { createdAt?: string; datasetID?: string; sport?: string; frameCount: number; annotationCount: number; classes: string[] } | null; latest?: BenchmarkReport | null };
};
type Gesture =
  | { type: 'draw'; pointerId: number; startX: number; startY: number; x: number; y: number; width: number; height: number }
  | { type: 'move'; pointerId: number; annotation: LocalAnnotation; startX: number; startY: number; current: LocalAnnotation }
  | { type: 'resize'; pointerId: number; annotation: LocalAnnotation; handle: Handle; current: LocalAnnotation }
  | { type: 'pan'; pointerId: number; startClientX: number; startClientY: number; startPanX: number; startPanY: number };

const DEFAULT_API = 'http://127.0.0.1:8766';
const cloneAnnotations = (items: LocalAnnotation[]) => items.map((item) => ({ ...item }));
const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));
const frameCopy = {
  de: { count: 'Bilder je Video', countHint: 'Jedes Video liefert bis zu dieser Anzahl', remove: 'Bild aus Training entfernen', confirm: 'Dieses extrahierte Bild aus dem Training entfernen? Das Quellvideo bleibt unverändert.' },
  en: { count: 'Images per video', countHint: 'Each video contributes up to this many images', remove: 'Remove image from training', confirm: 'Remove this extracted image from training? The source video remains unchanged.' },
  es: { count: 'Imágenes por vídeo', countHint: 'Cada vídeo aporta como máximo esta cantidad', remove: 'Quitar imagen del entrenamiento', confirm: '¿Quitar esta imagen extraída del entrenamiento? El vídeo original no se modifica.' },
  fr: { count: 'Images par vidéo', countHint: 'Chaque vidéo fournit au maximum ce nombre d’images', remove: 'Retirer l’image de l’entraînement', confirm: 'Retirer cette image extraite de l’entraînement ? La vidéo source reste inchangée.' },
};
const activeCopy = {
  de: { title: 'Datensatz erweitern', start: 'Neue Videos prüfen', detail: 'Getrennter Ordner · nutzt „Bilder je Video“ (max. 500) · alles bleibt lokal.', queue: 'Prüfkandidaten', ball: 'Ball', noBall: 'Kein Ball', correct: 'Box korrigieren', skip: 'Überspringen', hint: 'Nur bestätigte Bilder gelangen ins Training. Das Quellvideo bleibt unverändert.', needsBox: 'Falls der Ball stimmt, Box prüfen oder korrigieren.' },
  en: { title: 'Expand dataset', start: 'Review new videos', detail: 'Separate folder · uses “Images per video” (max. 500) · entirely local.', queue: 'Review candidates', ball: 'Ball', noBall: 'No ball', correct: 'Correct box', skip: 'Skip', hint: 'Only confirmed images enter training. Source videos remain unchanged.', needsBox: 'If this is a ball, review or correct its box.' },
  es: { title: 'Ampliar conjunto', start: 'Revisar vídeos nuevos', detail: 'Carpeta separada · usa «Imágenes por vídeo» (máx. 500) · todo local.', queue: 'Candidatos', ball: 'Balón', noBall: 'No es balón', correct: 'Corregir cuadro', skip: 'Omitir', hint: 'Solo las imágenes confirmadas pasan al entrenamiento. Los vídeos originales no cambian.', needsBox: 'Si es un balón, revisa o corrige el cuadro.' },
  fr: { title: 'Étendre le jeu', start: 'Vérifier de nouvelles vidéos', detail: 'Dossier séparé · utilise « Images par vidéo » (max. 500) · tout reste local.', queue: 'Candidats', ball: 'Ballon', noBall: 'Pas de ballon', correct: 'Corriger la boîte', skip: 'Ignorer', hint: 'Seules les images confirmées entrent dans l’entraînement. Les vidéos sources restent inchangées.', needsBox: 'Si c’est un ballon, vérifiez ou corrigez sa boîte.' },
};

const copy = {
  de: {
    preview: 'Lokales Training auf diesem Rechner', note: 'Videos, Frames und Modelle bleiben auf diesem Rechner', sport: 'Sportart', videos: 'Videos', choose: 'Videoordner auswählen', analyze: 'Videos lokal vorbereiten', refresh: 'Frames sicher aktualisieren', analyzing: 'Extrahiere echte Frames …', images: 'Trainingsbilder', annotate: 'Klasse', ball: 'Ball', puck: 'Puck', player: 'Spieler', goalkeeper: 'Torwart', referee: 'Schiedsrichter', goal: 'Tor', goalpost: 'Torpfosten', hoop: 'Korb', football: 'Fußball', americanFootball: 'American Football', select: 'Auswählen', draw: 'Neue Box', pan: 'Verschieben', undo: 'Rückgängig', redo: 'Wiederholen', remove: 'Löschen', accept: 'Auto übernehmen', reject: 'Auto verwerfen', reset: 'Ansicht zurücksetzen', drag: 'Boxen anklicken, verschieben oder an den Ecken ändern · Pinch/Mausrad zoomt', improve: 'Modell verbessern', setup: 'ML einrichten', auto: 'Automatisch markieren', train: 'Lokal trainieren', export: 'Apple-Modell', share: 'Paket erstellen', importModel: 'Modell importieren', activeModel: 'Aktives Austauschmodell', noModel: 'Kein Austauschmodell aktiv', trustWarning: 'Nur Modellpakete aus einer vertrauenswürdigen Quelle importieren.', privacy: 'Lokal und privat', noFolder: 'Noch kein Ordner gewählt', folderReady: 'Ordner verbunden. Markierungen werden lokal gespeichert und vor Änderungen gesichert.', emptyFrames: 'Ordner wählen und Videos lokal vorbereiten.', localReady: 'lokal vorbereitet', offline: 'Lokaler Worker nicht erreichbar. Reco Trainer mit dem Starter neu öffnen.', confidence: 'Mindest-Sicherheit', marked: 'markierte Frames', boxes: 'Boxen', pending: 'Auto offen', noTraining: 'Noch kein abgeschlossenes Training', testGood: 'Unabhängiger Testsatz vorhanden', testWeak: 'Für einen unabhängigen Test werden mindestens drei Videos benötigt.', previewOnly: 'Oberflächenvorschau', validation: 'Validierung', help: 'Ablauf erklären', walkthroughTitle: 'So verbesserst du dein Reco-Modell', back: 'Zurück', next: 'Weiter', done: 'Loslegen', close: 'Schließen', step: 'Schritt',
    walkthrough: [
      ['Sportart wählen', 'Wähle zuerst die Sportart. Jede Sportart besitzt passende Objektklassen und eine getrennte Modellbibliothek.'],
      ['Lokalen Videoordner verbinden', 'Wähle den Ordner mit deinen Videos. Die Aufnahmen bleiben auf diesem Rechner und werden nicht hochgeladen.'],
      ['Trainingsbilder vorbereiten', 'Reco Trainer extrahiert lokal einzelne Frames und legt das Trainingsprojekt im gewählten Ordner an.'],
      ['Automatisch markieren und prüfen', 'Wähle eine Klasse und starte die automatische Markierung. Korrigiere Boxen, lösche falsche Treffer und bestätige nur geprüfte Markierungen.'],
      ['Lokal trainieren', 'Richte ML einmalig ein und trainiere anschließend das sportartspezifische Modell auf deiner lokalen Hardware.'],
      ['Mit neuen Videos erweitern', 'Wähle „Neue Videos prüfen“ und einen getrennten Ordner. Bestätige Ball oder Kein Ball, korrigiere Boxen oder überspringe. Nur geprüfte Bilder werden trainiert.'],
      ['Exportieren oder austauschen', 'Exportiere ein Gerätemodell oder erstelle ein .recomodel-Paket. Das Paket enthält nur Gewichte und zusammengefasste Metadaten – keine Videos oder Frames.'],
      ['Modelle objektiv vergleichen', 'Lege geprüfte Markierungen als richtige Antworten fest und teste alle kompatiblen lokalen Modelle auf denselben Bildern. Das Ranking kombiniert mAP@0.50 und F1.'],
    ],
  },
  en: {
    preview: 'Local training on this computer', note: 'Videos, frames, and models remain on this computer', sport: 'Sport', videos: 'Videos', choose: 'Select video folder', analyze: 'Prepare videos locally', refresh: 'Safely refresh frames', analyzing: 'Extracting real frames …', images: 'Training images', annotate: 'Class', ball: 'Ball', puck: 'Puck', player: 'Player', goalkeeper: 'Goalkeeper', referee: 'Referee', goal: 'Goal', goalpost: 'Goalpost', hoop: 'Hoop', football: 'Football', americanFootball: 'American Football', select: 'Select', draw: 'New box', pan: 'Pan', undo: 'Undo', redo: 'Redo', remove: 'Delete', accept: 'Accept auto', reject: 'Reject auto', reset: 'Reset view', drag: 'Select, move, or resize boxes at their corners · pinch/wheel to zoom', improve: 'Improve model', setup: 'Set up ML', auto: 'Auto-label', train: 'Train locally', export: 'Apple model', share: 'Create package', importModel: 'Import model', activeModel: 'Active exchange model', noModel: 'No exchange model active', trustWarning: 'Import model packages only from a trusted source.', privacy: 'Local and private', noFolder: 'No folder selected', folderReady: 'Folder connected. Annotations are stored locally and backed up before changes.', emptyFrames: 'Select a folder and prepare videos locally.', localReady: 'prepared locally', offline: 'Local worker unavailable. Reopen Reco Trainer with its starter.', confidence: 'Minimum confidence', marked: 'annotated frames', boxes: 'boxes', pending: 'auto pending', noTraining: 'No completed training yet', testGood: 'Independent test set available', testWeak: 'At least three videos are needed for an independent test.', previewOnly: 'UI preview', validation: 'Validation', help: 'Show workflow', walkthroughTitle: 'How to improve your Reco model', back: 'Back', next: 'Next', done: 'Get started', close: 'Close', step: 'Step',
    walkthrough: [
      ['Choose a sport', 'Choose the sport first. Each sport has suitable object classes and a separate model library.'],
      ['Connect a local video folder', 'Select the folder containing your videos. Recordings stay on this computer and are never uploaded.'],
      ['Prepare training images', 'Reco Trainer extracts individual frames locally and creates the training project inside the selected folder.'],
      ['Auto-label and review', 'Choose a class and run auto-labeling. Correct boxes, delete false detections, and accept only reviewed annotations.'],
      ['Train locally', 'Set up ML once, then train the sport-specific model on your local hardware.'],
      ['Expand with new videos', 'Choose “Review new videos” and a separate folder. Confirm Ball or No ball, correct boxes, or skip. Only reviewed images are trained.'],
      ['Export or exchange', 'Export a device model or create a .recomodel package. The package contains only weights and aggregate metadata—no videos or frames.'],
      ['Compare models objectively', 'Freeze reviewed annotations as the correct answers and test every compatible local model on the same images. The ranking combines mAP@0.50 and F1.'],
    ],
  },
  es: {
    preview: 'Entrenamiento local en este equipo', note: 'Los vídeos, fotogramas y modelos permanecen en este equipo', sport: 'Deporte', videos: 'Vídeos', choose: 'Seleccionar carpeta de vídeos', analyze: 'Preparar vídeos localmente', refresh: 'Actualizar fotogramas de forma segura', analyzing: 'Extrayendo fotogramas reales …', images: 'Imágenes de entrenamiento', annotate: 'Clase', ball: 'Balón', puck: 'Disco', player: 'Jugador', goalkeeper: 'Portero', referee: 'Árbitro', goal: 'Portería', goalpost: 'Postes', hoop: 'Canasta', football: 'Fútbol', americanFootball: 'Fútbol americano', select: 'Seleccionar', draw: 'Nuevo cuadro', pan: 'Mover vista', undo: 'Deshacer', redo: 'Rehacer', remove: 'Eliminar', accept: 'Aceptar automático', reject: 'Rechazar automático', reset: 'Restablecer vista', drag: 'Selecciona, mueve o cambia el tamaño de los cuadros · pellizca/rueda para ampliar', improve: 'Mejorar modelo', setup: 'Configurar ML', auto: 'Marcar automáticamente', train: 'Entrenar localmente', export: 'Modelo Apple', share: 'Crear paquete', importModel: 'Importar modelo', activeModel: 'Modelo de intercambio activo', noModel: 'Ningún modelo de intercambio activo', trustWarning: 'Importa paquetes de modelos solo de una fuente de confianza.', privacy: 'Local y privado', noFolder: 'Ninguna carpeta seleccionada', folderReady: 'Carpeta conectada. Las anotaciones se guardan localmente y se respaldan antes de los cambios.', emptyFrames: 'Selecciona una carpeta y prepara los vídeos localmente.', localReady: 'preparado localmente', offline: 'El proceso local no está disponible. Abre Reco Trainer de nuevo con su iniciador.', confidence: 'Confianza mínima', marked: 'fotogramas marcados', boxes: 'cuadros', pending: 'automáticos pendientes', noTraining: 'Aún no hay entrenamiento finalizado', testGood: 'Conjunto de prueba independiente disponible', testWeak: 'Se necesitan al menos tres vídeos para una prueba independiente.', previewOnly: 'Vista previa de la interfaz', validation: 'Validación', help: 'Mostrar el flujo', walkthroughTitle: 'Cómo mejorar tu modelo Reco', back: 'Atrás', next: 'Siguiente', done: 'Empezar', close: 'Cerrar', step: 'Paso',
    walkthrough: [
      ['Elige un deporte', 'Elige primero el deporte. Cada deporte tiene clases de objetos adecuadas y una biblioteca de modelos independiente.'],
      ['Conecta una carpeta local', 'Selecciona la carpeta que contiene tus vídeos. Las grabaciones permanecen en este equipo y nunca se suben.'],
      ['Prepara las imágenes', 'Reco Trainer extrae fotogramas localmente y crea el proyecto de entrenamiento dentro de la carpeta seleccionada.'],
      ['Marca automáticamente y revisa', 'Elige una clase y ejecuta el marcado automático. Corrige cuadros, elimina detecciones falsas y acepta solo las marcas revisadas.'],
      ['Entrena localmente', 'Configura ML una vez y entrena el modelo específico del deporte con tu hardware local.'],
      ['Amplía con vídeos nuevos', 'Elige «Revisar vídeos nuevos» y una carpeta separada. Confirma Balón o No es balón, corrige cuadros u omite. Solo se entrenan imágenes revisadas.'],
      ['Exporta o intercambia', 'Exporta un modelo para el dispositivo o crea un paquete .recomodel. Solo contiene pesos y metadatos agregados; nunca vídeos ni fotogramas.'],
      ['Compara modelos objetivamente', 'Fija las anotaciones revisadas como respuestas correctas y prueba todos los modelos locales compatibles con las mismas imágenes. La clasificación combina mAP@0.50 y F1.'],
    ],
  },
  fr: {
    preview: 'Entraînement local sur cet ordinateur', note: 'Les vidéos, images et modèles restent sur cet ordinateur', sport: 'Sport', videos: 'Vidéos', choose: 'Sélectionner le dossier vidéo', analyze: 'Préparer les vidéos localement', refresh: 'Actualiser les images en sécurité', analyzing: 'Extraction des images réelles …', images: 'Images d’entraînement', annotate: 'Classe', ball: 'Ballon', puck: 'Palet', player: 'Joueur', goalkeeper: 'Gardien', referee: 'Arbitre', goal: 'But', goalpost: 'Poteaux', hoop: 'Panier', football: 'Football', americanFootball: 'Football américain', select: 'Sélectionner', draw: 'Nouvelle boîte', pan: 'Déplacer la vue', undo: 'Annuler', redo: 'Rétablir', remove: 'Supprimer', accept: 'Accepter auto', reject: 'Refuser auto', reset: 'Réinitialiser la vue', drag: 'Sélectionnez, déplacez ou redimensionnez les boîtes · pincez/molette pour zoomer', improve: 'Améliorer le modèle', setup: 'Configurer le ML', auto: 'Marquage automatique', train: 'Entraîner localement', export: 'Modèle Apple', share: 'Créer le paquet', importModel: 'Importer un modèle', activeModel: 'Modèle d’échange actif', noModel: 'Aucun modèle d’échange actif', trustWarning: 'Importez uniquement des paquets provenant d’une source fiable.', privacy: 'Local et privé', noFolder: 'Aucun dossier sélectionné', folderReady: 'Dossier connecté. Les annotations sont stockées localement et sauvegardées avant modification.', emptyFrames: 'Sélectionnez un dossier et préparez les vidéos localement.', localReady: 'préparé localement', offline: 'Le processus local est indisponible. Rouvrez Reco Trainer avec son lanceur.', confidence: 'Confiance minimale', marked: 'images annotées', boxes: 'boîtes', pending: 'autos en attente', noTraining: 'Aucun entraînement terminé', testGood: 'Jeu de test indépendant disponible', testWeak: 'Au moins trois vidéos sont nécessaires pour un test indépendant.', previewOnly: 'Aperçu de l’interface', validation: 'Validation', help: 'Afficher le parcours', walkthroughTitle: 'Comment améliorer votre modèle Reco', back: 'Retour', next: 'Suivant', done: 'Commencer', close: 'Fermer', step: 'Étape',
    walkthrough: [
      ['Choisissez un sport', 'Choisissez d’abord le sport. Chaque sport possède des classes adaptées et une bibliothèque de modèles distincte.'],
      ['Connectez un dossier local', 'Sélectionnez le dossier contenant vos vidéos. Les enregistrements restent sur cet ordinateur et ne sont jamais envoyés.'],
      ['Préparez les images', 'Reco Trainer extrait localement des images et crée le projet d’entraînement dans le dossier sélectionné.'],
      ['Marquez automatiquement et vérifiez', 'Choisissez une classe et lancez le marquage automatique. Corrigez les boîtes, supprimez les détections erronées et n’acceptez que les annotations vérifiées.'],
      ['Entraînez localement', 'Configurez le ML une fois, puis entraînez le modèle propre au sport sur votre matériel local.'],
      ['Étendez avec de nouvelles vidéos', 'Choisissez « Vérifier de nouvelles vidéos » et un dossier séparé. Confirmez Ballon ou Pas de ballon, corrigez ou ignorez. Seules les images vérifiées sont entraînées.'],
      ['Exportez ou échangez', 'Exportez un modèle pour l’appareil ou créez un paquet .recomodel. Il contient uniquement les poids et des métadonnées agrégées, sans vidéos ni images.'],
      ['Comparez objectivement les modèles', 'Figez les annotations vérifiées comme bonnes réponses et testez tous les modèles locaux compatibles sur les mêmes images. Le classement combine mAP@0.50 et F1.'],
    ],
  },
};

export default function Home() {
  const [platform, setPlatform] = useState<Platform>('mac');
  const [language, setLanguage] = useState<Language>('de');
  const [sport, setSport] = useState<Sport>('basketball');
  const [category, setCategory] = useState('ball');
  const [mode, setMode] = useState<ToolMode>('draw');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [threshold, setThreshold] = useState(.35);
  const [framesPerVideo, setFramesPerVideo] = useState(240);
  const [modelSize, setModelSize] = useState<ModelSize>('nano');
  const [workspaceView, setWorkspaceView] = useState<'training' | 'benchmark'>('training');
  const [walkthroughOpen, setWalkthroughOpen] = useState(false);
  const [walkthroughStep, setWalkthroughStep] = useState(0);
  const [languagePromptOpen, setLanguagePromptOpen] = useState(false);
  const [worker, setWorker] = useState<WorkerStatus | null>(null);
  const [workerOnline, setWorkerOnline] = useState(false);
  const [api, setAPI] = useState(DEFAULT_API);
  const [selectedFrameID, setSelectedFrameID] = useState<string | null>(null);
  const [selectedAnnotationID, setSelectedAnnotationID] = useState<string | null>(null);
  const [gesture, setGesture] = useState<Gesture | null>(null);
  const [history, setHistory] = useState<LocalAnnotation[][]>([]);
  const [future, setFuture] = useState<LocalAnnotation[][]>([]);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ distance: number; centerX: number; centerY: number; zoom: number; panX: number; panY: number } | null>(null);

  const t = copy[language];
  const ft = frameCopy[language];
  const at = activeCopy[language];
  const frames = worker?.frames ?? [];
  const candidates = frames.filter((frame) => frame.reviewStatus === 'candidate').sort((left, right) => {
    const confidence = (frame: LocalFrame) => frame.annotations.find((item) => item.category === category)?.confidence;
    const a = confidence(left), b = confidence(right);
    if (a == null && b != null) return 1;
    if (a != null && b == null) return -1;
    return Math.abs((a ?? 1) - .35) - Math.abs((b ?? 1) - .35);
  });
  const framesReady = frames.length > 0;
  const selectedFrameIndex = Math.max(0, frames.findIndex((frame) => frame.id === selectedFrameID));
  const selectedFrame = frames.find((frame) => frame.id === selectedFrameID) ?? frames[0];
  const selectedAnnotation = selectedFrame?.annotations.find((annotation) => annotation.id === selectedAnnotationID) ?? null;
  const folderName = worker?.folderName ?? '';
  const isWorking = worker?.busy ?? false;
  const categories = sport === 'basketball' ? ['ball', 'player', 'referee', 'hoop'] : sport === 'hockey' ? ['puck', 'player', 'goalkeeper', 'referee', 'goal'] : sport === 'rugby' || sport === 'american_football' ? ['ball', 'player', 'referee', 'goalpost'] : ['ball', 'player', 'goalkeeper', 'referee', 'goal'];
  const sportName = sport === 'basketball' ? 'Basketball' : sport === 'football' ? t.football : sport === 'futsal' ? 'Futsal' : sport === 'handball' ? 'Handball' : sport === 'hockey' ? 'Hockey' : sport === 'rugby' ? 'Rugby' : sport === 'lacrosse' ? 'Lacrosse' : t.americanFootball;
  const categoryName = (value: string) => ({ ball: t.ball, puck: t.puck, player: t.player, goalkeeper: t.goalkeeper, referee: t.referee, goal: t.goal, goalpost: t.goalpost, hoop: t.hoop }[value] ?? value);
  const activeModel = worker?.modelLibrary?.packages.find((item) => item.packageID === worker.modelLibrary?.activePackageID);
  const modelUi = language === 'de'
    ? { title: 'Modellverwaltung', active: 'AKTIV', best: 'BESTES', activate: 'Aktivieren', rename: 'Umbenennen', remove: 'Löschen', prompt: 'Neuer Modellname', confirm: 'Dieses archivierte Modell dauerhaft löschen?' }
    : language === 'es'
      ? { title: 'Biblioteca de modelos', active: 'ACTIVO', best: 'MEJOR', activate: 'Activar', rename: 'Renombrar', remove: 'Eliminar', prompt: 'Nuevo nombre del modelo', confirm: '¿Eliminar permanentemente este modelo archivado?' }
      : language === 'fr'
        ? { title: 'Bibliothèque de modèles', active: 'ACTIF', best: 'MEILLEUR', activate: 'Activer', rename: 'Renommer', remove: 'Supprimer', prompt: 'Nouveau nom du modèle', confirm: 'Supprimer définitivement ce modèle archivé ?' }
        : { title: 'Model library', active: 'ACTIVE', best: 'BEST', activate: 'Activate', rename: 'Rename', remove: 'Delete', prompt: 'New model name', confirm: 'Permanently delete this archived model?' };

  useEffect(() => {
    if (candidates.length && selectedFrame?.reviewStatus !== 'candidate' && worker?.operation === 'ready') {
      setSelectedFrameID(candidates[0].id);
    }
  }, [candidates.length, worker?.operation]);

  useEffect(() => {
    const candidate = new URLSearchParams(window.location.search).get('worker');
    if (candidate && /^http:\/\/127\.0\.0\.1:\d{2,5}$/.test(candidate)) setAPI(candidate);
    const savedLanguage = window.localStorage.getItem('reco-language-v1') as Language | null;
    if (savedLanguage && ['de', 'en', 'es', 'fr'].includes(savedLanguage)) {
      setLanguage(savedLanguage);
      if (window.localStorage.getItem('reco-walkthrough-complete-v2') !== 'yes') setWalkthroughOpen(true);
    } else {
      setLanguagePromptOpen(true);
    }
  }, []);

  function chooseLanguage(nextLanguage: Language) {
    setLanguage(nextLanguage);
    window.localStorage.setItem('reco-language-v1', nextLanguage);
    setLanguagePromptOpen(false);
    setWalkthroughStep(0);
    setWalkthroughOpen(true);
  }

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch(`${api}/api/status`, { cache: 'no-store' });
        if (!response.ok) throw new Error('worker offline');
        const status = await response.json() as WorkerStatus;
        if (!active) return;
        setWorker(status); setWorkerOnline(true);
        setSelectedFrameID((current) => current && status.frames.some((frame) => frame.id === current) ? current : status.frames[0]?.id ?? null);
      } catch { if (active) setWorkerOnline(false); }
    };
    void poll();
    const timer = window.setInterval(poll, 900);
    return () => { active = false; window.clearInterval(timer); };
  }, [api]);

  useEffect(() => {
    if (activeModel?.modelSize) setModelSize(activeModel.modelSize);
  }, [activeModel?.packageID, activeModel?.modelSize]);

  useEffect(() => { setSelectedAnnotationID(null); setGesture(null); setHistory([]); setFuture([]); }, [selectedFrameID]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches('input, select, textarea, button, [contenteditable="true"]')) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) void redo(); else void undo(); }
      else if ((event.key === 'Delete' || event.key === 'Backspace') && selectedAnnotationID) { event.preventDefault(); void removeSelected(); }
      else if (event.key === 'ArrowRight' && frames[selectedFrameIndex + 1]) { event.preventDefault(); chooseFrame(frames[selectedFrameIndex + 1].id); }
      else if (event.key === 'ArrowLeft' && frames[selectedFrameIndex - 1]) { event.preventDefault(); chooseFrame(frames[selectedFrameIndex - 1].id); }
      else if (event.key === 'Escape') { setSelectedAnnotationID(null); setGesture(null); }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  async function post(path: string, payload: Record<string, unknown> = {}) {
    const response = await fetch(`${api}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Local worker failed');
    if (result.status) setWorker(result.status as WorkerStatus);
    return result;
  }

  function updateFrameLocally(annotations: LocalAnnotation[]) {
    if (!selectedFrame) return;
    setWorker((current) => current ? { ...current, frames: current.frames.map((frame) => frame.id === selectedFrame.id ? { ...frame, annotations: cloneAnnotations(annotations) } : frame) } : current);
  }

  async function commitAnnotations(annotations: LocalAnnotation[], remember = true) {
    if (!selectedFrame) return;
    const before = cloneAnnotations(selectedFrame.annotations);
    if (remember) { setHistory((items) => [...items, before].slice(-60)); setFuture([]); }
    updateFrameLocally(annotations);
    try { await post('/api/annotations', { frameId: selectedFrame.id, annotations }); }
    catch (error) { updateFrameLocally(before); setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }

  async function chooseFolder() {
    try {
      const result = await post('/api/select-folder');
      if (result.status?.sport) { setSport(result.status.sport as Sport); setCategory(result.status.sport === 'hockey' ? 'puck' : 'ball'); }
      if (result.status?.framesPerVideo) setFramesPerVideo(result.status.framesPerVideo as number);
    } catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }
  async function preparePreview() { if (!folderName || isWorking) return; try { await post('/api/prepare', { sport, framesPerVideo }); } catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); } }
  async function removeFrame() {
    if (!selectedFrame || isWorking || !window.confirm(ft.confirm)) return;
    const next = frames[selectedFrameIndex + 1] ?? frames[selectedFrameIndex - 1];
    try {
      await post('/api/remove-frame', { frameId: selectedFrame.id });
      setSelectedFrameID(next?.id ?? null);
      setSelectedAnnotationID(null); setHistory([]); setFuture([]);
    } catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }
  async function runML(path: string) { try { await post(path, { model: modelSize, epochs: 20, language, category, threshold }); } catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); } }
  async function startActiveLearning() {
    try { await post('/api/active-learning', { model: modelSize, language, category, threshold: .12, framesPerVideo }); }
    catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }
  async function reviewCandidate(decision: 'ball' | 'no-ball') {
    if (!selectedFrame || selectedFrame.reviewStatus !== 'candidate') return;
    const next = candidates.find((frame) => frame.id !== selectedFrame.id);
    try {
      await post('/api/review-candidate', { frameId: selectedFrame.id, decision, category });
      setSelectedFrameID(next?.id ?? selectedFrame.id);
      setSelectedAnnotationID(null); setHistory([]); setFuture([]);
    } catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }
  async function skipCandidate() {
    if (!selectedFrame || selectedFrame.reviewStatus !== 'candidate') return;
    const next = candidates.find((frame) => frame.id !== selectedFrame.id);
    try { await post('/api/remove-frame', { frameId: selectedFrame.id }); setSelectedFrameID(next?.id ?? null); }
    catch (error) { setWorker((current) => current ? { ...current, error: String(error) } : current); }
  }

  function framePoint(clientX: number, clientY: number) {
    if (!selectedFrame || !svgRef.current) return { x: 0, y: 0 };
    const matrix = svgRef.current.getScreenCTM();
    if (!matrix) return { x: 0, y: 0 };
    const point = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
    return { x: clamp(point.x, 0, selectedFrame.width), y: clamp(point.y, 0, selectedFrame.height) };
  }

  function displayedAnnotations() {
    if (!selectedFrame) return [];
    if (gesture?.type === 'move' || gesture?.type === 'resize') return selectedFrame.annotations.map((item) => item.id === gesture.current.id ? gesture.current : item);
    return selectedFrame.annotations;
  }

  function startPointer(event: React.PointerEvent<SVGSVGElement>) {
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointers.current.size >= 2) {
      const values = [...pointers.current.values()].slice(0, 2);
      pinch.current = { distance: Math.hypot(values[1].x - values[0].x, values[1].y - values[0].y), centerX: (values[0].x + values[1].x) / 2, centerY: (values[0].y + values[1].y) / 2, zoom, panX: pan.x, panY: pan.y };
      setGesture(null); return;
    }
    if (mode === 'pan') setGesture({ type: 'pan', pointerId: event.pointerId, startClientX: event.clientX, startClientY: event.clientY, startPanX: pan.x, startPanY: pan.y });
    else if (mode === 'draw') { const point = framePoint(event.clientX, event.clientY); setSelectedAnnotationID(null); setGesture({ type: 'draw', pointerId: event.pointerId, startX: point.x, startY: point.y, x: point.x, y: point.y, width: 0, height: 0 }); }
    else setSelectedAnnotationID(null);
  }

  function movePointer(event: React.PointerEvent<SVGSVGElement>) {
    if (pointers.current.has(event.pointerId)) pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.current.size >= 2 && pinch.current) {
      const values = [...pointers.current.values()].slice(0, 2);
      const distance = Math.max(1, Math.hypot(values[1].x - values[0].x, values[1].y - values[0].y));
      const centerX = (values[0].x + values[1].x) / 2, centerY = (values[0].y + values[1].y) / 2;
      setZoom(clamp(pinch.current.zoom * distance / Math.max(1, pinch.current.distance), 1, 5));
      setPan({ x: pinch.current.panX + centerX - pinch.current.centerX, y: pinch.current.panY + centerY - pinch.current.centerY }); return;
    }
    if (!gesture || gesture.pointerId !== event.pointerId || !selectedFrame) return;
    if (gesture.type === 'pan') { setPan({ x: gesture.startPanX + event.clientX - gesture.startClientX, y: gesture.startPanY + event.clientY - gesture.startClientY }); return; }
    const point = framePoint(event.clientX, event.clientY);
    if (gesture.type === 'draw') setGesture({ ...gesture, x: Math.min(gesture.startX, point.x), y: Math.min(gesture.startY, point.y), width: Math.abs(point.x - gesture.startX), height: Math.abs(point.y - gesture.startY) });
    else if (gesture.type === 'move') {
      const dx = point.x - gesture.startX, dy = point.y - gesture.startY;
      setGesture({ ...gesture, current: { ...gesture.annotation, x: clamp(gesture.annotation.x + dx, 0, selectedFrame.width - gesture.annotation.width), y: clamp(gesture.annotation.y + dy, 0, selectedFrame.height - gesture.annotation.height), source: 'manual' } });
    } else {
      const original = gesture.annotation;
      let left = original.x, top = original.y, right = original.x + original.width, bottom = original.y + original.height;
      if (gesture.handle.includes('w')) left = clamp(point.x, 0, right - 3); if (gesture.handle.includes('e')) right = clamp(point.x, left + 3, selectedFrame.width);
      if (gesture.handle.includes('n')) top = clamp(point.y, 0, bottom - 3); if (gesture.handle.includes('s')) bottom = clamp(point.y, top + 3, selectedFrame.height);
      setGesture({ ...gesture, current: { ...original, x: left, y: top, width: right - left, height: bottom - top, source: 'manual' } });
    }
  }

  async function endPointer(event: React.PointerEvent<SVGSVGElement>) {
    pointers.current.delete(event.pointerId); if (pointers.current.size < 2) pinch.current = null;
    try { event.currentTarget.releasePointerCapture(event.pointerId); } catch { /* already released */ }
    if (!gesture || gesture.pointerId !== event.pointerId || pointers.current.size > 0) return;
    if (gesture.type === 'draw' && gesture.width >= 3 && gesture.height >= 3 && selectedFrame) await commitAnnotations([...selectedFrame.annotations, { id: crypto.randomUUID(), category, x: gesture.x, y: gesture.y, width: gesture.width, height: gesture.height, source: 'manual' }]);
    else if ((gesture.type === 'move' || gesture.type === 'resize') && selectedFrame) await commitAnnotations(selectedFrame.annotations.map((item) => item.id === gesture.current.id ? gesture.current : item));
    setGesture(null);
  }

  function startBox(event: React.PointerEvent<SVGGElement>, annotation: LocalAnnotation) {
    event.stopPropagation(); if (!svgRef.current || mode === 'draw') return;
    setSelectedAnnotationID(annotation.id); if (mode !== 'select') return;
    const point = framePoint(event.clientX, event.clientY); svgRef.current.setPointerCapture(event.pointerId); pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    setGesture({ type: 'move', pointerId: event.pointerId, annotation: { ...annotation }, startX: point.x, startY: point.y, current: { ...annotation } });
  }
  function startResize(event: React.PointerEvent<SVGRectElement>, annotation: LocalAnnotation, handle: Handle) {
    event.stopPropagation(); if (!svgRef.current) return; setSelectedAnnotationID(annotation.id); svgRef.current.setPointerCapture(event.pointerId); pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    setGesture({ type: 'resize', pointerId: event.pointerId, annotation: { ...annotation }, handle, current: { ...annotation } });
  }
  async function removeSelected() { if (!selectedFrame || !selectedAnnotationID) return; await commitAnnotations(selectedFrame.annotations.filter((item) => item.id !== selectedAnnotationID)); setSelectedAnnotationID(null); }
  async function acceptAutomatic() { if (selectedFrame) await commitAnnotations(selectedFrame.annotations.map((item) => item.source === 'auto' ? { ...item, source: 'manual' } : item)); }
  async function rejectAutomatic() { if (!selectedFrame) return; await commitAnnotations(selectedFrame.annotations.filter((item) => item.source !== 'auto')); setSelectedAnnotationID(null); }
  async function undo() { if (!selectedFrame || !history.length) return; const previous = history[history.length - 1]; setHistory((items) => items.slice(0, -1)); setFuture((items) => [...items, cloneAnnotations(selectedFrame.annotations)]); await commitAnnotations(previous, false); }
  async function redo() { if (!selectedFrame || !future.length) return; const next = future[future.length - 1]; setFuture((items) => items.slice(0, -1)); setHistory((items) => [...items, cloneAnnotations(selectedFrame.annotations)]); await commitAnnotations(next, false); }
  function chooseFrame(id: string) { setSelectedFrameID(id); setPan({ x: 0, y: 0 }); setZoom(1); }

  const shown = displayedAnnotations();
  const autoOnFrame = selectedFrame?.annotations.filter((item) => item.source === 'auto').length ?? 0;
  const stats = worker?.stats ?? { frameCount: frames.length, annotatedFrames: frames.filter((frame) => frame.annotations.length).length, annotationCount: frames.reduce((sum, frame) => sum + frame.annotations.length, 0), automaticCount: 0, classes: [] };
  const hardware = worker?.hardware;
  const hardwareText = hardware ? `${hardware.accelerator} · ${hardware.memoryGB ?? '?'} GB · ${hardware.cpuCores} CPU` : 'Lokale Hardware';
  const validationScore = worker?.lastTraining ? Object.entries(worker.lastTraining.validationMetrics ?? {}).find(([key]) => key.includes('mAP_50_95'))?.[1] : undefined;
  const benchmarkLabel = language === 'de' ? 'Modelle testen' : language === 'es' ? 'Probar modelos' : language === 'fr' ? 'Tester les modèles' : 'Test models';
  const trainingLabel = language === 'de' ? 'Training' : language === 'es' ? 'Entrenamiento' : language === 'fr' ? 'Entraînement' : 'Training';

  return <main className="preview-page">
    <header className="preview-header">
      <div className="brand-block"><div className="brand-mark" aria-hidden="true">R</div><div><h1>Reco Trainer <small>0.11</small></h1><p>{t.preview}</p></div></div>
      <div className="switches"><div className="segmented" role="group" aria-label="Plattform">{(['mac', 'windows', 'linux'] as Platform[]).map((item) => <button type="button" key={item} className={platform === item ? 'active' : ''} onClick={() => setPlatform(item)}>{item === 'mac' ? 'macOS' : item === 'windows' ? 'Windows' : 'Linux'}</button>)}</div><div className="segmented language-switch" role="group" aria-label="Sprache">{(['de', 'en', 'es', 'fr'] as Language[]).map((item) => <button type="button" key={item} className={language === item ? 'active' : ''} onClick={() => chooseLanguage(item)}>{item.toUpperCase()}</button>)}</div></div>
      <div className="header-actions"><button type="button" className={workspaceView === 'benchmark' ? 'walkthrough-launch active-view' : 'walkthrough-launch'} onClick={() => setWorkspaceView((view) => view === 'training' ? 'benchmark' : 'training')}>{workspaceView === 'training' ? `◇ ${benchmarkLabel}` : `← ${trainingLabel}`}</button><button type="button" className="walkthrough-launch" onClick={() => { setWalkthroughStep(0); setWalkthroughOpen(true); }}>? {t.help}</button><div className="local-note"><span>●</span>{t.note}</div></div>
    </header>
    <section className="preview-stage" aria-label={`${platform} Vorschau`}><div className={`app-window platform-${platform}`}><div className="app-body">
      <aside className="app-sidebar">
        <section className="side-section"><label>1 · {t.sport}</label><select aria-label={t.sport} value={sport} onChange={(event) => { const next = event.target.value as Sport; setSport(next); setCategory(next === 'hockey' ? 'puck' : 'ball'); }}><option value="basketball">Basketball</option><option value="football">{t.football}</option><option value="futsal">Futsal</option><option value="handball">Handball</option><option value="hockey">Hockey</option><option value="rugby">Rugby</option><option value="lacrosse">Lacrosse</option><option value="american_football">{t.americanFootball}</option></select>{platform !== 'mac' && <small className="preview-only">{t.previewOnly}</small>}</section>
        <section className="side-section"><label>2 · {t.videos}</label><button type="button" className="secondary-button folder-button" onClick={chooseFolder} disabled={!workerOnline || isWorking}>▣&nbsp; {t.choose}</button><span className={folderName ? 'fake-path selected' : 'fake-path'}>{worker?.selectedFolder || t.noFolder}</span><label className="frame-count-label" htmlFor="frames-per-video">{ft.count}</label><input id="frames-per-video" className="frame-count-input" type="number" min="4" max="5000" step="10" value={framesPerVideo} disabled={isWorking} onChange={(event) => setFramesPerVideo(clamp(Number(event.target.value) || 4, 4, 5000))} /><small className="frame-count-hint">{ft.countHint}</small><button type="button" className="primary-button" onClick={preparePreview} disabled={!folderName || isWorking}>{isWorking && ['scanning', 'extracting'].includes(worker?.operation ?? '') ? t.analyzing : framesReady ? t.refresh : t.analyze}</button>{folderName && <small className="folder-safety">{t.folderReady}</small>}{!workerOnline && <small className="worker-offline">{t.offline}</small>}</section>
        <section className="side-section active-learning-section"><label>{at.title}</label><button type="button" className="secondary-button" onClick={() => void startActiveLearning()} disabled={!framesReady || isWorking}>＋ {at.start}</button><small>{at.detail}</small>{candidates.length > 0 && <strong>{candidates.length} {at.queue}</strong>}</section>
        <section className="side-section frames-section"><label>3 · {t.images}</label><div className="frame-list">{framesReady ? frames.map((frame) => <button type="button" className={`${selectedFrame?.id === frame.id ? 'frame active' : 'frame'} ${frame.reviewStatus === 'candidate' ? 'candidate' : ''}`} key={frame.id} onClick={() => chooseFrame(frame.id)}><img className="frame-thumb" src={`${api}/api/frame?id=${encodeURIComponent(frame.id)}`} alt="" loading="lazy" decoding="async" /><span>{frame.videoName}<small>{String(Math.floor(frame.timestamp / 60)).padStart(2, '0')}:{String(Math.floor(frame.timestamp % 60)).padStart(2, '0')} · {frame.annotations.length}{frame.reviewStatus === 'candidate' ? ` · ${at.queue}` : ''}</small></span></button>) : <p className="empty-frames">{worker?.message || t.emptyFrames}</p>}</div></section>
        <div className="privacy-lock"><span>●</span> {t.privacy}</div>
      </aside>
      {workspaceView === 'benchmark' ? <BenchmarkPanel language={language} busy={isWorking} operation={worker?.operation} progress={worker?.progress ?? 0} message={worker?.message} error={worker?.error} log={worker?.log} frameCount={stats.frameCount} automaticCount={stats.automaticCount} sport={worker?.sport ?? sport} models={worker?.modelLibrary?.packages ?? []} benchmark={worker?.benchmark} post={post} /> : <section className={`workspace ${framesReady ? '' : 'empty-workspace'}`}>
        {framesReady && <div className="annotation-toolbar"><div className="tool-group"><button type="button" className={mode === 'select' ? 'active' : ''} onClick={() => setMode('select')}>⌖ {t.select}</button><button type="button" className={mode === 'draw' ? 'active' : ''} onClick={() => setMode('draw')}>＋ {t.draw}</button><button type="button" className={mode === 'pan' ? 'active' : ''} onClick={() => setMode('pan')}>↔ {t.pan}</button></div><label htmlFor="category">{t.annotate}</label><select id="category" value={category} onChange={(event) => setCategory(event.target.value)}>{categories.map((item) => <option value={item} key={item}>{categoryName(item)}</option>)}</select><div className="edit-actions"><button type="button" onClick={() => void undo()} disabled={!history.length} title={t.undo}>↶</button><button type="button" onClick={() => void redo()} disabled={!future.length} title={t.redo}>↷</button><button type="button" onClick={() => void removeSelected()} disabled={!selectedAnnotation}>{t.remove}</button>{autoOnFrame > 0 && <><button type="button" className="accept" onClick={() => void acceptAutomatic()}>{t.accept}</button><button type="button" className="reject" onClick={() => void rejectAutomatic()}>{t.reject}</button></>}</div></div>}
        <div className={`editor-shell ${framesReady ? 'video-ready' : ''}`}>
          {selectedFrame ? <div className="local-frame-viewport" onWheel={(event) => { event.preventDefault(); setZoom((value) => clamp(value * (event.deltaY < 0 ? 1.12 : .89), 1, 5)); }}><div className="frame-transform" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}><img className="local-frame-image" src={`${api}/api/frame?id=${encodeURIComponent(selectedFrame.id)}`} alt={`${selectedFrame.videoName} ${selectedFrame.timestamp.toFixed(1)}s`} decoding="async" draggable={false} /><svg ref={svgRef} className={`local-frame-canvas mode-${mode}`} viewBox={`0 0 ${selectedFrame.width} ${selectedFrame.height}`} preserveAspectRatio="xMidYMid meet" aria-label={`${selectedFrame.videoName} ${selectedFrame.timestamp.toFixed(1)}s`} onPointerDown={startPointer} onPointerMove={movePointer} onPointerUp={(event) => void endPointer(event)} onPointerCancel={(event) => void endPointer(event)}>
            {shown.map((box) => <g key={box.id} className={`annotation-box ${box.source === 'auto' ? 'automatic' : ''} ${selectedAnnotationID === box.id ? 'selected' : ''}`} onPointerDown={(event) => startBox(event, box)}><rect x={box.x} y={box.y} width={box.width} height={box.height} /><text x={box.x} y={Math.max(14, box.y - 5)}>{categoryName(box.category)}{box.confidence != null ? ` ${Math.round(box.confidence * 100)}%` : ''}</text>{selectedAnnotationID === box.id && (['nw', 'ne', 'sw', 'se'] as Handle[]).map((handle) => { const size = 12 / zoom; const cx = handle.includes('w') ? box.x : box.x + box.width; const cy = handle.includes('n') ? box.y : box.y + box.height; return <rect key={handle} className={`resize-handle handle-${handle}`} x={cx - size / 2} y={cy - size / 2} width={size} height={size} onPointerDown={(event) => startResize(event, box, handle)} />; })}</g>)}
            {gesture?.type === 'draw' && <rect className="draft-box" x={gesture.x} y={gesture.y} width={gesture.width} height={gesture.height} />}
          </svg></div></div> : <div className="empty-video-viewport"><strong>{sportName}</strong></div>}
          {framesReady && <div className="editor-controls"><button type="button" onClick={() => frames[selectedFrameIndex - 1] && chooseFrame(frames[selectedFrameIndex - 1].id)} disabled={!frames[selectedFrameIndex - 1]}>←</button><strong>{selectedFrameIndex + 1} / {frames.length}</strong><button type="button" onClick={() => frames[selectedFrameIndex + 1] && chooseFrame(frames[selectedFrameIndex + 1].id)} disabled={!frames[selectedFrameIndex + 1]}>→</button><span className="instruction">{t.drag}</span><button type="button" className="remove-frame" onClick={() => void removeFrame()} disabled={isWorking}>⌫ {ft.remove}</button><div className="zoom-controls"><button type="button" onClick={() => setZoom((value) => clamp(value - .25, 1, 5))}>−</button><strong>{zoom.toFixed(2)}×</strong><button type="button" onClick={() => setZoom((value) => clamp(value + .25, 1, 5))}>+</button><button type="button" className="reset" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>{t.reset}</button></div></div>}
        </div>
        {selectedFrame?.reviewStatus === 'candidate' && <section className="candidate-review"><div><strong>{at.queue}: {candidates.indexOf(selectedFrame) + 1} / {candidates.length}</strong><small>{at.needsBox} {at.hint}</small></div><button type="button" className="candidate-yes" onClick={() => void reviewCandidate('ball')}>✓ {at.ball}</button><button type="button" className="candidate-no" onClick={() => void reviewCandidate('no-ball')}>× {at.noBall}</button><button type="button" onClick={() => setMode('draw')}>⌖ {at.correct}</button><button type="button" onClick={() => void skipCandidate()}>{at.skip}</button></section>}
        {framesReady && <section className="training-card">
          <div className="training-topline"><div><strong>4 · {t.improve}</strong><select className="model-select" aria-label="Modellgröße" value={modelSize} onChange={(event) => setModelSize(event.target.value as ModelSize)}><option value="nano">RF-DETR Nano</option><option value="small">RF-DETR Small</option></select></div><span className="hardware-pill">◆ {hardwareText}</span></div>
          <div className="dataset-summary"><span><strong>{stats.annotatedFrames}/{stats.frameCount}</strong>{t.marked}</span><span><strong>{stats.annotationCount}</strong>{t.boxes}</span><span className={stats.automaticCount ? 'attention' : ''}><strong>{stats.automaticCount}</strong>{t.pending}</span><span><strong>{worker?.lastTraining ? `${worker.lastTraining.model} · ${worker.lastTraining.device.toUpperCase()}${validationScore != null ? ` · ${Math.round(validationScore * 1000) / 10}% mAP` : ''}` : '–'}</strong>{worker?.lastTraining ? `${t.validation} · ${worker.lastTraining.independentTest ? t.testGood : t.testWeak}` : t.noTraining}</span></div>
          <div className="training-controls"><label>{t.confidence} <strong>{Math.round(threshold * 100)}%</strong><input type="range" min="0.05" max="0.90" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label><div className="training-actions"><button type="button" className="secondary-button" onClick={() => void runML('/api/setup')} disabled={isWorking}>{t.setup}</button><button type="button" className="secondary-button" onClick={() => void runML('/api/autolabel')} disabled={isWorking}>{t.auto}: {categoryName(category)}</button><button type="button" className="primary-button" onClick={() => void runML('/api/train')} disabled={isWorking}>{t.train}</button><button type="button" className="secondary-button" onClick={() => void runML('/api/export-coreml')} disabled={isWorking}>{t.export}</button><button type="button" className="secondary-button" onClick={() => void runML('/api/package-model')} disabled={isWorking}>{t.share}</button><button type="button" className="secondary-button" onClick={() => void runML('/api/import-model')} disabled={isWorking || !folderName}>{t.importModel}</button></div></div>
          <div className="exchange-status"><strong>{t.activeModel}:</strong> {activeModel ? `${activeModel.packageID} · ${activeModel.modelSize.toUpperCase()} · ${activeModel.classes.map(categoryName).join(', ')}` : t.noModel}{activeModel?.description && <span className="package-description">{activeModel.description}</span>}<small>{t.trustWarning}</small></div>
          <section className="model-library"><strong>{modelUi.title}</strong><div className="model-library-list">{(worker?.modelLibrary?.packages ?? []).map((model) => <article key={model.packageID} className={model.isActive ? 'active' : ''}><div><b>{model.displayName || model.packageID}</b><small>{model.modelSize.toUpperCase()} · {model.testScore != null ? `Test mAP ${model.testScore.toFixed(4)}` : model.validationScore != null ? `Val mAP ${model.validationScore.toFixed(4)}` : 'mAP –'}</small><code>{model.packageID}</code></div><span>{model.isActive && <em>{modelUi.active}</em>}{model.isBest && <em>{modelUi.best}</em>}</span><button type="button" disabled={isWorking || model.isActive} onClick={() => void post('/api/activate-model', { packageID: model.packageID, language })}>{modelUi.activate}</button><button type="button" disabled={isWorking} onClick={() => { const name = window.prompt(modelUi.prompt, model.displayName || ''); if (name?.trim()) void post('/api/rename-model', { packageID: model.packageID, name: name.trim(), language }); }}>{modelUi.rename}</button><button type="button" className="danger" disabled={isWorking || model.isActive} onClick={() => { if (window.confirm(modelUi.confirm)) void post('/api/delete-model', { packageID: model.packageID, language }); }}>{modelUi.remove}</button></article>)}</div></section>
          <div className="progress-row"><span style={{ width: `${Math.max(worker?.progress ?? 0, .02) * 100}%` }} /><i>{worker?.error || worker?.message || t.localReady}</i></div>{worker?.log && <details className="worker-details"><summary>Protokoll / Log</summary><pre className="worker-log">{worker.log}</pre></details>}
        </section>}
      </section>}
    </div></div></section>
    {languagePromptOpen && <div className="walkthrough-backdrop language-backdrop"><section className="language-dialog" role="dialog" aria-modal="true" aria-labelledby="language-title"><div className="language-globe" aria-hidden="true">文</div><h2 id="language-title">Sprache wählen · Choose your language</h2><p>Elige tu idioma · Choisissez votre langue</p><div className="language-choices">{([['de', 'Deutsch'], ['en', 'English'], ['es', 'Español'], ['fr', 'Français']] as [Language, string][]).map(([code, label]) => <button type="button" key={code} onClick={() => chooseLanguage(code)}><strong>{label}</strong><span>{code.toUpperCase()} <b aria-hidden="true">→</b></span></button>)}</div><small>Videos and images always remain on this computer.</small></section></div>}
    {walkthroughOpen && !languagePromptOpen && <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setWalkthroughOpen(false); }}><section className="walkthrough-dialog" role="dialog" aria-modal="true" aria-labelledby="walkthrough-title"><button type="button" className="walkthrough-close" aria-label={t.close} onClick={() => setWalkthroughOpen(false)}>×</button><div className="walkthrough-kicker">{t.step} {walkthroughStep + 1} / {t.walkthrough.length}</div><h2 id="walkthrough-title">{t.walkthroughTitle}</h2><div className="walkthrough-progress" aria-hidden="true">{t.walkthrough.map((_, index) => <span key={index} className={index <= walkthroughStep ? 'active' : ''} />)}</div><article><div className="walkthrough-number">{walkthroughStep + 1}</div><div><h3>{t.walkthrough[walkthroughStep][0]}</h3><p>{t.walkthrough[walkthroughStep][1]}</p></div></article><footer><button type="button" className="secondary-button" onClick={() => setWalkthroughStep((value) => Math.max(0, value - 1))} disabled={walkthroughStep === 0}>{t.back}</button><button type="button" className="primary-button walkthrough-next" onClick={() => { if (walkthroughStep < t.walkthrough.length - 1) setWalkthroughStep((value) => value + 1); else { window.localStorage.setItem('reco-walkthrough-complete-v2', 'yes'); setWalkthroughOpen(false); } }}><span>{walkthroughStep < t.walkthrough.length - 1 ? t.next : t.done}</span><b className="forward-arrow" aria-hidden="true">{walkthroughStep < t.walkthrough.length - 1 ? '→' : '✓'}</b></button></footer></section></div>}
  </main>;
}
